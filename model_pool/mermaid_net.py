from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

from model_pool.mermaid_call import _get_low_res_size_from_size, _get_low_res_spacing_from_spacing, \
    _compute_low_res_image
from model_pool.modules import *
from functions.bilinear import *
from model_pool.reg_net_expr import *
from model_pool.utils import sigmoid_explode, get_inverse_affine_param, get_warped_img_map_param, update_affine_param
from models.net_utils import init_weights
import mermaid.pyreg.module_parameters as pars
import mermaid.pyreg.model_factory as py_mf
import mermaid.pyreg.utils as py_utils
from functools import partial
import mermaid.pyreg.image_sampling as py_is
from mermaid.pyreg.libraries.functions.stn_nd import STNFunction_ND_BCXYZ
from model_pool.global_variable import *


class MermaidNet(nn.Module):
    """
    this network is an end to end system for momentum generation and mermaid registration
    include the following parts

    1 . affine net the affine network is used to affine the source and target image (though affine is optional, we currently put it here)
    2. the  momentum generation net work, this network should be an auto-encoder system like the Xiao's work
    3. the mermaid part, an single optimizer would be called from the mermaid code

    In detail of implementation, we should take care of the memory issue, one possible solution is using low-resolution mapping

    1. affinenet work, this is a pretrained network, so the only the forward flow is used, the input should be set as volatile,
        in current  design, the input and output of this net is of the full resolution
    2. momentum generation net, this is a trainable network, but we would have a low-res factor to train it at a low-resolution
        the input may still at original resolution, but the output size may be determined by the low-res factor

    3. mermaid part, this is an shooting unit, where should call the single-scale optimizer, and the output should be upsampled to the
        full resolution size.

    so the input and the output of each part should be

    1. affine: input: st_concated, s,   output: s_warped, affine_map
    2. momentum: input: st_concated , low_res_factor  output: m
    3. mermaid: input: s,t,m, low_res_factor  output: s_warped


    """

    def __init__(self, img_sz=None, opt=None):
        super(MermaidNet, self).__init__()
        self.load_external_model = False
        self.intra_training = intra_training


        cur_gpu_id = opt['tsk_set']['gpu_ids']
        old_gpu_id = opt['tsk_set']['old_gpu_ids']
        low_res_factor = opt['tsk_set']['reg'][('low_res_factor',1.,"factor of low-resolution map")]
        batch_sz = opt['tsk_set']['batch_sz']
        self.loss_type = opt['tsk_set']['loss']['type']
        self.img_sz = [batch_sz, 1] + img_sz
        self.dim = len(img_sz)
        self.gpu_switcher = (cur_gpu_id, old_gpu_id)
        self.low_res_factor = low_res_factor
        self.using_sym_on = use_sym
        self.using_analyic_af_inverse = using_analyic_af_inverse
        self.sym_factor = 1.
        self.momentum_net = MomentumNet(low_res_factor)
        self.init_affine_net()
        self.step = 1 if not use_mermaid_multi_step else 6

        spacing = 1. / (np.array(img_sz) - 1)
        self.spacing = spacing
        self.mermaid_unit = None
        self.init_mermaid_env(spacing)
        self.bilinear = Bilinear()
        self.debug_count = 0
        self.affined_img = None

    def init_affine_net(self):
        self.affine_net = AffineNetCycle(self.img_sz[2:])

        if self.load_external_model:
            model_path = '/playpen/zyshen/data/reg_debug_3000_pair_oai_reg_intra/train_mermaid_net_reisd_2step_lncc_recbi/checkpoints/epoch_270_'
            checkpoint = torch.load(model_path,  map_location='cpu')
            self.load_state_dict(checkpoint['state_dict'])
            self.cuda()
            print("Attention, the external model is loaded !!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.affine_net.eval()

    def init_mermaid_env(self, spacing):
        params = pars.ParameterDict()
        if not use_llddmm:
            params.load_JSON( '../mermaid/demos/cur_settings_lbfgs_debug.json') #''../model_pool/cur_settings_svf.json')######TODO ###########
        else:
            params.load_JSON( '../mermaid/demos/cur_settings_lbfgs_forlddmm.json') #''../model_pool/cur_settings_svf.json')
        model_name = params['model']['registration_model']['type']
        use_map = params['model']['deformation']['use_map']
        compute_similarity_measure_at_low_res = params['model']['deformation'][
            ('compute_similarity_measure_at_low_res', False, 'to compute Sim at lower resolution')]
        params['model']['registration_model']['similarity_measure']['type'] =self.loss_type
        params.print_settings_off()

        self.mermaid_low_res_factor = self.low_res_factor

        lowResSize = None
        lowResSpacing = None
        ##
        if self.mermaid_low_res_factor == 1.0 or self.mermaid_low_res_factor == [1., 1., 1.]:
            self.mermaid_low_res_factor = None
        ##
        if self.mermaid_low_res_factor is not None:
            lowResSize = _get_low_res_size_from_size(self.img_sz, self.mermaid_low_res_factor)
            lowResSpacing = _get_low_res_spacing_from_spacing(spacing, self.img_sz, lowResSize)
            self.lowResSpacing = lowResSpacing
            self.lowRes_fn = partial(_compute_low_res_image, spacing=spacing, low_res_size=lowResSize)

        if self.mermaid_low_res_factor is not None:
            # computes model at a lower resolution than the image similarity
            if compute_similarity_measure_at_low_res:
                mf = py_mf.ModelFactory(lowResSize, lowResSpacing, lowResSize, lowResSpacing)
            else:
                mf = py_mf.ModelFactory(self.img_sz, spacing, lowResSize, lowResSpacing)
        else:
            # computes model and similarity at the same resolution
            mf = py_mf.ModelFactory(self.img_sz, spacing, self.img_sz, spacing)

        model, criterion = mf.create_registration_model(model_name, params['model'], compute_inverse_map=False)
        # model.eval()

        if use_map:
            # create the identity map [0,1]^d, since we will use a map-based implementation
            _id = py_utils.identity_map_multiN(self.img_sz, spacing)
            self.identityMap = torch.from_numpy(_id).cuda()
            if self.mermaid_low_res_factor is not None:
                # create a lower resolution map for the computations
                lowres_id = py_utils.identity_map_multiN(lowResSize, lowResSpacing)
                self.lowResIdentityMap = torch.from_numpy(lowres_id).cuda()
                print(torch.min(self.lowResIdentityMap))

        self.mermaid_unit = model.cuda()
        self.criterion = criterion

    def __cal_sym_loss(self,source=None):
        trans1 = STNFunction_ND_BCXYZ(self.spacing,zero_boundary=False)
        trans2 = STNFunction_ND_BCXYZ(self.spacing,zero_boundary=False)
        identity_map  = self.identityMap.expand_as(self.rec_phiWarped[0])
        trans_st  = trans1(identity_map,self.rec_phiWarped[0])
        trans_st_ts = trans2(trans_st,self.rec_phiWarped[1])
        return torch.mean((identity_map- trans_st_ts)**2)



    def do_criterion_cal(self, ISource, ITarget,cur_epoch=-1):
        ISource = (ISource + 1.) / 2.
        ITarget = (ITarget + 1.) / 2.
        if not self.using_sym_on:
            loss_overall_energy, sim_energy, reg_energy = self.criterion(self.identityMap, self.rec_phiWarped, ISource,
                                                                         ITarget, None,
                                                                         self.mermaid_unit.get_variables_to_transfer_to_loss_function(),
                                                                         None)
            if self.debug_count % 10 == 0:
                print('the loss_over_all:{} sim_energy:{}, reg_energy:{}\n'.format(loss_overall_energy.item(),
                                                                                   sim_energy.item(),
                                                                                   reg_energy.item()))
        else:
            loss_overall_energy_st, sim_energy_st, reg_energy_st = self.criterion(self.identityMap, self.rec_phiWarped[0], ISource,
                                                                         ITarget, None,
                                                                         self.mermaid_unit.get_variables_to_transfer_to_loss_function(),
                                                                         None)
            loss_overall_energy_ts, sim_energy_ts, reg_energy_ts = self.criterion(self.identityMap,
                                                                                  self.rec_phiWarped[1], ITarget,
                                                                                  ISource, None,
                                                                                  self.mermaid_unit.get_variables_to_transfer_to_loss_function(),
                                                                                  None)
            loss_overall_energy = (loss_overall_energy_st + loss_overall_energy_ts)/2
            sim_energy =  (sim_energy_st + sim_energy_ts)/2
            reg_energy = (reg_energy_st + reg_energy_ts)/2
            sym_energy = self.__cal_sym_loss()
            sym_factor = min(sigmoid_explode(cur_epoch,static=1, k=8)*0.01*gl_sym_factor,1.*gl_sym_factor) #static=5, k=4)*0.01,1) static=10, k=10)*0.01
            loss_overall_energy = loss_overall_energy + sym_factor*sym_energy
            if self.debug_count % 10 == 0:
                print('the loss_over_all:{} sim_energy:{},sym_factor: {} sym_energy: {} reg_energy:{}\n'.format(loss_overall_energy.item(),
                                                                                   sim_energy.item(),
                                                                                    sym_factor,
                                                                                     sym_energy.item(),
                                                                                   reg_energy.item()))



        self.debug_count += 1
        return loss_overall_energy, sim_energy, reg_energy

    def set_mermaid_param(self, s, t, m):
        self.mermaid_unit.set_dictionary_to_pass_to_integrator({'I0': s, 'I1': t})
        self.mermaid_unit.m = m
        self.criterion.m = m

    def do_mermaid_reg(self, s, t, m, phi):
        if self.mermaid_low_res_factor is not None:
            self.set_mermaid_param(s, t, m)
            maps = self.mermaid_unit(self.lowRes_fn(phi), s)
            # now up-sample to correct resolution
            desiredSz = self.img_sz[2:]
            sampler = py_is.ResampleImage()
            rec_phiWarped, _ = sampler.upsample_image_to_size(maps, self.lowResSpacing, desiredSz, 1,zero_boundary=False)

        else:
            self.set_mermaid_param(s, t, m)
            maps = self.mermaid_unit(self.identityMap, s)
            rec_phiWarped = maps
        rec_IWarped = py_utils.compute_warped_image_multiNC(s, rec_phiWarped, self.spacing, 1,zero_boundary=True)
        self.rec_phiWarped = rec_phiWarped

        return rec_IWarped, rec_phiWarped


    def single_forward(self,input,moving,target=None):
        with torch.no_grad():
            affine_img, affine_map, _ = self.affine_net(input, moving, target)
        input = torch.cat((affine_img, target), 1)
        m = self.momentum_net(input)
        # m.register_hook(print)
        # self.momentum_net.register_backward_hook(bh)
        moving = (moving + 1) / 2.
        target = (target + 1) / 2.
        affine_map = (affine_map + 1) / 2.
        rec_IWarped, rec_phiWarped = self.do_mermaid_reg(moving, target, m, affine_map)
        return (rec_IWarped*2.-1.).detach(), (rec_phiWarped*2.-1.).detach(), affine_img.detach()



    def sym_forward(self,input,moving,target=None):
        #raise("not implemented yet")
        with torch.no_grad():
            affine_img_st, affine_map_st, affine_param = self.affine_net(input, moving, target)
            if not self.using_analyic_af_inverse:
                affine_img_ts, affine_map_ts, _ = self.affine_net(input, target, moving)
            else:
                affine_inverse = get_inverse_affine_param(affine_param)
                affine_img_ts,affine_map_ts, _ = get_warped_img_map_param(affine_inverse,self.img_sz[2:],target)

        input_st = torch.cat((affine_img_st, target), 1)
        input_ts = torch.cat((affine_img_ts, moving), 1)
        m_st = self.momentum_net(input_st)
        m_ts = self.momentum_net(input_ts)
        moving = (moving + 1) / 2.
        target = (target + 1) / 2.
        affine_map_st = (affine_map_st + 1) / 2.
        affine_map_ts = (affine_map_ts + 1) / 2.
        rec_IWarped_st, rec_phiWarped_st = self.do_mermaid_reg(moving, target, m_st, affine_map_st)
        rec_IWarped_ts, rec_phiWarped_ts = self.do_mermaid_reg(target, moving, m_ts, affine_map_ts)
        self.rec_phiWarped = (rec_phiWarped_st,rec_phiWarped_ts)
        return (rec_IWarped_st * 2. - 1.).detach_(), (rec_phiWarped_st * 2. - 1.).detach_(), affine_img_st.detach_()

    def cyc_forward(self,input,moving,target=None):

        with torch.no_grad():
            affine_img, affine_map, _ = self.affine_net(input, moving, target)
            moving_n = (moving + 1) / 2.  # [-1,1] ->[0,1]
            target_n = (target + 1) / 2.  # [-1,1] ->[0,1]
            affine_map = (affine_map + 1) / 2.  # [-1,1] ->[0,1]

        warped_img =affine_img
        init_map = affine_map

        for _ in range(self.step):
            input = torch.cat((warped_img, target), 1)
            m = self.momentum_net(input)
            rec_IWarped, rec_phiWarped = self.do_mermaid_reg(moving_n, target_n, m, init_map)
            warped_img = rec_IWarped*2-1 #[0,1] -> [-1,1]
            init_map = rec_phiWarped #[0,1]

        return (rec_IWarped*2.-1.).detach(), (rec_phiWarped*2.-1.).detach(), affine_img.detach()


    def cyc_sym_forward(self,input,moving, target= None):
        with torch.no_grad():
            affine_img_st, affine_map_st, _ = self.affine_net(input, moving, target)
            affine_img_ts, affine_map_ts, _ = self.affine_net(input, target, moving)
            moving_n = (moving + 1) / 2.  # [-1,1] ->[0,1]
            target_n = (target + 1) / 2.  # [-1,1] ->[0,1]
            affine_map_st = (affine_map_st + 1) / 2.  # [-1,1] ->[0,1]
            affine_map_ts = (affine_map_ts + 1) / 2.  # [-1,1] ->[0,1]

        warped_img_st = affine_img_st
        init_map_st = affine_map_st
        warped_img_ts = affine_img_ts
        init_map_ts = affine_map_ts

        for _ in range(self.step):
            input_st = torch.cat((warped_img_st, target), 1)
            input_ts = torch.cat((warped_img_ts, moving), 1)
            m_st = self.momentum_net(input_st)
            m_ts = self.momentum_net(input_ts)
            rec_IWarped_st, rec_phiWarped_st = self.do_mermaid_reg(moving_n, target_n, m_st, init_map_st)
            rec_IWarped_ts, rec_phiWarped_ts = self.do_mermaid_reg(target_n, moving_n, m_ts, init_map_ts)
            warped_img_st = rec_IWarped_st * 2 - 1  # [0,1] -> [-1,1]
            init_map_st = rec_phiWarped_st  # [0,1]
            warped_img_ts = rec_IWarped_ts * 2 - 1
            init_map_ts = rec_phiWarped_ts
        self.rec_phiWarped = (rec_phiWarped_st,rec_phiWarped_ts)

        return warped_img_st.detach(), (rec_phiWarped_st * 2. - 1.).detach(), affine_img_st.detach()

    def get_affine_map(self,moving, target):
        with torch.no_grad():
            affine_img, affine_map, _ = self.affine_net(input, moving, target)
        return affine_map


    def forward(self, input, moving, target=None):

        if use_mermaid_multi_step and self.using_sym_on:
            return self.cyc_sym_forward(input,moving,target)
        if use_mermaid_multi_step:
            return self.cyc_forward(input, moving, target)
        if not self.using_sym_on:
            return self.single_forward(input,moving, target)
        else:
            return self.sym_forward(input,moving,target)


def bh(m, gi, go):
    print("Grad Input")
    print((torch.sum(gi[0].data), torch.sum(gi[1].data)))
    print("Grad Output")
    print(torch.sum(go[0].data))
    return gi[0], gi[1], gi[2]

