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
from models.net_utils import init_weights
import mermaid.pyreg.module_parameters as pars
import mermaid.pyreg.model_factory as py_mf
import mermaid.pyreg.utils as py_utils
from functools import partial
import mermaid.pyreg.image_sampling as py_is



class MermaidNet(nn.Module):
    """
    this network is an end to end system for momentum generation and mermaid registration
    include the following parts

    1 . affine net the affine network is used to affine the source and target image (though this is optional, we currently put it here)
    2. the  momentum generation net work, this network should be an auto-encoder system like the Xiao's work
    3. the mermaid part, an single optimizer would be called from the mermaid code

    In detail of implementation, we should take care of the memory issue, one possible solution is using low-resolution mapping

    1. affinenet work, this is a pretrained network, so the only the forward flow is used, the input should be set as volatile,
        in current  design, the input and output of this net is of the original size
    2. momentum generation net, this is a trainable network, but we would have a low-res factor to make it train at a low-resolution
        the input may still at original resolution, but the output size may be determined by the low-res factor

    3. mermaid part, this is an inference unit, where should call the single-scale optimizer, and the output should be upsampled to the
        original size.

    so the input and the output of each part should be

    1. affine: input: st_concated, s,   output: s_warped -> s
    2. momentum: input: st_concated , low_res_factor  output: m
    3. mermaid: input: s,t,m, low_res_factor  output: s_warped


    """
    def __init__(self, img_sz=None, opt=None):
        super(MermaidNet, self).__init__()
        cur_gpu_id = opt['tsk_set']['gpu_ids']
        old_gpu_id = opt['tsk_set']['old_gpu_ids']
        low_res_factor = opt['tsk_set']['low_res_factor']
        batch_sz = opt['tsk_set']['batch_sz']
        self.img_sz = [batch_sz,1]+img_sz
        self.dim = len(img_sz)
        self.gpu_switcher = (cur_gpu_id,old_gpu_id)
        self.init_affine_net()
        self.low_res_factor = low_res_factor
        self.momentum_net = MomentumNet(low_res_factor)

        spacing = 1./(np.array(img_sz)-1)
        self.spacing = spacing
        self.mermaid_unit = None
        self.init_mermaid_env(spacing)
        self.bilinear = Bilinear()
        self.debug_count = 0
        self.affined_img = None

    def init_affine_net(self):
        model_path ="/playpen/zyshen/data/reg_debug_oai_reg_intra/reg_affine_cycle_debug/checkpoints/epoch_360_"
        checkpoint = torch.load(model_path,
                                map_location={'cuda:' + str(self.gpu_switcher[0]): 'cuda:' + str(self.gpu_switcher[1])})
        self.affine_net = AffineNetCycle(self.img_sz[2:])
        self.affine_net.load_state_dict(checkpoint['state_dict'])
        print("Affine network successfully initialized,{}".format(model_path))
        print(self.affine_net)
        self.affine_net.eval()



    def init_mermaid_env(self, spacing):
        params = pars.ParameterDict()
        params.load_JSON('../mermaid/demos/cur_settings_svf.json')
        model_name = params['model']['registration_model']['type']
        use_map = params['model']['deformation']['use_map']
        compute_similarity_measure_at_low_res = params['model']['deformation'][
            ('compute_similarity_measure_at_low_res', False, 'to compute Sim at lower resolution')]
        self.mermaid_low_res_factor = self.low_res_factor

        lowResSize = None
        lowResSpacing = None
        ##
        if self.mermaid_low_res_factor == 1.0 or self.mermaid_low_res_factor==[1.,1.,1.]:
            self.mermaid_low_res_factor = None
        ##
        if self.mermaid_low_res_factor is not None:
            lowResSize = _get_low_res_size_from_size(self.img_sz, self.mermaid_low_res_factor)
            lowResSpacing = _get_low_res_spacing_from_spacing(spacing, self.img_sz, lowResSize)
            self.lowResSpacing = lowResSpacing
            self.lowRes_fn = partial(_compute_low_res_image,spacing=spacing,low_res_size=lowResSize)

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
        #model.eval()

        if use_map:
            # create the identity map [-1,1]^d, since we will use a map-based implementation
            _id = py_utils.identity_map_multiN(self.img_sz, spacing)
            self.identityMap = Variable(torch.from_numpy(_id), requires_grad=False).cuda()
            if self.mermaid_low_res_factor is not None:
                # create a lower resolution map for the computations
                lowres_id = py_utils.identity_map_multiN(lowResSize, lowResSpacing)
                self.lowResIdentityMap = Variable(torch.from_numpy(lowres_id), requires_grad=False).cuda()

        self.mermaid_unit = model.cuda()
        self.criterion = criterion

    def do_criterion_cal(self,ISource, ITarget):
        ISource = (ISource+1.)/2.
        ITarget = (ITarget+1.)/2.
        loss_overall_energy, sim_energy, reg_energy = self.criterion(self.identityMap, self.rec_phiWarped, ISource,
                                                                     ITarget, None,
                                                                     self.mermaid_unit.get_variables_to_transfer_to_loss_function(),
                                                                     None)
        if self.debug_count%10 ==0:
            tofloat = lambda x: x.data.cpu().numpy()[0]
            print('the loss_over_all:{} sim_energy:{}, reg_energy:{}\n'.format(tofloat(loss_overall_energy), tofloat(sim_energy), tofloat(reg_energy)))
        self.debug_count += 1
        return loss_overall_energy, sim_energy, reg_energy



    def set_mermaid_param(self,s,t,m):
        self.mermaid_unit.set_dictionary_to_pass_to_integrator({'I0':s,'I1':t})
        self.mermaid_unit.m  =m
        self.criterion.m = m
        
    def do_mermaid_reg(self,s,t,m,phi):
        if self.mermaid_low_res_factor is not None:
            self.set_mermaid_param(s, t, m)
            maps = self.mermaid_unit(self.lowRes_fn(phi), s)
            # now up-sample to correct resolution
            desiredSz = self.img_sz[2:]
            sampler = py_is.ResampleImage()
            rec_phiWarped, _ = sampler.upsample_image_to_size(maps, self.lowResSpacing, desiredSz,1)
    
        else:
            self.set_mermaid_param(s, t, m)
            maps = self.mermaid_unit(self.identityMap, s)
            rec_phiWarped = maps
        rec_IWarped = py_utils.compute_warped_image_multiNC(s, rec_phiWarped, self.spacing, 1)
        self.rec_phiWarped = rec_phiWarped

        return rec_IWarped*2-1,rec_phiWarped*2-1



    def forward(self,input,moving,target=None):
        moving.volatile = True
        target.volatile = True
        affine_img,affine_map,_ = self.affine_net(input,moving,target)
        input = torch.cat((affine_img,target),1)
        input.volatile =False
        affine_img.volatile = False
        affine_map.volatile = False
        target.volatile=False
        moving.volatile=False
        m = self.momentum_net(input)
        #m.register_hook(print)
        #self.momentum_net.register_backward_hook(bh)
        moving = (moving+1)/2.
        target = (target+1)/2.
        affine_map = (affine_map+1)/2.
        #affine_img = (affine_img +1)/2.
        rec_IWarped,rec_phiWarped =self.do_mermaid_reg(moving,target,m,affine_map)
        return rec_IWarped,rec_phiWarped,affine_img


def bh(m,gi,go):
    print("Grad Input")
    print((torch.sum(gi[0].data), torch.sum(gi[1].data)))
    print("Grad Output")
    print(torch.sum(go[0].data))
    return gi[0],gi[1], gi[2]

