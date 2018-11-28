from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from model_pool.network_pool import *
from model_pool.utils import sigmoid_explode, get_inverse_affine_param, get_warped_img_map_param, update_affine_param
import mermaid.pyreg.module_parameters as pars
import mermaid.pyreg.model_factory as py_mf
import mermaid.pyreg.utils as py_utils
from functools import partial
import mermaid.pyreg.image_sampling as py_is
from mermaid.pyreg.libraries.functions.stn_nd import STNFunction_ND_BCXYZ
from model_pool.global_variable import *


def _get_low_res_size_from_size(sz, factor):
    """
    Returns the corresponding low-res size from a (high-res) sz
    :param sz: size (high-res)
    :param factor: low-res factor (needs to be <1)
    :return: low res size
    """
    if (factor is None) :
        print('WARNING: Could not compute low_res_size as factor was ' + str( factor ))
        return sz
    else:
        lowResSize = np.array(sz)
        if not isinstance(factor, list):
            lowResSize[2::] = (np.ceil((np.array(sz[2:]) * factor))).astype('int16')
        else:
            lowResSize[2::] = (np.ceil((np.array(sz[2:]) * np.array(factor)))).astype('int16')

        if lowResSize[-1]%2!=0:
            lowResSize[-1]-=1
            print('\n\nWARNING: forcing last dimension to be even: fix properly in the Fourier transform later!\n\n')

        return lowResSize


def _get_low_res_spacing_from_spacing(spacing, sz, lowResSize):
    """
    Computes spacing for the low-res parameterization from image spacing
    :param spacing: image spacing
    :param sz: size of image
    :param lowResSize: size of low re parameterization
    :return: returns spacing of low res parameterization
    """
    #todo: check that this is the correct way of doing it
    return spacing * (np.array(sz[2::])-1) / (np.array(lowResSize[2::])-1)

def _compute_low_res_image(I,spacing,low_res_size):
    sampler = py_is.ResampleImage()
    low_res_image, _ = sampler.downsample_image_to_size(I, spacing, low_res_size[2::],1)
    return low_res_image



class MermaidNet(nn.Module):
    """
    this network is an end to end system for momentum generation and mermaid registration
    include the following parts

    1 . (optional) affine net the affine network is used to affine the source and target image
    2. the momentum generation net work, this network is a u-net like encoder decoder
    3. the mermaid part, an single-scale optimizer would be called from the mermaid code

    In detail of implementation, we should take care of the memory issue, one possible solution is using low-resolution mapping

    1. affine network, this is a pretrained network, so the only the forward flow is used, the input should be set as volatile,
        in current  design, the input and output of this net is of the full resolution
    2. momentum generation net, this is a trainable network, but we would have a low-res factor to train it at a low-resolution
        the input may still at original resolution, but the output size may be determined by the low-res factor

    3. mermaid part, this is an non-parametric unit, where should call the single-scale optimizer, and the output should be upsampled to the
        full resolution size.

    so the input and the output of each part should be

    1. affine: input: source, target,   output: s_warped, affine_map
    2. momentum: input: init_warped_source, target,  output: low_res_m
    3. mermaid: input: s, low_res_m, low_res_initial_map  output: map, warped_source

    """

    def __init__(self, img_sz=None, opt=None):
        super(MermaidNet, self).__init__()
        self.load_external_model = False

        cur_gpu_id = opt['tsk_set']['gpu_ids']
        old_gpu_id = opt['tsk_set']['old_gpu_ids']
        opt_mermaid = opt['tsk_set']['reg']['mermaid_net']
        low_res_factor = opt['tsk_set']['reg'][('low_res_factor',1.,"factor of low-resolution map")]
        batch_sz = opt['tsk_set']['batch_sz']

        self.using_index_coord = opt_mermaid['using_index_coord']
        self.using_llddmm = opt_mermaid['using_llddmm']
        self.loss_type = opt['tsk_set']['loss']['type']
        self.using_sym_on = opt_mermaid['using_sym']
        self.sym_factor = opt_mermaid['sym_factor']
        self.using_mermaid_multi_step = opt_mermaid['using_multi_step']
        self.step = 1 if not self.using_mermaid_multi_step else opt_mermaid['num_step']
        self.using_affine_init = opt_mermaid['using_affine_init']
        self.affine_init_path = opt_mermaid['affine_init_path']


        self.img_sz = [batch_sz, 1] + img_sz
        self.dim = len(img_sz)
        spacing = 1. / (np.array(img_sz) - 1)
        self.spacing = spacing
        self.gpu_switcher = (cur_gpu_id, old_gpu_id)
        self.low_res_factor = low_res_factor
        self.momentum_net = MomentumNet(low_res_factor,opt_mermaid)
        if self.using_affine_init:
            self.init_affine_net(opt)
        else:
            print("Attention, the affine net is not used")


        self.mermaid_unit = None
        self.init_mermaid_env(spacing)
        self.print_count = 0

    def init_affine_net(self,opt):
        self.affine_net = AffineNetCycle(self.img_sz[2:],opt)

        if self.load_external_model:
            model_path = self.affine_init_path
            checkpoint = torch.load(model_path,  map_location='cpu')
            self.affine_net.load_state_dict(checkpoint['state_dict'])
            self.affine_net.cuda()
            print("Affine model is initialized!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.affine_net.eval()

    def set_cur_epoch(self,epoch=-1):
        self.epoch = epoch

    def init_mermaid_env(self, spacing):
        """setup the mermaid"""
        params = pars.ParameterDict()
        if not self.using_llddmm:
            params.load_JSON( '../mermaid/demos/cur_settings_lbfgs.json') #''../model_pool/cur_settings_svf.json')######TODO ###########
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

    def __cal_sym_loss(self,rec_phiWarped):
        trans1 = STNFunction_ND_BCXYZ(self.spacing,zero_boundary=False)
        trans2 = STNFunction_ND_BCXYZ(self.spacing,zero_boundary=False)
        identity_map  = self.identityMap.expand_as(rec_phiWarped[0])
        trans_st  = trans1(identity_map,rec_phiWarped[0])
        trans_st_ts = trans2(trans_st,rec_phiWarped[1])
        return torch.mean((identity_map- trans_st_ts)**2)

    def do_criterion_cal(self, ISource, ITarget,cur_epoch=-1):
        ISource = (ISource + 1.) / 2.
        ITarget = (ITarget + 1.) / 2.
        if not self.using_sym_on:
            loss_overall_energy, sim_energy, reg_energy = self.criterion(self.identityMap, self.rec_phiWarped, ISource,
                                                                         ITarget, None,
                                                                         self.mermaid_unit.get_variables_to_transfer_to_loss_function(),
                                                                         None)
            if self.print_count % 10 == 0 and cur_epoch>=0:
                print('the loss_over_all:{} sim_energy:{}, reg_energy:{}\n'.format(loss_overall_energy.item(),
                                                                                   sim_energy.item(),
                                                                                   reg_energy.item()))
            if self.using_mermaid_multi_step and self.step_loss is not None:
                self.step_loss += loss_overall_energy
                loss_overall_energy = self.step_loss
            if self.using_mermaid_multi_step and self.cur_step<self.step-1:
                self.print_count -= 1
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
            sym_energy = self.__cal_sym_loss(self.rec_phiWarped)
            sym_factor = sym_factor_in_mermaid #min(sigmoid_explode(cur_epoch,static=1, k=8)*0.01*gl_sym_factor,1.*gl_sym_factor) #static=5, k=4)*0.01,1) static=10, k=10)*0.01
            loss_overall_energy = loss_overall_energy + sym_factor*sym_energy
            if self.print_count % 10 == 0 and cur_epoch>=0:
                print('the loss_over_all:{} sim_energy:{},sym_factor: {} sym_energy: {} reg_energy:{}\n'.format(loss_overall_energy.item(),
                                                                                   sim_energy.item(),
                                                                                    sym_factor,
                                                                                     sym_energy.item(),
                                                                                   reg_energy.item()))
            if self.using_mermaid_multi_step and self.step_loss is not None:
                self.step_loss += loss_overall_energy
                loss_overall_energy = self.step_loss
            if self.using_mermaid_multi_step and self.cur_step<self.step-1:
                self.print_count -= 1
        self.print_count += 1
        return loss_overall_energy, sim_energy, reg_energy

    def set_mermaid_param(self, s, t, m):
        self.mermaid_unit.set_dictionary_to_pass_to_integrator({'I0': s, 'I1': t})
        self.mermaid_unit.m = m
        self.criterion.m = m

    def do_mermaid_reg(self, s, t, m, phi):
        """
        :param s: source image
        :param t: target image
        :param m: initial momentum
        :param phi: initial deformation field
        :return:  warped image, deformation field
        """
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

    def __transfer_return_var(self,rec_IWarped,rec_phiWarped,affine_img):
        return (rec_IWarped * 2. - 1.).detach(), (rec_phiWarped * 2. - 1.).detach(), affine_img.detach()


    def single_forward(self, moving, target=None):
        if self.using_affine_init:
            with torch.no_grad():
                affine_img, affine_map, _ = self.affine_net(moving, target)
                affine_map = (affine_map + 1) / 2.
                if self.using_index_coord:
                    affine_map[:, 0] = affine_map[:, 0] * self.spacing_record[1] / self.spacing_record[0]
        else:
            affine_map = self.identityMap.clone()
            affine_img = moving
        input = torch.cat((affine_img, target), 1)
        m = self.momentum_net(input)
        moving = (moving + 1) / 2.
        target = (target + 1) / 2.

        rec_IWarped, rec_phiWarped = self.do_mermaid_reg(moving, target, m, affine_map)
        self.rec_phiWarped = rec_phiWarped
        rec_phiWarped_tmp = rec_phiWarped.detach().clone()
        if self.using_index_coord:
            rec_phiWarped_tmp[:, 0] = rec_phiWarped[:, 0] * self.spacing_record[0] / self.spacing_record[1]
            rec_phiWarped = rec_phiWarped_tmp
        return self.__transfer_return_var(rec_IWarped, rec_phiWarped, affine_img)


    def sym_forward(self, moving, target=None):
        if self.using_affine_init:
            with torch.no_grad():
                affine_img_st, affine_map_st, affine_param = self.affine_net(moving, target)
                affine_img_ts, affine_map_ts, _ = self.affine_net(target, moving)
                affine_map_st = (affine_map_st + 1) / 2.
                affine_map_ts = (affine_map_ts + 1) / 2.
                if self.using_index_coord:
                    affine_map_st[:, 0] = affine_map_st[:, 0] * self.spacing_record[1] / self.spacing_record[0]
                    affine_map_ts[:, 0] = affine_map_ts[:, 0] * self.spacing_record[1] / self.spacing_record[0]
        else:
            affine_map_st = self.identityMap.clone()
            affine_map_ts = self.identityMap.clone()
            affine_img_st = moving
            affine_img_ts = target
        input_st = torch.cat((affine_img_st, target), 1)
        input_ts = torch.cat((affine_img_ts, moving), 1)
        m_st = self.momentum_net(input_st)
        m_ts = self.momentum_net(input_ts)
        moving = (moving + 1) / 2.
        target = (target + 1) / 2.
        rec_IWarped_st, rec_phiWarped_st = self.do_mermaid_reg(moving, target, m_st, affine_map_st)
        rec_IWarped_ts, rec_phiWarped_ts = self.do_mermaid_reg(target, moving, m_ts, affine_map_ts)
        self.rec_phiWarped = (rec_phiWarped_st, rec_phiWarped_ts)
        rec_phiWarped_tmp = rec_phiWarped_st.detach().clone()
        if self.using_index_coord:
            rec_phiWarped_tmp[:, 0] = rec_phiWarped_st[:, 0] * self.spacing_record[0] / self.spacing_record[1]
            rec_phiWarped_st = rec_phiWarped_tmp
        return self.__transfer_return_var(rec_IWarped_st, rec_phiWarped_st, affine_img_st)

    def cyc_forward(self, moving,target=None):
        self.step_loss = None
        if self.using_affine_init:
            with torch.no_grad():
                affine_img, affine_map, _ = self.affine_net(moving, target)
                moving_n = (moving + 1) / 2.  # [-1,1] ->[0,1]
                target_n = (target + 1) / 2.  # [-1,1] ->[0,1]
                affine_map = (affine_map + 1) / 2.  # [-1,1] ->[0,1]
                if self.using_index_coord:
                    affine_map[:, 0] = affine_map[:, 0] * self.spacing_record[1] / self.spacing_record[0]
        else:
            affine_map = self.identityMap.clone()
            affine_img = moving

        warped_img = affine_img
        init_map = affine_map
        rec_IWarped = None
        rec_phiWarped = None

        for i in range(self.step):
            self.cur_step = i
            input = torch.cat((warped_img, target), 1)
            m = self.momentum_net(input)
            rec_IWarped, rec_phiWarped = self.do_mermaid_reg(moving_n, target_n, m, init_map)
            warped_img = rec_IWarped * 2 - 1  # [0,1] -> [-1,1]
            init_map = rec_phiWarped  # [0,1]
            self.rec_phiWarped = rec_phiWarped
            if self.using_mermaid_multi_step and i < self.step - 1:
                self.step_loss, _, _ = self.do_criterion_cal(moving, target, self.epoch)

        rec_phiWarped_tmp = rec_phiWarped.detach().clone()
        if self.using_index_coord:
            rec_phiWarped_tmp[:, 0] = rec_phiWarped[:, 0] * self.spacing_record[0] / self.spacing_record[1]
            rec_phiWarped = rec_phiWarped_tmp

        return self.__transfer_return_var(rec_IWarped, rec_phiWarped, affine_img)



    def cyc_sym_forward(self,moving, target= None):
        self.step_loss=None
        if self.using_affine_init:
            with torch.no_grad():
                affine_img_st, affine_map_st, _ = self.affine_net(moving, target)
                affine_img_ts, affine_map_ts, _ = self.affine_net(target, moving)
                moving_n = (moving + 1) / 2.  # [-1,1] ->[0,1]
                target_n = (target + 1) / 2.  # [-1,1] ->[0,1]
                affine_map_st = (affine_map_st + 1) / 2.  # [-1,1] ->[0,1]
                affine_map_ts = (affine_map_ts + 1) / 2.  # [-1,1] ->[0,1]
                if self.using_index_coord:
                    affine_map_st[:,0] = affine_map_st[:,0] * self.spacing_record[1]/self.spacing_record[0]
                    affine_map_ts[:,0] = affine_map_ts[:,0] * self.spacing_record[1]/self.spacing_record[0]
        else:
            affine_map_st = self.identityMap.clone()
            affine_map_ts = self.identityMap.clone()
            affine_img_st = moving
            affine_img_ts = target

        warped_img_st = affine_img_st
        init_map_st = affine_map_st
        warped_img_ts = affine_img_ts
        init_map_ts = affine_map_ts
        rec_phiWarped_st = None
        rec_IWarped_st = None
        for i in range(self.step):
            self.cur_step = i
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
            if self.using_mermaid_multi_step and i<self.step-1:
                self.step_loss,_,_ = self.do_criterion_cal(moving,target,self.epoch)
        rec_phiWarped_tmp = rec_phiWarped_st.detach().clone()
        if self.using_index_coord:
            rec_phiWarped_tmp[:,0] = rec_phiWarped_st[:,0] * self.spacing_record[0]/self.spacing_record[1]
            rec_phiWarped_st = rec_phiWarped_tmp

        return self.__transfer_return_var(rec_IWarped_st, rec_phiWarped_st, affine_img_st)

    def get_affine_map(self,moving, target):
        with torch.no_grad():
            affine_img, affine_map, _ = self.affine_net(moving, target)
        return affine_map


    def forward(self, moving, target=None):

        if self.using_mermaid_multi_step and self.using_sym_on:
            if not self.print_count:
                print(" The mermaid network is in multi-step and symmetric mode")
            return self.cyc_sym_forward(moving,target)
        if self.using_mermaid_multi_step:
            if not self.print_count:
                print(" The mermaid network is in multi-step mode")
            return self.cyc_forward(moving, target)
        if not self.using_sym_on:
            if not self.print_count:
                print(" The mermaid network is in simple mode")
            return self.single_forward(moving, target)
        else:
            if not self.print_count:
                print(" The mermaid network is in symmetric mode")
            return self.sym_forward(moving,target)


def bh(m, gi, go):
    print("Grad Input")
    print((torch.sum(gi[0].data), torch.sum(gi[1].data)))
    print("Grad Output")
    print(torch.sum(go[0].data))
    return gi[0], gi[1], gi[2]

