from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model_pool.utils import *
from model_pool.network_pool import *
import mermaid.module_parameters as pars
import mermaid.model_factory as py_mf
import mermaid.utils as py_utils
from functools import partial
from mermaid.libraries.functions.stn_nd import STNFunction_ND_BCXYZ





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

        cur_gpu_id = opt['tsk_set']['gpu_ids']
        old_gpu_id = opt['tsk_set'][('old_gpu_ids',0,'the gpu id of the trained model')]
        opt_mermaid = opt['tsk_set']['reg']['mermaid_net']
        low_res_factor = opt['tsk_set']['reg'][('low_res_factor',1.,"factor of low-resolution map")]
        batch_sz = opt['tsk_set']['batch_sz']
        self.record_path = opt['tsk_set']['path']['record_path']

        self.is_train = opt['tsk_set']['train']
        self.epoch = 0

        self.using_physical_coord = opt_mermaid[('using_physical_coord',False,'use physical coordinate system')]
        self.loss_type = opt['tsk_set']['loss']['type']
        self.compute_inverse_map = opt['tsk_set']['reg'][('compute_inverse_map', False,"compute the inverse transformation map")]
        self.mermaid_net_json_pth = opt_mermaid[('mermaid_net_json_pth','../mermaid/demos/cur_settings_lbfgs.json',"the path for mermaid settings json")]
        self.using_sym_on = opt_mermaid[('using_sym',False,'compute symmetric loss')]
        self.sym_factor = opt_mermaid[('sym_factor',1,'factor on symmetric loss')]
        self.using_mermaid_multi_step = opt_mermaid['using_multi_step']
        self.step = 1 if not self.using_mermaid_multi_step else opt_mermaid[('num_step',2,'compute multi-step loss')]
        self.using_affine_init = opt_mermaid[('using_affine_init',True,'True, deploy an affine network before mermaid-net')]
        self.load_trained_affine_net = opt_mermaid[('load_trained_affine_net',True,'load the trained affine network')]
        self.affine_init_path = opt_mermaid[('affine_init_path','',"the path of trained affined network")]
        self.optimize_momentum_network = opt_mermaid[('optimize_momentum_network',True,'true if optimize the momentum network')]
        self.epoch_list_fixed_momentum_network = opt_mermaid[('epoch_list_fixed_momentum_network',[-1],'list of epoch, fix the momentum network')]
        self.epoch_list_fixed_deep_smoother_network = opt_mermaid[('epoch_list_fixed_deep_smoother_network',[-1],'epoch_list_fixed_deep_smoother_network')]
        self.clamp_momentum = opt_mermaid[('clamp_momentum',False,'clamp_momentum')]
        self.clamp_thre = 1.
        self.use_adaptive_smoother = False

        if self.clamp_momentum:
            print("Attention, the clamp momentum is on")
        ##### TODO  the sigma also need to be set like sqrt(batch_sz) ##########



        batch_sz = batch_sz if not self.using_sym_on  else batch_sz*2
        self.img_sz = [batch_sz, 1] + img_sz
        self.dim = len(img_sz)
        self.standard_spacing = 1. / (np.array(img_sz) - 1)
        """ here we define the standard spacing measures the image from 0 to 1"""
        self.spacing = np.asarray(opt['dataset']['spacing_to_refer']) if self.using_physical_coord else 1. / (
                    np.array(img_sz) - 1)
        self.spacing = normalize_spacing(self.spacing, self.input_img_sz) if self.using_physical_coord else self.spacing
        self.spacing = np.array(self.spacing) if type(self.spacing) is not np.ndarray else self.spacing
        self.gpu_switcher = (cur_gpu_id, old_gpu_id)
        self.low_res_factor = low_res_factor
        self.momentum_net = MomentumNet(low_res_factor,opt_mermaid)
        if self.using_affine_init:
            self.init_affine_net(opt)
        else:
            print("Attention, the affine net is not used")


        self.mermaid_unit_st = None
        self.init_mermaid_env(self.spacing)
        self.print_count = 0
        self.print_every_epoch_flag = True
        self.n_batch = -1
        self.inverse_map = None

    def init_affine_net(self,opt):
        self.affine_net = AffineNetSym(self.img_sz[2:],opt)
        self.affine_net.compute_loss = False
        self.affine_net.epoch_activate_sym = 1e7  # todo to fix
        model_path = self.affine_init_path
        if self.load_trained_affine_net:
            checkpoint = torch.load(model_path,  map_location='cpu')
            self.affine_net.load_state_dict(checkpoint['state_dict'])
            self.affine_net.cuda()
            print("Affine model is initialized!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            print("The Affine model is added, but not initialized")
        self.affine_net.eval()

    def set_cur_epoch(self,epoch=-1):
        if self.epoch !=epoch+1:
            self.print_every_epoch_flag=True
        self.epoch = epoch+1


    def set_loss_fn(self, loss_fn):
        pass


    def save_cur_mermaid_settings(self,params):
        saving_path = os.path.join(self.record_path,'cur_settings_mermaid_output.json')
        params.write_JSON(saving_path, save_int=False)


    def init_mermaid_env(self, spacing):
        """setup the mermaid"""
        params = pars.ParameterDict()
        params.load_JSON( self.mermaid_net_json_pth) #''../model_pool/cur_settings_svf.json')
        self.save_cur_mermaid_settings(params)
        #params.load_JSON( '../mermaid/demos/cur_settings_lbfgs_forlddmm.json') #''../model_pool/cur_settings_svf.json')
        print(" The mermaid setting from {} included:".format(self.mermaid_net_json_pth))
        print(params)
        model_name = params['model']['registration_model']['type']
        use_map = params['model']['deformation']['use_map']
        compute_similarity_measure_at_low_res = params['model']['deformation'][
            ('compute_similarity_measure_at_low_res', False, 'to compute Sim at lower resolution')]
        params['model']['registration_model']['similarity_measure']['type'] =self.loss_type
        params.print_settings_off()
        self.mermaid_low_res_factor = self.low_res_factor
        smoother_type =  params['model']['registration_model']['forward_model']['smoother']['type']
        self.use_adaptive_smoother =smoother_type=='learned_multiGaussianCombination'

        lowResSize = None
        lowResSpacing = None
        ##
        if self.mermaid_low_res_factor == 1.0 or self.mermaid_low_res_factor == [1., 1., 1.]:
            self.mermaid_low_res_factor = None

        self.lowResSize = self.img_sz
        self.lowResSpacing = spacing
        ##
        if self.mermaid_low_res_factor is not None:
            lowResSize = get_res_size_from_size(self.img_sz, self.mermaid_low_res_factor)
            lowResSpacing = get_res_spacing_from_spacing(spacing, self.img_sz, lowResSize)
            self.lowResSize = lowResSize
            self.lowResSpacing = lowResSpacing

        if self.mermaid_low_res_factor is not None:
            # computes model at a lower resolution than the image similarity
            if compute_similarity_measure_at_low_res:
                mf = py_mf.ModelFactory(lowResSize, lowResSpacing, lowResSize, lowResSpacing)
            else:
                mf = py_mf.ModelFactory(self.img_sz, spacing, lowResSize, lowResSpacing)
        else:
            # computes model and similarity at the same resolution
            mf = py_mf.ModelFactory(self.img_sz, spacing, self.img_sz, spacing)
        model_st, criterion_st = mf.create_registration_model(model_name, params['model'], compute_inverse_map=self.compute_inverse_map)


        if use_map:
            # create the identity map [0,1]^d, since we will use a map-based implementation
            _id = py_utils.identity_map_multiN(self.img_sz, spacing)
            self.identityMap = torch.from_numpy(_id).cuda()
            if self.mermaid_low_res_factor is not None:
                # create a lower resolution map for the computations
                lowres_id = py_utils.identity_map_multiN(lowResSize, lowResSpacing)
                self.lowResIdentityMap = torch.from_numpy(lowres_id).cuda()
                print(torch.min(self.lowResIdentityMap))
        self.lowRes_fn = partial(get_resampled_image, spacing=spacing, desiredSize=lowResSize, zero_boundary=False,identity_map=self.lowResIdentityMap)
        self.mermaid_unit_st = model_st.cuda()
        self.criterion_st = criterion_st
        self.mermaid_unit_st.associate_parameters_with_module()

    def get_loss(self):
        return self.overall_loss

    def __cal_sym_loss(self,rec_phiWarped):
        trans1 = STNFunction_ND_BCXYZ(self.spacing,zero_boundary=False)
        trans2 = STNFunction_ND_BCXYZ(self.spacing,zero_boundary=False)
        st_map = rec_phiWarped[:self.n_batch]
        ts_map = rec_phiWarped[self.n_batch:]
        identity_map  = self.identityMap[0:self.n_batch]
        trans_st  = trans1(identity_map,st_map)
        trans_st_ts = trans2(trans_st,ts_map)
        return torch.mean((identity_map- trans_st_ts)**2)

    def do_criterion_cal(self, ISource, ITarget,cur_epoch=-1):
        # todo the image is not necessary be normalized here, just keep -1,1 would be fine
        ISource = (ISource + 1.) / 2.
        ITarget = (ITarget + 1.) / 2.

        loss_overall_energy, sim_energy, reg_energy = self.criterion_st(self.identityMap, self.rec_phiWarped, ISource,
                                                                     ITarget, self.low_moving,
                                                                     self.mermaid_unit_st.get_variables_to_transfer_to_loss_function(),
                                                                     None)
        if not self.using_sym_on:
            if self.print_count % 10 == 0 and cur_epoch>=0:
                print('the loss_over_all:{} sim_energy:{}, reg_energy:{}'.format(loss_overall_energy.item(),
                                                                                   sim_energy.item(),
                                                                                   reg_energy.item()))
        else:
            sym_energy = self.__cal_sym_loss(self.rec_phiWarped)
            sym_factor = self.sym_factor  # min(sigmoid_explode(cur_epoch,static=1, k=8)*0.01*gl_sym_factor,1.*gl_sym_factor) #static=5, k=4)*0.01,1) static=10, k=10)*0.01
            loss_overall_energy = loss_overall_energy + sym_factor * sym_energy
            loss_overall_energy = loss_overall_energy + sym_factor * sym_energy
            if self.print_count % 10 == 0 and cur_epoch >= 0:
                print('the loss_over_all:{} sim_energy:{},sym_factor: {} sym_energy: {} reg_energy:{}'.format(
                    loss_overall_energy.item(),
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

    def set_mermaid_param(self,mermaid_unit,criterion, s, t, m,s_full=None):
        mermaid_unit.set_dictionary_to_pass_to_integrator({'I0': s, 'I1': t,'I0_full':s_full})
        criterion.set_dictionary_to_pass_to_smoother({'I0': s, 'I1': t,'I0_full':s_full})
        mermaid_unit.m = m
        criterion.m = m
    def __freeze_param(self,params):
        for param in params:
            param.requires_grad = False

    def __active_param(self,params):
        for param in params:
            param.requires_grad = True

    def get_inverse_map(self,use_01=False):
        if use_01 or self.inverse_map is None:
            return self.inverse_map
        else:
            return self.inverse_map*2-1




    def init_mermaid_param(self,s):
        if self.use_adaptive_smoother:
            if self.epoch in self.epoch_list_fixed_deep_smoother_network:
                #self.mermaid_unit_st.smoother._enable_force_nn_gradients_to_zero_hooks()
                self.__freeze_param(self.mermaid_unit_st.smoother.ws.parameters())
            else:
                self.__active_param(self.mermaid_unit_st.smoother.ws.parameters())


        if self.mermaid_low_res_factor is not None:
            if s.shape[0]==self.lowResIdentityMap.shape[0]:
                low_s= get_resampled_image(s, self.spacing, self.lowResSize, 1, zero_boundary=True, identity_map=self.lowResIdentityMap)
            else:
                n_batch = s.shape[0]
                lowResSize = self.lowResSize.copy()
                lowResSize[0] = n_batch
                low_s = get_resampled_image(s, self.spacing, lowResSize, 1, zero_boundary=True,
                                            identity_map=self.lowResIdentityMap[0:n_batch])
            return low_s
        else:
            return None

    def do_mermaid_reg(self,mermaid_unit,criterion, s, t, m, phi,low_s=None,low_t=None,inv_map=None):
        """
        :param s: source image
        :param t: target image
        :param m: initial momentum
        :param phi: initial deformation field
        :return:  warped image, deformation field
        """
        if self.mermaid_low_res_factor is not None:
            self.set_mermaid_param(mermaid_unit,criterion,low_s, low_t, m,s)  ##########3 TODO  here the input shouold be low_s low_t if self.mermaid_low_res_factor is not None otherwise s,t
            if not self.compute_inverse_map:
                maps = mermaid_unit(self.lowRes_fn(phi), low_s, variables_from_optimizer={'epoch':self.epoch})
            else:
                maps, inverse_maps = mermaid_unit(self.lowRes_fn(phi), low_s,phi_inv=self.lowRes_fn(inv_map), variables_from_optimizer={'epoch':self.epoch})

            desiredSz = self.img_sz
            rec_phiWarped = get_resampled_image(maps, self.lowResSpacing, desiredSz, 1,zero_boundary=False,identity_map=self.identityMap)
            if self.compute_inverse_map:
                self.inverse_map = get_resampled_image(inverse_maps, self.lowResSpacing, desiredSz, 1,
                                                                  zero_boundary=False,identity_map=self.identityMap)

        else:
            self.set_mermaid_param(mermaid_unit,criterion,s, t, m,s)
            if not self.compute_inverse_map:
                maps = mermaid_unit(phi, s, variables_from_optimizer={'epoch':self.epoch})
            else:
                maps, self.inverse_map = mermaid_unit(phi, s,phi_inv = inv_map, variables_from_optimizer = {'epoch': self.epoch})
            rec_phiWarped = maps
        rec_IWarped = py_utils.compute_warped_image_multiNC(s, rec_phiWarped, self.spacing, 1,zero_boundary=True)
        self.rec_phiWarped = rec_phiWarped

        return rec_IWarped, rec_phiWarped

    def __get_adaptive_smoother_map(self):

        adaptive_smoother_map = self.mermaid_unit_st.smoother.get_deep_smoother_weights()
        if not self.using_sym_on:
            adaptive_smoother_map = adaptive_smoother_map.detach()
        else:
            adaptive_smoother_map = adaptive_smoother_map[:self.n_batch].detach()
        gaussian_weights = self.mermaid_unit_st.smoother.get_gaussian_weights()
        gaussian_weights = gaussian_weights.detach()
        print(" the current global gaussian weight is {}".format(gaussian_weights))

        gaussian_stds = self.mermaid_unit_st.smoother.get_gaussian_stds()
        gaussian_stds = gaussian_stds.detach()
        print(" the current global gaussian stds is {}".format(gaussian_stds))
        view_sz = [1] + [len(gaussian_stds)] + [1] * dim
        gaussian_stds = gaussian_stds.view(*view_sz)
        adaptive_smoother_map = adaptive_smoother_map**2 # todo   this is true only when we use w_K_W

        smoother_map = adaptive_smoother_map*(gaussian_stds**2)
        smoother_map = torch.sqrt(torch.sum(smoother_map,1,keepdim=True))
        #_,smoother_map = torch.max(adaptive_smoother_map.detach(),dim=1,keepdim=True)
        self._display_stats(smoother_map.float(),'statistic for max_smoother map')
        return smoother_map

    def _display_stats(self, Ia, iname):

        Ia_min = Ia.min().detach().cpu().numpy()
        Ia_max = Ia.max().detach().cpu().numpy()
        Ia_mean = Ia.mean().detach().cpu().numpy()
        Ia_std = Ia.std().detach().cpu().numpy()

        print('{}:after: [{:.2f},{:.2f},{:.2f}]({:.2f})'.format(iname, Ia_min,Ia_mean,Ia_max,Ia_std))




    def get_extra_to_plot(self):
        if self.use_adaptive_smoother:
            # the last step adaptive smoother is returned, todo add the first stage smoother
            return self.__get_adaptive_smoother_map(), 'Inital_weight'
        else:
            return None, None


    def __transfer_return_var(self,rec_IWarped,rec_phiWarped,affine_img):
        return (rec_IWarped).detach(), (rec_phiWarped * 2. - 1.).detach(), ((affine_img+1.)/2.).detach()




    def single_forward(self, moving, target=None):
        if self.using_affine_init:
            with torch.no_grad():
                affine_img, affine_map, affine_param = self.affine_net(moving, target)
                affine_map = (affine_map + 1) / 2.
                inverse_map = None
                if self.compute_inverse_map:
                    affine_inverse = get_inverse_affine_param(affine_param)
                    affine_img_inverse, affine_map_inverse, _ = get_warped_img_map_param(affine_inverse, self.img_sz[2:], target)
                    inverse_map = (affine_map_inverse + 1) / 2.

                if self.using_physical_coord:
                    for i in range(self.dim):
                        affine_map[:, i] = affine_map[:, i] * self.spacing[i] / self.standard_spacing[i]
                    if self.compute_inverse_map:
                        for i in range(self.dim):
                            inverse_map[:, i] = inverse_map[:, i] * self.spacing[i] / self.standard_spacing[i]
                self.inverse_map = inverse_map
        else:
            affine_map = self.identityMap.clone()
            if self.compute_inverse_map:
                self.inverse_map = self.identityMap.clone()

            affine_img = moving
        record_is_grad_enabled = torch.is_grad_enabled()
        if not self.optimize_momentum_network or self.epoch in self.epoch_list_fixed_momentum_network:
            torch.set_grad_enabled(False)
        if self.print_every_epoch_flag:
            if self.epoch in self.epoch_list_fixed_momentum_network:
                print("In this epoch, the momentum network is fixed")
            if self.epoch in self.epoch_list_fixed_deep_smoother_network:
                print("In this epoch, the deep smoother deep network is fixed")
            self.print_every_epoch_flag = False
        input = torch.cat((affine_img, target), 1)
        m = self.momentum_net(input)
        if self.clamp_momentum:
            m=m.clamp(max=self.clamp_thre,min=-self.clamp_thre)
        moving = (moving + 1) / 2.
        target = (target + 1) / 2.
        self.low_moving = self.init_mermaid_param(moving)
        self.low_target = self.init_mermaid_param(target)
        torch.set_grad_enabled(record_is_grad_enabled)
        rec_IWarped, rec_phiWarped = self.do_mermaid_reg(self.mermaid_unit_st,self.criterion_st,moving, target, m, affine_map,self.low_moving, self.low_target,self.inverse_map)
        self.rec_phiWarped = rec_phiWarped
        if self.using_physical_coord:
            rec_phiWarped_tmp = rec_phiWarped.detach().clone()
            for i in range(self.dim):
                rec_phiWarped_tmp[:, i] = rec_phiWarped[:, i] * self.standard_spacing[i] / self.spacing[i]
            rec_phiWarped = rec_phiWarped_tmp
        self.overall_loss,_,_= self.do_criterion_cal(moving, target, cur_epoch=self.epoch)
        return self.__transfer_return_var(rec_IWarped, rec_phiWarped, affine_img)





    def sym_forward(self, moving, target=None):
        self.n_batch = moving.shape[0]
        moving_sym = torch.cat((moving, target), 0)
        target_sym = torch.cat((target, moving), 0)
        rec_IWarped_st, rec_phiWarped_st, affine_img_st = self.single_forward(moving_sym, target_sym)
        return rec_IWarped_st[:self.n_batch],rec_phiWarped_st[:self.n_batch], affine_img_st[:self.n_batch]




    def cyc_forward(self, moving,target=None):
        self.step_loss = None
        if self.using_affine_init:
            with torch.no_grad():
                affine_img, affine_map, affine_param = self.affine_net(moving, target)
                affine_map = (affine_map + 1) / 2.  # [-1,1] ->[0,1]
                inverse_map = None
                if self.compute_inverse_map:
                    affine_inverse = get_inverse_affine_param(affine_param)
                    affine_img_inverse, affine_map_inverse, _ = get_warped_img_map_param(affine_inverse, self.img_sz[2:], target)
                    inverse_map = (affine_map_inverse + 1) / 2.

                if self.using_physical_coord:
                    for i in range(self.dim):
                        affine_map[:, i] = affine_map[:, i] * self.spacing[i] / self.standard_spacing[i]
                    if self.compute_inverse_map:
                        for i in range(self.dim):
                            inverse_map[:, i] = inverse_map[:, i] * self.spacing[i] / self.standard_spacing[i]
                self.inverse_map = inverse_map
        else:
            affine_map = self.identityMap.clone()
            if self.compute_inverse_map:
                self.inverse_map = self.identityMap.clone()
            affine_img = moving
        warped_img = affine_img
        init_map = affine_map
        rec_IWarped = None
        rec_phiWarped = None
        moving_n = (moving + 1) / 2.  # [-1,1] ->[0,1]
        target_n = (target + 1) / 2.  # [-1,1] ->[0,1]
        self.low_moving = self.init_mermaid_param(moving_n)
        self.low_target = self.init_mermaid_param(target_n)

        for i in range(self.step):
            self.cur_step = i
            record_is_grad_enabled = torch.is_grad_enabled()
            if not self.optimize_momentum_network or self.epoch in self.epoch_list_fixed_momentum_network:
                torch.set_grad_enabled(False)
            if self.print_every_epoch_flag:
                if self.epoch in self.epoch_list_fixed_momentum_network:
                    print("In this epoch, the momentum network is fixed")
                if self.epoch in self.epoch_list_fixed_deep_smoother_network:
                    print("In this epoch, the deep smoother deep network is fixed")
                self.print_every_epoch_flag = False
            input = torch.cat((warped_img, target), 1)
            m = self.momentum_net(input)
            if self.clamp_momentum:
                m=m.clamp(max=self.clamp_thre,min=-self.clamp_thre)
            torch.set_grad_enabled(record_is_grad_enabled)
            rec_IWarped, rec_phiWarped = self.do_mermaid_reg(self.mermaid_unit_st,self.criterion_st,moving_n, target_n, m, init_map,self.low_moving, self.low_target, self.inverse_map)
            warped_img = rec_IWarped * 2 - 1  # [0,1] -> [-1,1]
            init_map = rec_phiWarped  # [0,1]
            self.rec_phiWarped = rec_phiWarped
            if self.using_mermaid_multi_step and i < self.step - 1:
                self.step_loss, _, _ = self.do_criterion_cal(moving, target, self.epoch)

        if self.using_physical_coord:
            rec_phiWarped_tmp = rec_phiWarped.detach().clone()
            for i in range(self.dim):
                rec_phiWarped_tmp[:, i] = rec_phiWarped[:, i] * self.standard_spacing[i] / self.spacing[i]
            rec_phiWarped = rec_phiWarped_tmp
        self.overall_loss,_,_= self.do_criterion_cal(moving, target, cur_epoch=self.epoch)
        return self.__transfer_return_var(rec_IWarped, rec_phiWarped, affine_img)



    def cyc_sym_forward(self,moving, target= None):
        self.n_batch = moving.shape[1]
        moving_sym = torch.cat((moving, target), 0)
        target_sym = torch.cat((target, moving), 0)
        rec_IWarped, rec_phiWarped, affine_img = self.cyc_forward(moving_sym, target_sym)
        return rec_IWarped[:self.n_batch], rec_phiWarped[:self.n_batch], affine_img[:self.n_batch]

    def get_affine_map(self,moving, target):
        with torch.no_grad():
            affine_img, affine_map, _ = self.affine_net(moving, target)
        return affine_map


    def forward(self, moving, target=None):

        if self.using_mermaid_multi_step and self.using_sym_on:
            if not self.print_count:
                print(" The mermaid network is in multi-step and symmetric mode, with step {}".format(self.step))
            return self.cyc_sym_forward(moving,target)
        if self.using_mermaid_multi_step:
            if not self.print_count:
                print(" The mermaid network is in multi-step mode, with step {}".format(self.step))
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

