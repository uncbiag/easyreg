from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .utils import *
from .affine_net import *
from .momentum_net import *
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
    3. the mermaid part, an map-based registration model would be called from the Mermaid tookit

    In detail of implementation, we should take care of the memory issue, one possible solution is using low-resolution mapping and then upsampling the transformation map

    1. affine network, this is a pretrained network, so only the forward model is used,
        in current  design, the input and output of this net is not downsampled
    2. momentum generation net, this is a trainable network, but we would have a low-res factor to train it at a low-resolution
        the input may still at original resolution (for high quality interpolation), but the size during the computation and of the output are determined by the low-res factor

    3. mermaid part, this is an non-parametric unit, where should call from the mermaid, and the output transformation map should be upsampled to the
        full resolution size. All momentum based mermaid registration method should be supported. (todo support velcoity methods)

    so the input and the output of each part should be

    1. affine: input: source, target,   output: s_warped, affine_map
    2. momentum: input: init_warped_source, target,  output: low_res_mom
    3. mermaid: input: s, low_res_mom, low_res_initial_map  output: map, warped_source

    pay attention in Mermaid toolkit, the image intensity and identity transformation coord are normalized into [0,1],
    while in networks the intensity and identity transformation coord are normalized into [-1,1],
    todo use the coordinate system consistent with mermaid [0,1]


    """

    def __init__(self, img_sz=None, opt=None):
        super(MermaidNet, self).__init__()

        opt_mermaid = opt['tsk_set']['reg']['mermaid_net']
        low_res_factor = opt['tsk_set']['reg'][('low_res_factor',1.,"factor of low-resolution map")]
        batch_sz = opt['tsk_set']['batch_sz']
        self.record_path = opt['tsk_set']['path'][('record_path',"","record path")]
        """record path of the task"""
        self.is_train = opt['tsk_set'][('train',False,'if is in train mode')]
        """if is in train mode"""
        self.epoch = 0
        """the current epoch"""
        self.using_physical_coord = opt_mermaid[('using_physical_coord',False,'use physical coordinate system')]
        """'use physical coordinate system"""
        self.loss_type = opt['tsk_set']['loss'][('type','lncc',"the similarity measure type, support list: 'l1','mse','ncc','lncc'")]
        """the similarity measure supported by the mermaid:  'ssd','ncc','ncc_positive','ncc_negative', 'lncc', 'omt'"""
        self.compute_inverse_map = opt['tsk_set']['reg'][('compute_inverse_map', False,"compute the inverse transformation map")]
        """compute the inverse transformation map"""
        self.mermaid_net_json_pth = opt_mermaid[('mermaid_net_json_pth','',"the path for mermaid settings json")]
        """the path for mermaid settings json"""
        self.sym_factor = opt_mermaid[('sym_factor',500,'factor on symmetric loss')]
        """factor on symmetric loss"""
        self.epoch_activate_sym = opt_mermaid[('epoch_activate_sym',-1,'epoch activate the symmetric loss')]
        """epoch activate the symmetric loss"""
        self.epoch_activate_multi_step = opt_mermaid[('epoch_activate_multi_step',-1,'epoch activate the multi-step')]
        """epoch activate the multi-step"""
        self.reset_lr_for_multi_step = opt_mermaid[('reset_lr_for_multi_step',False,'if True, reset learning rate when multi-step begins')]
        """if True, reset learning rate when multi-step begins"""
        self.lr_for_multi_step = opt_mermaid[('lr_for_multi_step',opt['tsk_set']['optim']['lr']/2,'if reset_lr_for_multi_step, reset learning rate when multi-step begins')]
        """if reset_lr_for_multi_step, reset learning rate when multi-step begins"""
        self.multi_step = opt_mermaid[('num_step',2,'compute multi-step loss')]
        """compute multi-step loss"""
        self.using_affine_init = opt_mermaid[('using_affine_init',True,'if ture, deploy an affine network before mermaid-net')]
        """if ture, deploy an affine network before mermaid-net"""
        self.load_trained_affine_net = opt_mermaid[('load_trained_affine_net',True,'if true load_trained_affine_net; if false, the affine network is not initialized')]
        """if true load_trained_affine_net; if false, the affine network is not initialized"""
        self.affine_init_path = opt_mermaid[('affine_init_path','',"the path of trained affined network")]
        """the path of trained affined network"""
        self.affine_resoltuion =  opt_mermaid[('affine_resoltuion',[-1,-1,-1],"the image resolution input for affine")]
        self.affine_refine_step = opt_mermaid[('affine_refine_step', 5, "the multi-step num in affine refinement")]
        """the multi-step num in affine refinement"""
        self.optimize_momentum_network = opt_mermaid[('optimize_momentum_network',True,'if true, optimize the momentum network')]
        """if true optimize the momentum network"""
        self.epoch_list_fixed_momentum_network = opt_mermaid[('epoch_list_fixed_momentum_network',[-1],'list of epoch, fix the momentum network')]
        """list of epoch, fix the momentum network"""
        self.epoch_list_fixed_deep_smoother_network = opt_mermaid[('epoch_list_fixed_deep_smoother_network',[-1],'epoch_list_fixed_deep_smoother_network')]
        """epoch_list_fixed_deep_smoother_network"""
        self.clamp_momentum = opt_mermaid[('clamp_momentum',False,'clamp_momentum')]
        """if true, clamp_momentum"""
        self.clamp_thre =opt_mermaid[('clamp_thre',1.0,'clamp momentum into [-clamp_thre, clamp_thre]')]
        """clamp momentum into [-clamp_thre, clamp_thre]"""
        self.mask_input_when_compute_loss = opt_mermaid[
            ('mask_input_when_compute_loss', False, 'mask_input_when_compute_loss')]
        """ mask input when compute loss"""
        self.use_adaptive_smoother = False
        self.print_loss_every_n_iter = 10 if self.is_train else 1
        self.using_sym_on = True if self.is_train else False

        if self.clamp_momentum:
            print("Attention, the clamp momentum is on")
        ##### TODO  the sigma also need to be set like sqrt(batch_sz) ##########



        batch_sz = batch_sz if not self.using_sym_on  else batch_sz*2
        self.img_sz = [batch_sz, 1] + img_sz
        self.affine_resoltuion =  [batch_sz, 1]+ self.affine_resoltuion
        self.dim = len(img_sz)
        self.standard_spacing = 1. / (np.array(img_sz) - 1)
        """ here we define the standard spacing measures the image coord from 0 to 1"""
        spacing_to_refer = opt['dataset'][('spacing_to_refer',[1, 1, 1],'the physical spacing in numpy coordinate, only activate when using_physical_coord is true')]
        self.spacing = normalize_spacing(spacing_to_refer, img_sz) if self.using_physical_coord else 1. / (
                    np.array(img_sz) - 1)
        self.spacing = normalize_spacing(self.spacing, self.input_img_sz) if self.using_physical_coord else self.spacing
        self.spacing = np.array(self.spacing) if type(self.spacing) is not np.ndarray else self.spacing
        self.low_res_factor = low_res_factor
        self.momentum_net = MomentumNet(low_res_factor,opt_mermaid)
        if self.using_affine_init:
            self.init_affine_net(opt)
        else:
            print("Attention, the affine net is not used")


        self.mermaid_unit_st = None
        self.init_mermaid_env()
        self.print_count = 0
        self.print_every_epoch_flag = True
        self.n_batch = -1
        self.inverse_map = None

    def load_pretrained_model(self, pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path, map_location="cpu")
        self.load_state_dict(checkpoint["state_dict"])
        print("load pretrained model from {}".format(pretrained_model_path))

    def check_if_update_lr(self):
        """
        check if the learning rate need to be updated,  in mermaid net, it is implemented for adjusting the lr in the multi-step training

        :return: if update the lr, return True and new lr, else return False and None
        """
        if self.epoch == self.epoch_activate_multi_step and self.reset_lr_for_multi_step:
            lr = self.lr_for_multi_step
            self.reset_lr_for_multi_step = False
            print("the lr is change into {} due to the activation of the multi-step".format(lr))
            return True, lr
        else:
            return False, None

    def init_affine_net(self,opt):
        """
        initialize the affine network, if an affine_init_path is given , then load the affine model from the path.

        :param opt: ParameterDict, task setting
        :return:
        """
        self.affine_net = AffineNetSym(self.img_sz[2:],opt)
        self.affine_param = None
        self.affine_net.compute_loss = False
        self.affine_net.epoch_activate_sym = 1e7  # todo to fix this unatural setting
        self.affine_net.set_step(self.affine_refine_step)
        model_path = self.affine_init_path
        if self.load_trained_affine_net and self.is_train:
            checkpoint = torch.load(model_path,  map_location='cpu')
            self.affine_net.load_state_dict(checkpoint['state_dict'])
            self.affine_net.cuda()
            print("Affine model is initialized!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            print("The Affine model is added, but not initialized, this should only take place when a complete checkpoint (including affine model) will be loaded")
        self.affine_net.eval()

    def set_cur_epoch(self,epoch=-1):
        """
        set current epoch

        :param epoch:
        :return:
        """
        if self.epoch !=epoch+1:
            self.print_every_epoch_flag=True
        self.epoch = epoch+1


    def set_loss_fn(self, loss_fn):
        """
        set loss function (disabled)

        :param loss_fn:
        :return:
        """
        pass


    def save_cur_mermaid_settings(self,params):
        """
        save the mermaid settings into task record folder

        :param params:
        :return:
        """
        if len(self.record_path):
            saving_path = os.path.join(self.record_path,'nonp_setting.json')
            params.write_JSON(saving_path, save_int=False)
            params.write_JSON_comments(saving_path.replace('.json','_comment.json'))


    def init_mermaid_env(self):
        """
        setup the mermaid environment
        * saving the settings into record folder
        * initialize model from model, criterion and related variables

        """
        spacing = self.spacing
        params = pars.ParameterDict()
        params.load_JSON( self.mermaid_net_json_pth) #''../easyreg/cur_settings_svf.json')
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
        model, criterion = mf.create_registration_model(model_name, params['model'], compute_inverse_map=self.compute_inverse_map)


        if use_map:
            # create the identity map [0,1]^d, since we will use a map-based implementation
            _id = py_utils.identity_map_multiN(self.img_sz, spacing)
            self.identityMap = torch.from_numpy(_id).cuda()
            if self.mermaid_low_res_factor is not None:
                # create a lower resolution map for the computations
                lowres_id = py_utils.identity_map_multiN(lowResSize, lowResSpacing)
                self.lowResIdentityMap = torch.from_numpy(lowres_id).cuda()

            resize_affine_input = all([sz != -1 for sz in self.affine_resoltuion[2:]])
            if resize_affine_input:
                self.affine_spacing = get_res_spacing_from_spacing(spacing,  self.img_sz, self.affine_resoltuion)
                affine_id = py_utils.identity_map_multiN(self.affine_resoltuion, self.affine_spacing)
                self.affineIdentityMap = torch.from_numpy(affine_id).cuda()

        self.lowRes_fn = partial(get_resampled_image, spacing=spacing, desiredSize=lowResSize, zero_boundary=False,identity_map=self.lowResIdentityMap)
        self.mermaid_unit_st = model.cuda()
        self.criterion = criterion
        self.mermaid_unit_st.associate_parameters_with_module()
        self.save_cur_mermaid_settings(params)




    def get_loss(self):
        """
        get the overall loss

        :return:
        """
        return self.overall_loss

    def __cal_sym_loss(self,rec_phiWarped):
        """
        compute the symmetric loss,
        :math: `loss_{sym} = \|(\varphi^{s t})^{-1} \circ(\varphi^{t s})^{-1}-i d\|_{2}^{2}`

        :param rec_phiWarped:the transformation map, including two direction ( s-t, t-s in batch dimension)
        :return: mean(`loss_{sym}`)
        """
        trans1 = STNFunction_ND_BCXYZ(self.spacing,zero_boundary=False)
        trans2 = STNFunction_ND_BCXYZ(self.spacing,zero_boundary=False)
        st_map = rec_phiWarped[:self.n_batch]
        ts_map = rec_phiWarped[self.n_batch:]
        identity_map  = self.identityMap[0:self.n_batch]
        trans_st  = trans1(identity_map,st_map)
        trans_st_ts = trans2(trans_st,ts_map)
        return torch.mean((identity_map- trans_st_ts)**2)

    def do_criterion_cal(self, ISource, ITarget,cur_epoch=-1):
        """
        get the loss according to mermaid criterion

        :param ISource: Source image with full size
        :param ITarget: Target image with full size
        :param cur_epoch: current epoch
        :return: overall loss (include sim, reg and sym(optional)), similarity loss and the regularization loss
        """
        # todo the image is not necessary be normalized to [0,1] here, just keep -1,1 would be fine
        ISource = (ISource + 1.) / 2.
        ITarget = (ITarget + 1.) / 2.
        low_moving = self.low_moving
        if self.mask_input_when_compute_loss and self.moving_mask is not None and self.target_mask is not None:
            ISource = ISource*self.moving_mask
            ITarget = ITarget*self.target_mask
            low_moving = low_moving*self.low_moving_mask

        loss_overall_energy, sim_energy, reg_energy = self.criterion(self.identityMap, self.rec_phiWarped, ISource,
                                                                     ITarget, low_moving,
                                                                     self.mermaid_unit_st.get_variables_to_transfer_to_loss_function(),
                                                                     None)
        if not self.using_sym_on:
            if self.print_count % self.print_loss_every_n_iter == 0 and cur_epoch>=0:
                print('the loss_over_all:{} sim_energy:{}, reg_energy:{}'.format(loss_overall_energy.item(),
                                                                                   sim_energy.item(),
                                                                                   reg_energy.item()))
        else:
            sym_energy = self.__cal_sym_loss(self.rec_phiWarped)
            sym_factor = self.sym_factor  # min(sigmoid_explode(cur_epoch,static=1, k=8)*0.01*gl_sym_factor,1.*gl_sym_factor) #static=5, k=4)*0.01,1) static=10, k=10)*0.01
            loss_overall_energy = loss_overall_energy + sym_factor * sym_energy
            if self.print_count % self.print_loss_every_n_iter == 0 and cur_epoch >= 0:
                print('the loss_over_all:{} sim_energy:{},sym_factor: {} sym_energy: {} reg_energy:{}'.format(
                    loss_overall_energy.item(),
                    sim_energy.item(),
                    sym_factor,
                    sym_energy.item(),
                    reg_energy.item()))
        if self.step_loss is not None:
            self.step_loss += loss_overall_energy
            loss_overall_energy = self.step_loss
        if self.cur_step<self.step-1:
            self.print_count -= 1
        self.print_count += 1
        return loss_overall_energy, sim_energy, reg_energy

    def set_mermaid_param(self,mermaid_unit,criterion, s, t, m,s_full=None):
        """
        set variables need to be passed into mermaid model and mermaid criterion

        :param mermaid_unit:  model created by mermaid
        :param criterion:  criterion create by mermaid
        :param s: source image (can be downsampled)
        :param t: target image (can be downsampled)
        :param m: momentum (can be downsampled)
        :param s_full: full resolution image ( to get better sampling results)
        :return:
        """
        mermaid_unit.set_dictionary_to_pass_to_integrator({'I0': s, 'I1': t,'I0_full':s_full})
        criterion.set_dictionary_to_pass_to_smoother({'I0': s, 'I1': t,'I0_full':s_full})
        mermaid_unit.m = m
        criterion.m = m

    def __freeze_param(self,params):
        """
        freeze the parameters during training

        :param params: the parameters to be trained
        :return:
        """
        for param in params:
            param.requires_grad = False

    def __active_param(self,params):
        """
        active the frozen parameters

        :param params: the parameters to be activated
        :return:
        """
        for param in params:
            param.requires_grad = True

    def get_inverse_map(self,use_01=False):
        """
        get the inverse map

        :param use_01: if ture, get the map in [0,1] else in [-1,1]
        :return: the inverse map
        """
        if use_01 or self.inverse_map is None:
            return self.inverse_map
        else:
            return self.inverse_map*2-1




    def init_mermaid_param(self,s):
        """
        initialize the  mermaid parameters

        :param s: source image taken as adaptive smoother input
        :return:
        """
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

    def do_mermaid_reg(self,mermaid_unit,criterion, s, t, m, phi,inv_map=None):
        """
        perform mermaid registrtion unit

        :param s: source image
        :param t: target image
        :param m: initial momentum
        :param phi: initial deformation field
        :param low_s: downsampled source
        :param low_t: downsampled target
        :param inv_map: inversed map
        :return:  warped image, transformation map
        """
        if self.mermaid_low_res_factor is not None:
            low_s, low_t = self.low_moving, self.low_target
            self.set_mermaid_param(mermaid_unit,criterion,low_s, low_t, m,s)
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


    def __get_momentum(self):
        momentum = self.mermaid_unit_st.m[:self.n_batch]
        return momentum

    def __get_adaptive_smoother_map(self):
        """
        get the adaptive smoother weight map from spatial-variant regualrizer model
        supported weighting type 'sqrt_w_K_sqrt_w' and 'w_K_w'
        for weighting type == 'w_k_w'
        :math:`\sigma^{2}(x)=\sum_{i=0}^{N-1} w^2_{i}(x) \sigma_{i}^{2}`
        for weighting type = 'sqrt_w_K_sqrt_w'
        :math:`\sigma^{2}(x)=\sum_{i=0}^{N-1} w_{i}(x) \sigma_{i}^{2}`

        :return: adapative smoother weight map `\sigma`
        """
        adaptive_smoother_map = self.mermaid_unit_st.smoother.get_deep_smoother_weights()
        weighting_type = self.mermaid_unit_st.smoother.weighting_type
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
        if weighting_type == 'w_K_w':
            adaptive_smoother_map = adaptive_smoother_map**2 # todo  add if judgement, this is true only when we use w_K_W

        smoother_map = adaptive_smoother_map*(gaussian_stds**2)
        smoother_map = torch.sqrt(torch.sum(smoother_map,1,keepdim=True))
        #_,smoother_map = torch.max(adaptive_smoother_map.detach(),dim=1,keepdim=True)
        self._display_stats(smoother_map.float(),'statistic for max_smoother map')
        return smoother_map

    def _display_stats(self, Ia, iname):
        """
        statistic analysis on variable, print min, mean, max and std

        :param Ia: the input variable
        :param iname: variable name
        :return:
        """

        Ia_min = Ia.min().detach().cpu().numpy()
        Ia_max = Ia.max().detach().cpu().numpy()
        Ia_mean = Ia.mean().detach().cpu().numpy()
        Ia_std = Ia.std().detach().cpu().numpy()

        print('{}:after: [{:.2f},{:.2f},{:.2f}]({:.2f})'.format(iname, Ia_min,Ia_mean,Ia_max,Ia_std))




    def get_extra_to_plot(self):
        """
        plot extra image, i.e. the initial weight map of rdmm model

        :return: extra image, name
        """
        if self.use_adaptive_smoother:
            # the last step adaptive smoother is returned, todo add the first stage smoother
            return self.__get_adaptive_smoother_map(), 'Inital_weight'
        else:
            return self.__get_momentum(), "Momentum"


    def __transfer_return_var(self,rec_IWarped,rec_phiWarped,affine_img):
        """
        normalize the image into [0,1] while map into [-1,1]

        :param rec_IWarped: warped image
        :param rec_phiWarped: transformation map
        :param affine_img: affine image
        :return:
        """
        return (rec_IWarped).detach(), (rec_phiWarped * 2. - 1.).detach(), ((affine_img+1.)/2.).detach()

    def affine_forward(self,moving, target=None):
        if self.using_affine_init:
            with torch.no_grad():
                toaffine_moving, toaffine_target = moving, target
                resize_affine_input = all([sz != -1 for sz in self.affine_resoltuion[2:]])
                if resize_affine_input:
                    toaffine_moving = get_resampled_image(toaffine_moving, self.spacing, self.affine_resoltuion, identity_map=self.affineIdentityMap)
                    toaffine_target = get_resampled_image(toaffine_target, self.spacing, self.affine_resoltuion, identity_map=self.affineIdentityMap)
                affine_img, affine_map, affine_param = self.affine_net(toaffine_moving, toaffine_target)
                self.affine_param = affine_param
                affine_map = (affine_map + 1) / 2.
                inverse_map = None
                if self.compute_inverse_map:
                    inverse_map = self.affine_net.get_inverse_map(use_01=True)
                if resize_affine_input:
                    affine_img = py_utils.compute_warped_image_multiNC(moving, affine_map, self.spacing, 1,
                                                                       zero_boundary=True, use_01_input=True)
                if self.using_physical_coord:
                    for i in range(self.dim):
                        affine_map[:, i] = affine_map[:, i] * self.spacing[i] / self.standard_spacing[i]
                    if self.compute_inverse_map:
                        for i in range(self.dim):
                            inverse_map[:, i] = inverse_map[:, i] * self.spacing[i] / self.standard_spacing[i]
                self.inverse_map = inverse_map
        else:
            num_b = moving.shape[0]
            affine_map = self.identityMap[:num_b].clone()
            if self.compute_inverse_map:
                self.inverse_map = self.identityMap[:num_b].clone()

            affine_img = moving
        return affine_img, affine_map


    def mutli_step_forward(self, moving,target=None,moving_mask=None, target_mask=None):
        """
        mutli-step mermaid registration

        :param moving: moving image with intensity [-1,1]
        :param target: target image with intensity [-1,1]
        :return: warped image with intensity[0,1], transformation map [-1,1], affined image [0,1] (if no affine trans used, return moving)
        """
        self.step_loss = None
        affine_img, affine_map = self.affine_forward(moving,target)
        warped_img = affine_img
        init_map = affine_map
        rec_IWarped = None
        rec_phiWarped = None
        moving_n = (moving + 1) / 2.  # [-1,1] ->[0,1]
        target_n = (target + 1) / 2.  # [-1,1] ->[0,1]


        self.low_moving = self.init_mermaid_param(moving_n)
        self.low_target = self.init_mermaid_param(target_n)
        self.moving_mask, self.target_mask = moving_mask, target_mask
        self.low_moving_mask, self.low_target_mask = None, None
        if self.mask_input_when_compute_loss and moving_mask is not None and target_mask is not None:
            self.low_moving_mask = self.init_mermaid_param(moving_mask)
            self.low_target_mask = self.init_mermaid_param(target_mask)

        for i in range(self.step):
            self.cur_step = i
            record_is_grad_enabled = torch.is_grad_enabled()
            if not self.optimize_momentum_network or self.epoch in self.epoch_list_fixed_momentum_network:
                torch.set_grad_enabled(False)
            if self.print_every_epoch_flag:
                if self.epoch in self.epoch_list_fixed_momentum_network:
                    print("In this epoch, the momentum network is fixed")
                if self.epoch in self.epoch_list_fixed_deep_smoother_network:
                    print("In this epoch, the deep regularizer network is fixed")
                self.print_every_epoch_flag = False
            input = torch.cat((warped_img, target), 1)
            m = self.momentum_net(input)
            if self.clamp_momentum:
                m=m.clamp(max=self.clamp_thre,min=-self.clamp_thre)
            torch.set_grad_enabled(record_is_grad_enabled)
            rec_IWarped, rec_phiWarped = self.do_mermaid_reg(self.mermaid_unit_st,self.criterion,moving_n, target_n, m, init_map, self.inverse_map)
            warped_img = rec_IWarped * 2 - 1  # [0,1] -> [-1,1]
            init_map = rec_phiWarped  # [0,1]
            self.rec_phiWarped = rec_phiWarped
            if  i < self.step - 1:
                self.step_loss, _, _ = self.do_criterion_cal(moving, target, self.epoch)

        if self.using_physical_coord:
            rec_phiWarped_tmp = rec_phiWarped.detach().clone()
            for i in range(self.dim):
                rec_phiWarped_tmp[:, i] = rec_phiWarped[:, i] * self.standard_spacing[i] / self.spacing[i]
            rec_phiWarped = rec_phiWarped_tmp
        self.overall_loss,_,_= self.do_criterion_cal(moving, target, cur_epoch=self.epoch)
        return self.__transfer_return_var(rec_IWarped, rec_phiWarped, affine_img)



    def mutli_step_sym_forward(self,moving, target=None,moving_mask=None, target_mask=None):
        """
         symmetric multi-step mermaid registration
         the "source" is concatenated by source and target, the "target" is concatenated by target and source
         then the multi-step forward is called

         :param moving: moving image with intensity [-1,1]
         :param target: target image with intensity [-1,1]
         :return: warped image with intensity[0,1], transformation map [-1,1], affined image [0,1] (if no affine trans used, return moving)
         """
        moving_sym = torch.cat((moving, target), 0)
        target_sym = torch.cat((target, moving), 0)
        moving_mask_sym, target_mask_sym = None, None
        if moving_mask is not None and target_mask is not None:
            moving_mask_sym = torch.cat((moving_mask, target_mask), 0)
            target_mask_sym = torch.cat((target_mask, moving_mask), 0)
        rec_IWarped, rec_phiWarped, affine_img = self.mutli_step_forward(moving_sym, target_sym,moving_mask_sym,target_mask_sym)
        return rec_IWarped[:self.n_batch], rec_phiWarped[:self.n_batch], affine_img[:self.n_batch]

    def get_affine_map(self,moving, target):
        """
        compute affine map from the affine registration network

        :param moving: moving image [-1, 1]
        :param target: target image [-1, 1]
        :return: affined image [-1,1]
        """
        with torch.no_grad():
            affine_img, affine_map, _ = self.affine_net(moving, target)
        return affine_map



    def get_step_config(self):
        """
        check if the multi-step, symmetric forward shoud be activated

        :return:
        """
        if self.is_train:
            self.step = self.multi_step if self.epoch > self.epoch_activate_multi_step else 1
            self.using_sym_on = True if self.epoch> self.epoch_activate_sym else False
        else:
            self.step = self.multi_step
            self.using_sym_on = False

    def forward(self, moving, target, moving_mask=None, target_mask=None):
        """
        forward the mermaid registration model

        :param moving: moving image intensity normalized in [-1,1]
        :param target: target image intensity normalized in [-1,1]
        :return: warped image with intensity[0,1], transformation map [-1,1], affined image [0,1] (if no affine trans used, return moving)
        """
        self.get_step_config()
        self.n_batch = moving.shape[0]
        if self.using_sym_on:
            if not self.print_count:
                print(" The mermaid network is in multi-step and symmetric mode, with step {}".format(self.step))
            return self.mutli_step_sym_forward(moving,target,moving_mask, target_mask)
        else:
            if not self.print_count:
                print(" The mermaid network is in multi-step mode, with step {}".format(self.step))
            return self.mutli_step_forward(moving, target,moving_mask, target_mask)


