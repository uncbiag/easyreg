from .base_mermaid import MermaidBase
from .utils import *
import mermaid.utils as py_utils
import mermaid.simple_interface as SI
class MermaidIter(MermaidBase):
    def name(self):
        return 'mermaid-iter'

    def initialize(self,opt):
        """
        :param opt: ParameterDict, task settings
        :return:
        """
        MermaidBase.initialize(self,opt)
        method_name =opt['tsk_set']['method_name']
        if method_name =='affine':
            self.affine_on = True
            self.nonp_on = False
        elif method_name =='nonp':
            self.affine_on = True
            self.nonp_on = True
        elif method_name=='nonp_only':
            self.affine_on = False
            self.nonp_on = True
        self.si = SI.RegisterImagePair()
        self.opt_optim = opt['tsk_set']['optim']
        self.compute_inverse_map = opt['tsk_set']['reg'][('compute_inverse_map', False,"compute the inverse transformation map")]
        self.opt_mermaid= self.opt['tsk_set']['reg']['mermaid_iter']
        self.use_init_weight = self.opt_mermaid[('use_init_weight',False,'whether to use init weight for RDMM registration')]
        self.init_weight = None
        self.setting_for_mermaid_affine = self.opt_mermaid[('mermaid_affine_json','','the json path for the setting for mermaid affine')]
        self.setting_for_mermaid_nonp = self.opt_mermaid[('mermaid_nonp_json','','the json path for the setting for mermaid non-parametric')]
        nonp_settings = pars.ParameterDict()
        nonp_settings.load_JSON(self.setting_for_mermaid_nonp)
        self.nonp_model_name  = nonp_settings['model']['registration_model']['type']
        self.weights_for_fg = self.opt_mermaid[('weights_for_fg',[0,0,0,0,1.],'regularizer weight for the foregound area, this should be got from the mermaid_json file')]
        self.weights_for_bg = self.opt_mermaid[('weights_for_bg',[0,0,0,0,1.],'regularizer weight for the background area')]
        self.saved_mermaid_setting_path = None
        self.saved_affine_setting_path = None
        self.inversed_map = None
        self.use_01 = True







    def set_input(self, data, is_train=True):
        data[0]['image'] =(data[0]['image'].cuda()+1)/2
        if 'label' in data[0]:
            data[0]['label'] =data[0]['label'].cuda()
        moving, target, l_moving,l_target = get_reg_pair(data[0])
        input = data[0]['image']
        self.input_img_sz  = list(moving.shape)[2:]
        self.original_spacing = data[0]['original_spacing']
        self.original_im_sz =  data[0]['original_sz']
        self.spacing = data[0]['spacing'][0] if self.use_physical_coord else 1. / (np.array(self.input_img_sz) - 1)
        self.spacing = np.array(self.spacing) if type(self.spacing) is not np.ndarray else self.spacing
        self.moving = moving
        self.target = target
        self.l_moving = l_moving
        self.l_target = l_target
        self.input = input
        self.fname_list = list(data[1])
        self.pair_path = data[0]['pair_path']




    def affine_optimization(self):
        """
        call affine optimization registration in mermaid
        :return: warped image, transformation map, affine parameter, loss(None)
        """
        self.si = SI.RegisterImagePair()
        extra_info = pars.ParameterDict()
        extra_info['pair_name'] = self.fname_list
        af_sigma = self.opt_mermaid['affine']['sigma']
        self.si.opt = None
        self.si.set_initial_map(None)
        if self.saved_affine_setting_path is None:
            self.saved_affine_setting_path = self.save_setting(self.setting_for_mermaid_affine,self.record_path,'affine_setting.json')

        cur_affine_json_saving_path =(os.path.join(self.record_path,'cur_settings_affine.json'),os.path.join(self.record_path,'cur_settings_affine_comment.json'))
        self.si.register_images(self.moving, self.target, self.spacing,extra_info=extra_info,LSource=self.l_moving,LTarget=self.l_target,
                                visualize_step=None,
                                use_multi_scale=True,
                                rel_ftol=0,
                                similarity_measure_sigma=af_sigma,
                                json_config_out_filename=cur_affine_json_saving_path,  #########################################
                                params =self.saved_affine_setting_path) #'../easyreg/cur_settings_affine_tmp.json'

        self.output = self.si.get_warped_image()
        self.phi = self.si.opt.optimizer.ssOpt.get_map()
        self.phi = self.phi.detach().clone()
        # for i in range(self.dim):
        #     self.phi[:, i, ...] = self.phi[:, i, ...] / ((self.input_img_sz[i] - 1) * self.spacing[i])

        Ab = self.si.opt.optimizer.ssOpt.model.Ab

        if self.compute_inverse_map:
            inv_Ab = py_utils.get_inverse_affine_param(Ab.detach())
            identity_map = py_utils.identity_map_multiN([1, 1] + self.input_img_sz, self.spacing)
            self.inversed_map = py_utils.apply_affine_transform_to_map_multiNC(inv_Ab, torch.Tensor(identity_map).cuda())  ##########################3
            self.inversed_map = self.inversed_map.detach()
        self.afimg_or_afparam = Ab
        save_affine_param_with_easyreg_custom(self.afimg_or_afparam,self.record_path,self.fname_list,affine_compute_from_mermaid=True)
        return self.output.detach_(), self.phi.detach_(), self.afimg_or_afparam.detach_(), None





    def nonp_optimization(self):
        """
        call non-parametric image registration in mermaid
        if the affine registration is performed first, the affine transformation map would be taken as the initial map
        if the init weight on mutli-gaussian regularizer are set, the initial weight map would be computed from the label map, make sure the model called support spatial variant regularizer

        :return: warped image, transformation map, affined image, loss(None)
        """
        affine_map = None
        if self.affine_on:
            affine_map = self.si.opt.optimizer.ssOpt.get_map()

        self.si =  SI.RegisterImagePair()
        extra_info = pars.ParameterDict()
        extra_info['pair_name'] = self.fname_list
        self.si.opt = None
        if affine_map is not None:
            self.si.set_initial_map(affine_map.detach(), self.inversed_map)

        if self.use_init_weight:
            init_weight = get_init_weight_from_label_map(self.l_moving, self.spacing,self.weights_for_bg,self.weights_for_fg)
            init_weight = py_utils.compute_warped_image_multiNC(init_weight,affine_map,self.spacing,spline_order=1,zero_boundary=False)
            self.si.set_weight_map(init_weight.detach(), freeze_weight=True)

        if self.saved_mermaid_setting_path is None:
            self.saved_mermaid_setting_path = self.save_setting(self.setting_for_mermaid_nonp,self.record_path,"nonp_setting.json")
        cur_mermaid_json_saving_path =(os.path.join(self.record_path,'cur_settings_nonp.json'),os.path.join(self.record_path,'cur_settings_nonp_comment.json'))
        self.si.register_images(self.moving, self.target, self.spacing, extra_info=extra_info, LSource=self.l_moving,
                                LTarget=self.l_target,
                                visualize_step=None,
                                use_multi_scale=True,
                                rel_ftol=0,
                                compute_inverse_map=self.compute_inverse_map,
                                json_config_out_filename=cur_mermaid_json_saving_path,
                                params=self.saved_mermaid_setting_path) #'../mermaid_settings/cur_settings_svf_dipr.json'
        self.afimg_or_afparam = self.output # here return the affine image
        self.output = self.si.get_warped_image()
        self.phi = self.si.opt.optimizer.ssOpt.get_map()
        # for i in range(self.dim):
        #     self.phi[:,i,...] = self.phi[:,i,...]/ ((self.input_img_sz[i]-1)*self.spacing[i])

        if self.compute_inverse_map:
            self.inversed_map = self.si.get_inverse_map().detach()
        return self.output.detach_(), self.phi.detach_(), self.afimg_or_afparam.detach_() if self.afimg_or_afparam is not None else None, None


    def save_setting(self,path, output_path,fname='mermaid_setting.json'):
        """
        save the mermaid settings into task record folder
        :param path: path of mermaid setting file
        :param output_path: path of task record folder
        :param fname: saving name
        :return: saved setting path
        """
        params = pars.ParameterDict()
        params.load_JSON(path)
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, fname)
        params.write_JSON(output_path, save_int=False)
        return output_path




    def save_image_into_original_sz_with_given_reference(self):
        """
        save the image into original sz (the sz before resampling) and with the original physical settings, i.e. spacing, origin, orientation
        :return:
        """
        # the original image sz in one batch should be the same
        self._save_image_into_original_sz_with_given_reference(self.pair_path,self.phi, inverse_phis=self.inversed_map, use_01=self.use_01)




    def forward(self,input=None):
        if self.affine_on and not self.nonp_on:
            return self.affine_optimization()
        elif self.affine_on and self.nonp_on:
            self.affine_optimization()
            return self.nonp_optimization()

        else:
            return self.nonp_optimization()



    def cal_val_errors(self):
        self.cal_test_errors()

    def cal_test_errors(self):
        self.get_evaluation()



    def get_jacobi_val(self):
        """
        :return: the sum of absolute value of  negative determinant jacobi, the num of negative determinant jacobi voxels
        """
        return self.jacobi_val

    def get_the_jacobi_val(self):
        return self.jacobi_val






    def __get_adaptive_smoother_map(self):
        """
        get the adaptive smoother weight map from spatial-variant regualrizer model
        supported weighting type 'sqrt_w_K_sqrt_w' and 'w_K_w'
        for weighting type == 'w_k_w'
        :math:'\sigma^{2}(x)=\sum_{i=0}^{N-1} w^2_{i}(x) \sigma_{i}^{2}'
        for weighting type = 'sqrt_w_K_sqrt_w'
        :math:'\sigma^{2}(x)=\sum_{i=0}^{N-1} w_{i}(x) \sigma_{i}^{2}'
        :return: adapative smoother weight map \sigma
        """
        model =  self.si.opt.optimizer.ssOpt.model
        smoother =  self.si.opt.optimizer.ssOpt.model.smoother
        adaptive_smoother_map = model.local_weights.detach()
        gaussian_weights = smoother.get_gaussian_weights().detach()
        print(" the current global gaussian weight is {}".format(gaussian_weights))

        gaussian_stds = smoother.get_gaussian_stds().detach()
        print(" the current global gaussian stds is {}".format(gaussian_stds))
        view_sz = [1] + [len(gaussian_stds)] + [1] * len(self.spacing)
        gaussian_stds = gaussian_stds.view(*view_sz)
        weighting_type = smoother.weighting_type
        if weighting_type == 'w_K_w':
            adaptive_smoother_map = adaptive_smoother_map**2 # todo   this is only necessary when we use w_K_W

        smoother_map = adaptive_smoother_map*(gaussian_stds**2)
        smoother_map = torch.sqrt(torch.sum(smoother_map,1,keepdim=True))
        #_,smoother_map = torch.max(adaptive_smoother_map.detach(),dim=1,keepdim=True)
        self._display_stats(smoother_map.float(),'statistic for weighted smoother map')
        return smoother_map



    def __get_momentum(self):
        param =  self.si.get_model_parameters()
        return param['m'].detach()
    def _display_stats(self, Ia, iname):
        """
        statistic analysis on variable
        :param Ia: the input variable
        :param iname: variable name
        :return:
        """

        Ia_min = Ia.min().detach().cpu().numpy()
        Ia_max = Ia.max().detach().cpu().numpy()
        Ia_mean = Ia.mean().detach().cpu().numpy()
        Ia_std = Ia.std().detach().cpu().numpy()

        print('{}:the: [min {:.2f},mean {:.2f},max {:.2f}](std {:.2f})'.format(iname, Ia_min,Ia_mean,Ia_max,Ia_std))




    def get_extra_to_plot(self):
        """
        plot extra image, i.e. the initial weight map of rdmm model
        :return:
        """
        if self.nonp_on:
            if self.nonp_model_name=='lddmm_adapt_smoother_map':
                return self.__get_adaptive_smoother_map(), 'inital_weight'
            else:
                return self.__get_momentum(), "Momentum"
        else:
            return None, None




    def set_val(self):
        self.is_train = False

    def set_debug(self):
        self.is_train = False

    def set_test(self):
        self.is_train = False



