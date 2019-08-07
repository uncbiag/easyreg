
from .base_mermaid import MermaidBase
from .losses import Loss
from .metrics import get_multi_metric
from model_pool.utils import *
try:
    from model_pool.nn_interpolation import get_nn_interpolation
except:
    pass
import SimpleITK as sitk
import mermaid.finite_differences as fdt
from mermaid.utils import compute_warped_image_multiNC

import mermaid.simple_interface as SI
import mermaid.fileio as FIO
class MermaidIter(MermaidBase):
    def name(self):
        return 'reg-unet'

    def initialize(self,opt):
        MermaidBase.initialize(self,opt)
        self.print_val_detail = opt['tsk_set']['print_val_detail']
        network_name =opt['tsk_set']['network_name']
        self.single_mod = True
        if network_name =='affine':
            self.affine_on = True
            self.svf_on = False
        elif network_name =='svf':
            self.affine_on = True
            self.svf_on = True
        self.si = SI.RegisterImagePair()
        self.im_io = FIO.ImageIO()
        self.criticUpdates = opt['tsk_set']['criticUpdates']
        self.loss_fn = Loss(opt)
        self.opt_optim = opt['tsk_set']['optim']
        self.step_count =0.
        self.opt_mermaid= self.opt['tsk_set']['reg']['mermaid_iter']
        self.use_init_weight = self.opt_mermaid[('use_init_weight',False,'whether to use init weight for RDMM registration')]
        self.init_weight = None
        self.nonp_model_name = self.opt_mermaid[('nonp_model_name','lddmm_adapt_smoother_map','the model name of the non-parametric registration')]
        self.setting_for_mermaid_affine = self.opt_mermaid[('mermaid_affine_json','','the json path for the setting for mermaid affine')]
        self.setting_for_mermaid_nonp = self.opt_mermaid[('mermaid_nonp_json','','the json path for the setting for mermaid non-parametric')]
        self.weights_for_fg = self.opt_mermaid[('weights_for_fg',[0,0,0,0,1.],'regularizer weight for the foregound area, this should be got from the mermaid_json file')]
        self.weights_for_bg = self.opt_mermaid[('weights_for_bg',[0,0,0,0,1.],'regularizer weight for the background area')]
        self.saved_mermaid_setting_path = None
        self.saved_affine_setting_path = None






    def set_input(self, data, is_train=True):
        data[0]['image'] =(data[0]['image'].cuda()+1)/2
        data[0]['label'] =data[0]['label'].cuda()
        moving, target, l_moving,l_target = get_pair(data[0])
        input = data[0]['image']
        self.img_sz  = list(moving.shape)[2:]
        self.input_img_sz = [int(self.img_sz[i] * self.input_resize_factor[i]) for i in range(len(self.img_sz))]
        self.spacing = self.opt['dataset'][('spacing', 1. / (np.array(self.input_img_sz) - 1),'spacing')]
        self.spacing = np.array(self.spacing) if type(self.spacing) is not np.ndarray else self.spacing
        self.moving = moving
        self.target = target
        self.l_moving = l_moving
        self.l_target = l_target
        self.input = input
        self.fname_list = list(data[1])
        #print(moving.shape,target.shape)



    def affine_optimization(self):
        self.si = SI.RegisterImagePair()

        self.si.set_light_analysis_on(True)
        extra_info={}
        extra_info['pair_name'] = self.fname_list[0]
        extra_info['batch_id'] = self.fname_list[0]
        af_sigma = self.opt_mermaid['affine']['sigma']
        self.si.opt = None
        self.si.set_initial_map(None)
        if self.saved_affine_setting_path is None:
            self.saved_affine_setting_path = self.save_setting(self.setting_for_mermaid_affine,self.record_path,'affine_setting.json')


        self.si.register_images(self.moving, self.target, self.spacing,extra_info=extra_info,LSource=self.l_moving,LTarget=self.l_target,
                                model_name='affine_map',
                                map_low_res_factor=1.0,
                                nr_of_iterations=100,
                                visualize_step=None,
                                optimizer_name='sgd',
                                use_multi_scale=True,
                                rel_ftol=0,
                                similarity_measure_type='lncc',
                                similarity_measure_sigma=af_sigma,
                                json_config_out_filename=os.path.join(self.record_path,'cur_settings_affine_output_tmp.json'),  #########################################
                                params =self.saved_affine_setting_path) #'../model_pool/cur_settings_affine_tmp.json'
        self.output = self.si.get_warped_image()
        self.phi = self.si.opt.optimizer.ssOpt.get_map()
        self.disp = self.si.opt.optimizer.ssOpt.model.Ab
        # self.phi = self.phi*2-1
        self.phi = self.phi.detach().clone()
        for i in range(self.dim):         #######################TODO #######################
            self.phi[:,i,...] = self.phi[:,i,...] *2/ ((self.input_img_sz[i]-1)*self.spacing[i]) -1.
        return self.output.detach_(), self.phi.detach_(), self.disp.detach_()


    def svf_optimization(self):
        affine_map = self.si.opt.optimizer.ssOpt.get_map()

        self.si =  SI.RegisterImagePair()
        self.si.set_light_analysis_on(True)
        extra_info = {}
        extra_info['pair_name'] = self.fname_list[0]
        extra_info['batch_id'] = self.fname_list[0]
        # self.si.opt.optimizer.ssOpt.set_source_label(self.l_moving)
        # LSource_warped = self.si.get_warped_label()
        # LSource_warped.detach_()


        self.si.opt = None
        self.si.set_initial_map(affine_map.detach())
        if self.saved_mermaid_setting_path is None:
            self.saved_mermaid_setting_path = self.save_setting(self.setting_for_mermaid_nonp,self.record_path)

        self.si.register_images(self.moving, self.target, self.spacing, extra_info=extra_info, LSource=self.l_moving,
                                LTarget=self.l_target,
                                map_low_res_factor=0.5,
                                model_name=self.nonp_model_name,
                                nr_of_iterations=100,
                                visualize_step=None,
                                optimizer_name='lbfgs_ls',
                                use_multi_scale=True,
                                rel_ftol=0,
                                similarity_measure_type='lncc',
                                similarity_measure_sigma=1,
                                json_config_out_filename=os.path.join(self.record_path,'cur_settings_mermaid_output.json'),
                                params=self.saved_mermaid_setting_path) #'../mermaid_settings/cur_settings_svf_dipr.json'
        self.disp = self.output
        self.output = self.si.get_warped_image()
        self.phi = self.si.opt.optimizer.ssOpt.get_map()
        for i in range(self.dim):         #######################TODO #######################
            self.phi[:,i,...] = self.phi[:,i,...] *2/ ((self.input_img_sz[i]-1)*self.spacing[i]) -1.
        return self.output.detach_(), self.phi.detach_(), self.disp.detach_()

    def modify_setting(self,params, stds, weights):
        if stds is not None:
            params['model']['registration_model']['forward_model']['smoother']['multi_gaussian_stds'] = stds
        if weights is not None:
            params['model']['registration_model']['forward_model']['smoother']['multi_gaussian_weights'] = weights
        return params

    def save_setting(self,path, output_path,fname='mermaid_setting.json', modify=False, stds=None, weights=None):
        params = pars.ParameterDict()
        params.load_JSON(path)
        if modify:
            params = self.modify_setting(params, stds, weights)
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, fname)
        params.write_JSON(output_path, save_int=False)
        return output_path

    # def save_setting(self,path, output_path,fname='mermaid_setting.json'):
    #     params = pars.ParameterDict()
    #     params.load_JSON(path)
    #     os.makedirs(output_path, exist_ok=True)
    #     output_path = os.path.join(output_path, fname)
    #     params.write_JSON(output_path, save_int=False)
    #     return output_path


    def init_weight_optimization(self):
        affine_map = self.si.opt.optimizer.ssOpt.get_map()

        self.si =  SI.RegisterImagePair()
        self.si.set_light_analysis_on(True)
        extra_info = {}
        extra_info['pair_name'] = self.fname_list[0]
        extra_info['batch_id'] = self.fname_list[0]
        # self.si.opt.optimizer.ssOpt.set_source_label(self.l_moving)
        # LSource_warped = self.si.get_warped_label()
        # LSource_warped.detach_()


        self.si.opt = None
        self.si.set_initial_map(affine_map.detach())
        if self.use_init_weight:
            init_weight = get_init_weight_from_label_map(self.l_moving, self.spacing,self.weights_for_bg,self.weights_for_fg)
            init_weight = compute_warped_image_multiNC(init_weight,affine_map,self.spacing,spline_order=1,zero_boundary=False)
            self.si.set_weight_map(init_weight.detach(), freeze_weight=True)

        self.nonp_model_name= 'lddmm_adapt_smoother_map'
        if self.saved_mermaid_setting_path is None:
            self.saved_mermaid_setting_path = self.save_setting(self.setting_for_mermaid_nonp,self.record_path,modify=True,weights=self.weights_for_fg)


        self.si.register_images(self.moving, self.target, self.spacing, extra_info=extra_info, LSource=self.l_moving,
                                LTarget=self.l_target,
                                map_low_res_factor=0.5,
                                model_name=self.nonp_model_name,
                                nr_of_iterations=100,
                                visualize_step=None,
                                optimizer_name='lbfgs_ls',
                                use_multi_scale=True,
                                rel_ftol=0,
                                similarity_measure_type='lncc',
                                similarity_measure_sigma=1,
                                json_config_out_filename=os.path.join(self.record_path,'cur_settings_mermaid_output.json'),
                                params=self.saved_mermaid_setting_path)
        self.disp = self.output
        self.output = self.si.get_warped_image()
        self.phi = self.si.opt.optimizer.ssOpt.get_map()
        for i in range(self.dim):         #######################TODO #######################
            self.phi[:,i,...] = self.phi[:,i,...] *2/ ((self.input_img_sz[i]-1)*self.spacing[i]) -1.
        return self.output.detach_(), self.phi.detach_(), self.disp.detach_()





    def forward(self,input=None):
        if self.affine_on and not self.svf_on:
            return self.affine_optimization()
        elif self.svf_on and not self.use_init_weight:
            self.affine_optimization()
            return self.svf_optimization()
        elif self.svf_on and self.use_init_weight:
            self.affine_optimization()
            return self.init_weight_optimization()






    # get image paths
    def get_image_paths(self):
        return self.fname_list





    def cal_val_errors(self):
        self.cal_test_errors()

    def cal_test_errors(self):
        self.get_evaluation()

    def get_evaluation(self):
        self.output, self.phi, self.disp= self.forward()
        self.warped_label_map = self.get_warped_label_map(self.l_moving,self.phi)
        warped_label_map_np= self.warped_label_map.detach().cpu().numpy()
        self.l_target_np= self.l_target.detach().cpu().numpy()

        self.val_res_dic = get_multi_metric(warped_label_map_np, self.l_target_np,rm_bg=False)
        self.jacobi_val = self.compute_jacobi_map(self.phi)
        print(" the current jcobi value of the phi is {}".format(self.jacobi_val))


    def get_extra_res(self):
        return self.jacobi_val

    def get_the_jacobi_val(self):
        return self.jacobi_val






    def __get_adaptive_smoother_map(self):
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
        self._display_stats(smoother_map.float(),'statistic for max_smoother map')
        return smoother_map

    def _display_stats(self, Ia, iname):

        Ia_min = Ia.min().detach().cpu().numpy()
        Ia_max = Ia.max().detach().cpu().numpy()
        Ia_mean = Ia.mean().detach().cpu().numpy()
        Ia_std = Ia.std().detach().cpu().numpy()

        print('{}:after: [{:.2f},{:.2f},{:.2f}]({:.2f})'.format(iname, Ia_min,Ia_mean,Ia_max,Ia_std))




    def get_extra_to_plot(self):
        if self.nonp_model_name=='lddmm_adapt_smoother_map':
            return self.__get_adaptive_smoother_map(), 'inital_weight'
        else:
            return None, None






    def save_fig(self,phase,standard_record=False,saving_gt=True):
        from model_pool.global_variable import save_extra_fig
        from model_pool.visualize_registration_results import  show_current_images
        visual_param={}
        visual_param['visualize'] = False
        visual_param['save_fig'] = True
        visual_param['save_fig_path'] = self.record_path
        visual_param['save_fig_path_byname'] = os.path.join(self.record_path, 'byname')
        visual_param['save_fig_path_byiter'] = os.path.join(self.record_path, 'byiter')
        visual_param['save_fig_num'] = 8
        visual_param['pair_path'] = self.fname_list
        visual_param['iter'] = phase+"_iter_" + str(self.iter_count)
        disp=None
        extra_title = 'disp'
        extraImage, extraName = self.get_extra_to_plot()

        if save_extra_fig and extraImage is not None:
            self.save_extra_fig(extraImage,extraName)


        if self.disp is not None and len(self.disp.shape)>2 and not self.svf_on:
            disp = ((self.disp[:,...]**2).sum(1))**0.5

        if self.svf_on:
            disp = self.disp[:,0,...]
            extra_title='affine'

        if self.jacobi_map is not None:
            disp = self.jacobi_map
            extra_title = 'jacobi det'
        show_current_images(self.iter_count, iS=self.moving,iT=self.target,iW=self.output,
                            iSL=self.l_moving,iTL=self.l_target, iWL=self.warped_label_map,
                            vizImages=disp, vizName=extra_title,phiWarped=self.phi,
                            visual_param=visual_param,extraImages=extraImage, extraName= extraName)





    def save_deformation(self):
        import nibabel as nib
        phi_np = self.phi.detach().cpu().numpy()
        for i in range(phi_np.shape[0]):
            phi = nib.Nifti1Image(phi_np[i], np.eye(4))
            nib.save(phi, os.path.join(self.record_path, self.fname_list[i]) + '_phi.nii.gz')


    def set_val(self):
        self.is_train = False

    def set_debug(self):
        self.is_train = False

    def set_test(self):
        self.is_train = False



