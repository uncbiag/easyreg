
from .base_model import BaseModel
from .losses import Loss
from .metrics import get_multi_metric
from model_pool.utils import *
from model_pool.nn_interpolation import get_nn_interpolation
import SimpleITK as sitk
import mermaid.pyreg.finite_differences as fdt

import mermaid.pyreg.simple_interface as SI
import mermaid.pyreg.fileio as FIO
class MermaidIter(BaseModel):
    def name(self):
        return 'reg-unet'

    def initialize(self,opt):
        BaseModel.initialize(self,opt)
        self.print_val_detail = opt['tsk_set']['print_val_detail']
        self.input_img_sz = [int(self.img_sz[i]*self.input_resize_factor[i]) for i in range(len(self.img_sz))]
        self.spacing= opt['tsk_set'][('spacing',1. / (np.array(self.input_img_sz) - 1),'spacing')] # np.array([0.00501306, 0.00261097, 0.00261097])*2
        self.spacing = np.array(self.spacing) if type(self.spacing) is not np.ndarray else self.spacing
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






    def set_input(self, data, is_train=True):
        data[0]['image'] =(data[0]['image'].cuda()+1)/2
        data[0]['label'] =data[0]['label'].cuda()
        moving, target, l_moving,l_target = get_pair(data[0])
        input = data[0]['image']
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
                                json_config_out_filename='cur_settings_affine_output_tmp.json',  #########################################
                                params ='../model_pool/cur_settings_affine_tmp.json')
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

        self.si.register_images(self.moving, self.target, self.spacing, extra_info=extra_info, LSource=self.l_moving,
                                LTarget=self.l_target,
                                map_low_res_factor=0.5,
                                model_name='lddmm_shooting_map',
                                nr_of_iterations=100,
                                visualize_step=None,
                                optimizer_name='lbfgs_ls',
                                use_multi_scale=True,
                                rel_ftol=0,
                                similarity_measure_type='lncc',
                                similarity_measure_sigma=1,
                                json_config_out_filename='cur_settings_svf_output_tmp1.json',
                                params='../mermaid_settings/cur_settings_svf_dipr.json')
        self.disp = self.output
        self.output = self.si.get_warped_image()
        self.phi = self.si.opt.optimizer.ssOpt.get_map()
        for i in range(self.dim):         #######################TODO #######################
            self.phi[:,i,...] = self.phi[:,i,...] *2/ ((self.input_img_sz[i]-1)*self.spacing[i]) -1.
        return self.output.detach_(), self.phi.detach_(), self.disp.detach_()


    def forward(self,input=None):
        if self.affine_on and not self.svf_on:
            return self.affine_optimization()
        elif self.svf_on:
            self.affine_optimization()
            return self.svf_optimization()






    # get image paths
    def get_image_paths(self):
        return self.fname_list


    def get_warped_img_map(self,img, phi):
        bilinear = Bilinear()
        warped_img_map = bilinear(img, phi)

        return warped_img_map


    def get_warped_label_map(self,label_map, phi, sched='nn'):
        if sched == 'nn':
            warped_label_map = get_nn_interpolation(label_map, phi)
            # check if here should be add assert
            assert abs(torch.sum(
                warped_label_map.detach() - warped_label_map.detach().round())) < 0.1, "nn interpolation is not precise"
        else:
            raise ValueError(" the label warpping method is not implemented")
        return warped_label_map

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




    def save_fig(self,phase,standard_record=False,saving_gt=True):
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
        if self.disp is not None and len(self.disp.shape)>2 and not self.svf_on:
            disp = ((self.disp[:,...]**2).sum(1))**0.5


        if self.svf_on:
            disp = self.disp[:,0,...]
            extra_title='affine'
        show_current_images(self.iter_count,  self.moving, self.target,self.output, self.l_moving,self.l_target,self.warped_label_map,
                            disp, extra_title, self.phi, visual_param=visual_param)


    def compute_jacobi_map(self,map):
        """ here we compute the jacobi in numpy coord. It is consistant to jacobi in image coord only when
          the image direction matrix is identity."""
        if type(map) == torch.Tensor:
            map = map.detach().cpu().numpy()
        input_img_sz = [int(self.img_sz[i] * self.input_resize_factor[i]) for i in range(len(self.img_sz))]
        spacing = 2. / (np.array(input_img_sz) - 1)  # the disp coorindate is [-1,1]
        fd = fdt.FD_np(spacing)
        dfx = fd.dXc(map[:, 0, ...])
        dfy = fd.dYc(map[:, 1, ...])
        dfz = fd.dZc(map[:, 2, ...])
        jacobi_det = dfx * dfy * dfz
        # self.temp_save_Jacobi_image(jacobi_det,map)
        jacobi_abs = - np.sum(jacobi_det[jacobi_det < 0.])  #
        jacobi_num = np.sum(jacobi_det < 0.)
        print("debugging {},{},{}".format(np.sum(dfx < 0.), np.sum(dfy < 0.), np.sum(dfz < 0.)))
        print("the jacobi_value of fold points for current batch is {}".format(jacobi_abs))
        print("the number of fold points for current batch is {}".format(jacobi_num))
        jacobi_abs_mean = jacobi_abs / map.shape[0]
        jacobi_num_mean = jacobi_num / map.shape[0]
        return jacobi_abs_mean, jacobi_num_mean

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



