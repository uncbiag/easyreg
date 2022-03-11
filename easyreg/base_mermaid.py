from time import time
from .base_reg_model import RegModelBase
from .utils import *
import mermaid.finite_differences as fdt
from mermaid.utils import compute_warped_image_multiNC
import tools.image_rescale as  ires
from .metrics import get_multi_metric
import SimpleITK as sitk


class MermaidBase(RegModelBase):
    """
    the base class of mermaid
    """

    def initialize(self, opt):
        """
        initialize env parameter in mermaid registration

        :param opt: ParameterDict, task setting
        :return:
        """
        RegModelBase.initialize(self, opt)
        self.affine_on = False
        self.nonp_on = False
        self.afimg_or_afparam = None
        self.save_extra_running_resolution_3d_img = opt['tsk_set'][('save_extra_running_resolution_3d_img', False, 'save extra image')]
        self.save_original_resol_by_type = opt['tsk_set'][(
            'save_original_resol_by_type', [True, True, True, True, True, True, True, True],
            'save_original_resol_by_type, save_s, save_t, save_w, save_phi, save_w_inv, save_phi_inv, save_disp, save_extra')]
        self.eval_metric_at_original_resol = opt['tsk_set'][
            ('eval_metric_at_original_resol', False, "evaluate the metric at original resolution")]
        self.external_eval = opt['tsk_set'][
            ('external_eval', '', "evaluate the metric using external metric but should follow easyreg format")]
        self.use_01 = True

    def get_warped_label_map(self, label_map, phi, sched='nn', use_01=False):
        """
        get warped label map

        :param label_map: label map to warp
        :param phi: transformation map
        :param sched: 'nn' neareast neighbor
        :param use_01: indicate the input phi is in [0,1] coord; else  the phi is assumed to be [-1,1]
        :return: the warped label map
        """
        if sched == 'nn':
            ###########TODO fix with new cuda interface,  now comment for torch1 compatability
            # try:
            #     print(" the cuda nn interpolation is used")
            #     warped_label_map = get_nn_interpolation(label_map, phi)
            # except:
            #     warped_label_map = compute_warped_image_multiNC(label_map,phi,self.spacing,spline_order=0,zero_boundary=True,use_01_input=use_01)
            warped_label_map = compute_warped_image_multiNC(label_map, phi, self.spacing, spline_order=0,
                                                            zero_boundary=True, use_01_input=use_01)
            # check if here should be add assert
            assert abs(torch.sum(
                warped_label_map.detach() - warped_label_map.detach().round())) < 0.1, "nn interpolation is not precise"
        else:
            raise ValueError(" the label warpping method is not implemented")
        return warped_label_map

    def get_evaluation(self):
        """
        evaluate the transformation by compute overlap on label map and folding in transformation

        :return:
        """

        s1 = time()
        self.output, self.phi, self.afimg_or_afparam, _ = self.forward()
        self.inverse_phi = self.network.get_inverse_map()
        self.warped_label_map = None
        if self.l_moving is not None:
            self.warped_label_map = self.get_warped_label_map(self.l_moving, self.phi, use_01=self.use_01)
            if not self.eval_metric_at_original_resol:
                print("Not take IO cost into consideration, the testing time cost is {}".format(time() - s1))
                warped_label_map_np = self.warped_label_map.detach().cpu().numpy()
                l_target_np = self.l_target.detach().cpu().numpy()
            else:
                moving_l_reference_list = self.pair_path[2]
                target_l_reference_list = self.pair_path[3]
                num_s = len(target_l_reference_list)
                assert num_s==1, "when call evaluation in original resolution, the bach num should be set to 1"
                phi = (self.phi + 1) / 2. if not self.use_01 else self.phi
                _,_, warped_label_map_np,_ = ires.resample_warped_phi_and_image(None,None, moving_l_reference_list[0], target_l_reference_list[0], phi,self.spacing)
                warped_label_map_np  = warped_label_map_np.detach().cpu().numpy()
                lt = [sitk.GetArrayFromImage(sitk.ReadImage(f)) for f in target_l_reference_list]
                sz = [num_s, 1] + list(lt[0].shape)
                l_target_np= np.stack(lt, axis=0)
                l_target_np = l_target_np.reshape(*sz).astype(np.float32)
            self.val_res_dic = get_multi_metric(warped_label_map_np, l_target_np, rm_bg=False)

        else:
            self.val_res_dic = {}
        self.jacobi_val = self.compute_jacobi_map((self.phi).detach().cpu().numpy(), crop_boundary=True,
                                                  use_01=self.use_01)

        print("current batch jacobi is {}".format(self.jacobi_val))
        self.eval_extern_metric()

    def eval_extern_metric(self):
        if len(self.external_eval):
            from data_pre.reg_preprocess_example.dirlab_eval import eval_on_dirlab
            supported_metric = {"dirlab":eval_on_dirlab}
            phi = (self.phi + 1) / 2. if not self.use_01 else self.phi
            inverse_phi = (self.inverse_phi + 1) / 2. if not self.use_01 else self.inverse_phi
            supported_metric[self.external_eval](phi, inverse_phi, self.fname_list, self.pair_path,moving = self.moving, target=self.target, record_path= self.record_path)




    def compute_jacobi_map(self, map, crop_boundary=True, use_01=False, save_jacobi_map=False, appendix='3D'):
        """
        compute determinant jacobi on transformatiomm map,  the coordinate should be canonical.

        :param map: the transformation map
        :param crop_boundary: if crop the boundary, then jacobi analysis would only analysis on cropped map
        :param use_01: infer the input map is in[0,1]  else is in [-1,1]
        :return: the sum of absolute value of  negative determinant jacobi, the num of negative determinant jacobi voxels
        """
        if type(map) == torch.Tensor:
            map = map.detach().cpu().numpy()
        span = 1.0 if use_01 else 2.0
        spacing = self.spacing * span  # the disp coorindate is [-1,1]
        fd = fdt.FD_np(spacing)
        dfx = fd.dXc(map[:, 0, ...])
        dfy = fd.dYc(map[:, 1, ...])
        dfz = fd.dZc(map[:, 2, ...])
        jacobi_det = dfx * dfy * dfz
        if crop_boundary:
            crop_range = 5
            jacobi_det_croped = jacobi_det[:, crop_range:-crop_range, crop_range:-crop_range, crop_range:-crop_range]
            jacobi_abs_croped = - np.sum(jacobi_det_croped[jacobi_det_croped < 0.])  #
            jacobi_num_croped = np.sum(jacobi_det_croped < 0.)
            print("Cropped! the jacobi_value of fold points for current batch is {}".format(jacobi_abs_croped))
            print("Cropped! the number of fold points for current batch is {}".format(jacobi_num_croped))
        # self.temp_save_Jacobi_image(jacobi_det,map)
        jacobi_abs = - np.sum(jacobi_det[jacobi_det < 0.])  #
        jacobi_num = np.sum(jacobi_det < 0.)
        print("print folds for each channel {},{},{}".format(np.sum(dfx < 0.), np.sum(dfy < 0.), np.sum(dfz < 0.)))
        print("the jacobi_value of fold points for current batch is {}".format(jacobi_abs))
        print("the number of fold points for current batch is {}".format(jacobi_num))
        jacobi_abs_mean = jacobi_abs / map.shape[0]
        jacobi_num_mean = jacobi_num / map.shape[0]
        self.jacobi_map = None
        jacobi_abs_map = np.abs(jacobi_det)
        if save_jacobi_map:
            jacobi_neg_map = np.zeros_like(jacobi_det)
            jacobi_neg_map[jacobi_det < 0] = 1
            for i in range(jacobi_abs_map.shape[0]):
                jacobi_img = sitk.GetImageFromArray(jacobi_abs_map[i])
                jacobi_neg_img = sitk.GetImageFromArray(jacobi_neg_map[i])
                jacobi_img.SetSpacing(np.flipud(self.spacing))
                jacobi_neg_img.SetSpacing(np.flipud(self.spacing))
                jacobi_saving = os.path.join(self.record_path, appendix)
                os.makedirs(jacobi_saving, exist_ok=True)
                pth = os.path.join(jacobi_saving,
                                   self.fname_list[i] + "_iter_" + str(self.iter_count) + '_jacobi_img.nii')
                n_pth = os.path.join(jacobi_saving,
                                     self.fname_list[i] + "_iter_" + str(self.iter_count) + '_jacobi_neg_img.nii')
                sitk.WriteImage(jacobi_img, pth)
                sitk.WriteImage(jacobi_neg_img, n_pth)
        self.jacobi_map = jacobi_abs_map
        return jacobi_abs_mean, jacobi_num_mean

    def get_extra_to_plot(self):
        """
        extra image needs to be plot

        :return: image to plot, name
        """
        return None, None

    def save_fig(self, phase):
        """
        save 2d center slice from x,y, z axis, for moving, target, warped, l_moving (optional), l_target(optional), (l_warped)

        :param phase: train|val|test|debug
        :return:
        """
        from .visualize_registration_results import show_current_images
        visual_param = {}
        visual_param['visualize'] = False
        visual_param['save_fig'] = True
        visual_param['save_fig_path'] = self.record_path
        visual_param['save_fig_path_byname'] = os.path.join(self.record_path, 'byname')
        visual_param['save_fig_path_byiter'] = os.path.join(self.record_path, 'byiter')
        visual_param['save_fig_num'] = 4
        visual_param['pair_name'] = self.fname_list
        visual_param['iter'] = phase + "_iter_" + str(self.iter_count)
        disp = None
        extra_title = 'disp'
        extraImage, extraName = self.get_extra_to_plot()

        if self.save_extra_running_resolution_3d_img and extraImage is not None:
            self.save_extra_img(extraImage, extraName)

        if self.afimg_or_afparam is not None and len(self.afimg_or_afparam.shape) > 2 and not self.nonp_on:
            raise ValueError("displacement field is removed from current version")
            # disp = ((self.afimg_or_afparam[:,...]**2).sum(1))**0.5

        if self.nonp_on and self.afimg_or_afparam is not None:
            disp = self.afimg_or_afparam[:, 0, ...]
            extra_title = 'affine'

        if self.jacobi_map is not None and self.nonp_on:
            disp = self.jacobi_map
            extra_title = 'jacobi det'
        show_current_images(self.iter_count, iS=self.moving, iT=self.target, iW=self.output,
                            iSL=self.l_moving, iTL=self.l_target, iWL=self.warped_label_map,
                            vizImages=disp, vizName=extra_title, phiWarped=self.phi,
                            visual_param=visual_param, extraImages=extraImage, extraName=extraName)

    def _save_image_into_original_sz_with_given_reference(self, pair_path, phis, inverse_phis=None, use_01=False):
        """
        the images (moving, target, warped, transformation map, inverse transformation map world coord[0,1] ) are saved in record_path/original_sz

        :param pair_path: list, moving image path, target image path
        :param phis: transformation map BDXYZ
        :param inverse_phi: inverse transformation map BDXYZ
        :param use_01: indicate the transformation use [0,1] coord or [-1,1] coord
        :return:
        """
        save_original_resol_by_type = self.save_original_resol_by_type
        save_s, save_t, save_w, save_phi, save_w_inv, save_phi_inv, save_disp, save_extra_not_used_here = save_original_resol_by_type
        spacing = self.spacing
        moving_reference_list = pair_path[0]
        target_reference_list = pair_path[1]
        moving_l_reference_list = None
        target_l_reference_list = None
        if len(pair_path) == 4:
            moving_l_reference_list = pair_path[2]
            target_l_reference_list = pair_path[3]
        phis = (phis + 1) / 2. if not use_01 else phis
        saving_original_sz_path = os.path.join(self.record_path, 'original_sz')
        os.makedirs(saving_original_sz_path, exist_ok=True)
        for i in range(len(moving_reference_list)):
            moving_reference = moving_reference_list[i]
            target_reference = target_reference_list[i]
            moving_l_reference = moving_l_reference_list[i] if moving_l_reference_list else None
            target_l_reference = target_l_reference_list[i] if target_l_reference_list else None
            fname = self.fname_list[i]
            phi = phis[i:i+1]
            inverse_phi = inverse_phis[i:i+1] if inverse_phis is not None else None

            # new_phi, warped, warped_l, new_spacing = ires.resample_warped_phi_and_image(moving_reference, target_reference,
            #                                                                             moving_l_reference,target_l_reference, phi, spacing)
            new_phi, warped, warped_l, new_spacing = ires.resample_warped_phi_and_image(moving_reference,target_reference,
                                                                                        moving_l_reference,
                                                                                        target_l_reference, phi,
                                                                                        spacing)
            if save_phi or save_disp:
                if save_phi:
                    ires.save_transfrom(new_phi, new_spacing, saving_original_sz_path, [fname])
                if save_disp:
                    cur_fname = fname + '_disp'
                    id_map = gen_identity_map(warped.shape[2:], resize_factor=1., normalized=True).cuda()
                    id_map = (id_map[None] + 1) / 2.
                    disp = new_phi - id_map
                    ires.save_transform_with_reference(disp, new_spacing, [moving_reference], [target_reference],
                                                       path=saving_original_sz_path, fname_list=[cur_fname],
                                                       save_disp_into_itk_format=True)
                    del id_map, disp
            del new_phi, phi
            if save_w:
                cur_fname = fname + '_warped'
                ires.save_image_with_given_reference(warped, [target_reference], saving_original_sz_path, [cur_fname])
                if warped_l is not None:
                    cur_fname = fname + '_warped_l'
                    ires.save_image_with_given_reference(warped_l, [target_l_reference], saving_original_sz_path, [cur_fname])
            del warped
            if save_s:
                cur_fname = fname + '_moving'
                ires.save_image_with_given_reference(None, [moving_reference], saving_original_sz_path, [cur_fname])
                if moving_l_reference is not None:
                    cur_fname = fname + '_moving_l'
                    ires.save_image_with_given_reference(None, [moving_l_reference], saving_original_sz_path, [cur_fname])
            if save_t:
                cur_fname = fname + '_target'
                ires.save_image_with_given_reference(None, [target_reference], saving_original_sz_path, [cur_fname])
                if target_l_reference is not None:
                    cur_fname = fname + '_target_l'
                    ires.save_image_with_given_reference(None, [target_l_reference], saving_original_sz_path, [cur_fname])
            if inverse_phi is not None:
                inverse_phi = (inverse_phi + 1) / 2. if not use_01 else inverse_phi
                new_inv_phi, inv_warped, inv_warped_l, new_spacing = ires.resample_warped_phi_and_image(
                    target_reference,moving_reference, target_l_reference, moving_l_reference,inverse_phi, spacing)
                if save_phi_inv:
                    cur_fname = fname + '_inv'
                    ires.save_transfrom(new_inv_phi, new_spacing, saving_original_sz_path, [cur_fname])
                if save_w_inv:
                    cur_fname = fname + '_inv_warped'
                    ires.save_image_with_given_reference(inv_warped, [moving_reference], saving_original_sz_path,
                                                         [cur_fname])
                    if moving_l_reference is not None:
                        cur_fname = fname + '_inv_warped_l'
                        ires.save_image_with_given_reference(inv_warped_l, [moving_l_reference], saving_original_sz_path,
                                                             [cur_fname])
                if save_disp:
                    cur_fname = fname + '_inv_disp'
                    id_map = gen_identity_map(inv_warped.shape[2:], resize_factor=1., normalized=True).cuda()
                    id_map = (id_map[None] + 1) / 2.
                    inv_disp = new_inv_phi - id_map
                    ires.save_transform_with_reference(inv_disp, new_spacing, [target_reference], [moving_reference],
                                                       path=saving_original_sz_path, fname_list=[cur_fname],
                                                       save_disp_into_itk_format=True)
                    del id_map, inv_disp
                del new_inv_phi, inv_warped, inverse_phi

    def save_extra_img(self, img, title):
        """
        the propose of this function is for visualize the reg performance
        the extra image not include moving, target, warped, transformation map, which can refers to save save_fig_3D, save_deformation
        this function is for result analysis, for the saved image sz is equal to input_sz
        the physical information like  origin, orientation is not saved, todo, include this information

        :param img: extra image, BCXYZ
        :param title: extra image name
        :return:
        """
        import SimpleITK as sitk
        import nibabel as nib
        num_img = img.shape[0]
        assert (num_img == len(self.fname_list))
        input_img_sz = self.input_img_sz if not self.save_original_resol_by_type[-1] else self.original_im_sz[
            0].cpu().numpy().tolist()  # [int(self.img_sz[i] * self.input_resize_factor[i]) for i in range(len(self.img_sz))]
        # img = get_resampled_image(img, self.spacing, desiredSize=[num_img, 1] + input_img_sz, spline_order=1)
        img_np = img.cpu().numpy()
        for i in range(num_img):
            if img_np.shape[1] == 1:
                img_to_save = img_np[i, 0]
                fpath = os.path.join(self.record_path,
                                     self.fname_list[i] + '_{:04d}'.format(self.cur_epoch + 1) + title + '.nii.gz')
                img_to_save = sitk.GetImageFromArray(img_to_save)
                img_to_save.SetSpacing(np.flipud(self.spacing))
                sitk.WriteImage(img_to_save, fpath)
            else:
                multi_ch_img = nib.Nifti1Image(img_np[i], np.eye(4))
                fpath = os.path.join(self.record_path, self.fname_list[i] + '_{:04d}'.format(
                    self.cur_epoch + 1) + "_" + title + '.nii.gz')
                nib.save(multi_ch_img, fpath)

    def save_deformation(self):
        """
        save deformation in [0,1] coord, no physical spacing is included

        :return:
        """

        import nibabel as nib
        phi_np = self.phi.detach().cpu().numpy()
        phi_np = phi_np if self.use_01 else (phi_np + 1.) / 2.  # normalize the phi into 0, 1
        for i in range(phi_np.shape[0]):
            phi = nib.Nifti1Image(phi_np[i], np.eye(4))
            nib.save(phi, os.path.join(self.record_path, self.fname_list[i]) + '_phi.nii.gz')
        # if self.affine_on:
        #     # todo the affine param is assumed in -1, 1 phi coord, to be fixed into 0,1 coord
        #     affine_param = self.afimg_or_afparam
        #     if isinstance(affine_param, list):
        #         affine_param = self.afimg_or_afparam[0]
        #     affine_param = affine_param.detach().cpu().numpy()
        #     for i in range(affine_param.shape[0]):
        #         np.save(os.path.join(self.record_path, self.fname_list[i]) + '_affine_param.npy', affine_param[i])
