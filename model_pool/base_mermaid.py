from time import time
from .base_model import ModelBase
from model_pool.utils import *
import mermaid.finite_differences as fdt
from mermaid.utils import compute_warped_image_multiNC
import tools.image_rescale as  ires
from .metrics import get_multi_metric
import SimpleITK as sitk


class MermaidBase(ModelBase):
    """
    the base class of mermaid
    """

    def initialize(self, opt):
        """
        initialize env parameter in mermaid registration

        :param opt: ParameterDict, task setting
        :return:
        """
        ModelBase.initialize(self, opt)
        self.affine_on = False
        self.nonp_on = False
        self.afimg_or_afparam = None
        self.save_extra_3d_img = opt['tsk_set'][('save_extra_3d_img', False, 'save extra image')]
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
        self.warped_label_map = None
        if self.l_moving is not None:
            self.warped_label_map = self.get_warped_label_map(self.l_moving, self.phi, use_01=self.use_01)
            print("Not take IO cost into consideration, the testing time cost is {}".format(time() - s1))
            warped_label_map_np = self.warped_label_map.detach().cpu().numpy()
            l_target_np = self.l_target.detach().cpu().numpy()

            self.val_res_dic = get_multi_metric(warped_label_map_np,l_target_np, rm_bg=False)
        self.jacobi_val = self.compute_jacobi_map((self.phi).detach().cpu().numpy(), crop_boundary=True,
                                                  use_01=self.use_01)
        print("current batch jacobi is {}".format(self.jacobi_val))

    def compute_jacobi_map(self, map, crop_boundary=True, use_01=False,save_jacobi_map=False):
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
                pth = os.path.join(self.record_path,
                                   self.fname_list[i] + '_{:04d}'.format(self.cur_epoch + 1) + 'jacobi_img.nii')
                n_pth = os.path.join(self.record_path,
                                     self.fname_list[i] + '_{:04d}'.format(self.cur_epoch + 1) + 'jacobi_neg_img.nii')
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
        from tools.visualize_registration_results import show_current_images
        visual_param = {}
        visual_param['visualize'] = False
        visual_param['save_fig'] = True
        visual_param['save_fig_path'] = self.record_path
        visual_param['save_fig_path_byname'] = os.path.join(self.record_path, 'byname')
        visual_param['save_fig_path_byiter'] = os.path.join(self.record_path, 'byiter')
        visual_param['save_fig_num'] = 4
        visual_param['pair_path'] = self.fname_list
        visual_param['iter'] = phase + "_iter_" + str(self.iter_count)
        disp = None
        extra_title = 'disp'
        extraImage, extraName = self.get_extra_to_plot()

        if self.save_extra_3d_img and extraImage is not None:
            self.save_extra_img(extraImage, extraName)

        if self.afimg_or_afparam is not None and len(self.afimg_or_afparam.shape) > 2 and not self.nonp_on:
            raise ValueError("displacement field is removed from current version")
            # disp = ((self.afimg_or_afparam[:,...]**2).sum(1))**0.5

        if self.nonp_on:
            disp = self.afimg_or_afparam[:, 0, ...]
            extra_title = 'affine'

        if self.jacobi_map is not None:
            disp = self.jacobi_map
            extra_title = 'jacobi det'
        show_current_images(self.iter_count, iS=self.moving, iT=self.target, iW=self.output,
                            iSL=self.l_moving, iTL=self.l_target, iWL=self.warped_label_map,
                            vizImages=disp, vizName=extra_title, phiWarped=self.phi,
                            visual_param=visual_param, extraImages=extraImage, extraName=extraName)

    def _save_image_into_original_sz_with_given_reference(self, pair_path, phi, inverse_phi=None, use_01=False):
        """
        the images (moving, target, warped, transformation map, inverse transformation map world coord[0,1] ) are saved in record_path/original_sz

        :param pair_path: list, moving image path, target image path
        :param phi: transformation map BDXYZ
        :param inverse_phi: inverse transformation map BDXYZ
        :param use_01: indicate the transformation use [0,1] coord or [-1,1] coord
        :return:
        """
        spacing = self.spacing
        moving_list = pair_path[0]
        target_list = pair_path[1]
        phi = (phi + 1) / 2. if not use_01 else phi
        new_phi, warped, new_spacing = ires.resample_warped_phi_and_image(moving_list, phi, spacing)

        saving_original_sz_path = os.path.join(self.record_path, 'original_sz')
        os.makedirs(saving_original_sz_path, exist_ok=True)
        fname_list = list(self.fname_list)
        ires.save_transfrom(new_phi, new_spacing, saving_original_sz_path, fname_list)
        reference_list = pair_path[0]
        fname_list = [fname + '_warped' for fname in self.fname_list]
        ires.save_image_with_given_reference(warped, reference_list, saving_original_sz_path, fname_list)
        fname_list = [fname + '_moving' for fname in self.fname_list]
        ires.save_image_with_given_reference(None, reference_list, saving_original_sz_path, fname_list)
        reference_list = pair_path[1]
        fname_list = [fname + '_target' for fname in self.fname_list]
        ires.save_image_with_given_reference(None, reference_list, saving_original_sz_path, fname_list)
        if inverse_phi is not None:
            inverse_phi = (inverse_phi + 1) / 2. if not use_01 else inverse_phi
            new_inv_phi, inv_warped, new_spacing = ires.resample_warped_phi_and_image(target_list, inverse_phi, spacing)
            fname_list = [fname + '_inv' for fname in self.fname_list]
            ires.save_transfrom(new_inv_phi, new_spacing, saving_original_sz_path, fname_list)
            fname_list = [fname + '_inv_warped' for fname in self.fname_list]
            ires.save_image_with_given_reference(inv_warped, reference_list, saving_original_sz_path, fname_list)

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
        num_img = img.shape[0]
        assert (num_img == len(self.fname_list))
        input_img_sz = self.input_img_sz  # [int(self.img_sz[i] * self.input_resize_factor[i]) for i in range(len(self.img_sz))]
        img = get_resampled_image(img, self.spacing, desiredSize=[num_img, 1] + input_img_sz, spline_order=1)
        img_np = img.cpu().numpy()
        for i in range(num_img):
            img_to_save = img_np[i, 0]
            fpath = os.path.join(self.record_path,
                                 self.fname_list[i] + '_{:04d}'.format(self.cur_epoch + 1) + title + '.nii.gz')
            img_to_save = sitk.GetImageFromArray(img_to_save)
            img_to_save.SetSpacing(np.flipud(self.spacing))
            sitk.WriteImage(img_to_save, fpath)

    def save_deformation(self):
        """
        save deformation in [0,1] coord, no physical spacing is included

        :return:
        """
        if not self.affine_on:
            import nibabel as nib
            phi_np = self.phi.detach().cpu().numpy()
            phi_np = phi_np if self.use_01 else (phi_np + 1.) / 2.  # normalize the phi into 0, 1
            for i in range(phi_np.shape[0]):
                phi = nib.Nifti1Image(phi_np[i], np.eye(4))
                nib.save(phi, os.path.join(self.record_path, self.fname_list[i]) + '_phi.nii.gz')
        else:
            # todo the affine param is assumed in -1, 1 phi coord, to be fixed into 0,1 coord
            affine_param = self.afimg_or_afparam
            if isinstance(affine_param, list):
                affine_param = self.afimg_or_afparam[0]
            affine_param = affine_param.detach().cpu().numpy()
            for i in range(affine_param.shape[0]):
                np.save(os.path.join(self.record_path, self.fname_list[i]) + 'affine_param.npy', affine_param[i])
