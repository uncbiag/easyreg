
from .base_model import BaseModel
from model_pool.utils import *
try:
    from model_pool.nn_interpolation import get_nn_interpolation
except:
    pass
import mermaid.pyreg.finite_differences as fdt
from mermaid.pyreg.utils import compute_warped_image_multiNC







class MermaidBase(BaseModel):
    def initialize(self, opt):
        BaseModel.initialize(self,opt)




    def get_warped_img_map(self,img, phi):
        bilinear = Bilinear()
        warped_img_map = bilinear(img, phi)

        return warped_img_map

    def get_warped_label_map(self,label_map, phi, sched='nn'):
        if sched == 'nn':
            ###########TODO temporal comment for torch1 compatability
            try:
                print(" the cuda nn interpolation is used")
                warped_label_map = get_nn_interpolation(label_map, phi)
            except:
                warped_label_map = compute_warped_image_multiNC(label_map,phi,self.spacing,spline_order=0,zero_boundary=True,use_01_input=False)
            # check if here should be add assert
            assert abs(torch.sum(
                warped_label_map.detach() - warped_label_map.detach().round())) < 0.1, "nn interpolation is not precise"
        else:
            raise ValueError(" the label warpping method is not implemented")
        return warped_label_map


    def compute_jacobi_map(self,map,crop_boundary=True):
        """ here we compute the jacobi in numpy coord. It is consistant to jacobi in image coord only when
          the image direction matrix is identity."""
        from model_pool.global_variable import save_jacobi_map
        import SimpleITK as sitk
        if type(map) == torch.Tensor:
            map = map.detach().cpu().numpy()
        input_img_sz = [int(self.img_sz[i] * self.input_resize_factor[i]) for i in range(len(self.img_sz))]
        spacing = 2. / (np.array(input_img_sz) - 1)  # the disp coorindate is [-1,1]
        fd = fdt.FD_np(spacing)
        dfx = fd.dXc(map[:, 0, ...])
        dfy = fd.dYc(map[:, 1, ...])
        dfz = fd.dZc(map[:, 2, ...])
        jacobi_det = dfx * dfy * dfz
        if crop_boundary:
            crop_range = 5
            jacobi_det_croped = jacobi_det[:,crop_range:-crop_range,crop_range:-crop_range,crop_range:-crop_range]
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
        if save_jacobi_map:
            jacobi_abs_map = np.abs(jacobi_det)
            jacobi_neg_map = np.zeros_like(jacobi_det)
            jacobi_neg_map[jacobi_det<0] =1
            for i in range(jacobi_abs_map.shape[0]):
                jacobi_img = sitk.GetImageFromArray(jacobi_abs_map[i])
                jacobi_neg_img = sitk.GetImageFromArray(jacobi_neg_map[i])
                pth = os.path.join(self.record_path, self.fname_list[i] +'_{:04d}'.format(self.cur_epoch+1)+ 'jacobi_img.nii')
                n_pth = os.path.join(self.record_path, self.fname_list[i] +'_{:04d}'.format(self.cur_epoch+1)+ 'jacobi_neg_img.nii')
                sitk.WriteImage(jacobi_img, pth)
                sitk.WriteImage(jacobi_neg_img, n_pth)
            self.jacobi_map =jacobi_abs_map
        return jacobi_abs_mean, jacobi_num_mean

    def save_extra_fig(self, img, title):
        import SimpleITK as sitk
        num_img = img.shape[0]
        assert (num_img == len(self.fname_list))
        input_img_sz = [int(self.img_sz[i] * self.input_resize_factor[i]) for i in range(len(self.img_sz))]

        img = get_resampled_image(img, None, desiredSize=[num_img, 1] + input_img_sz, spline_order=1)
        img_np = img.cpu().numpy()
        for i in range(num_img):
            img_to_save = img_np[i, 0]
            fpath = os.path.join(self.record_path,
                                 self.fname_list[i] + '_{:04d}'.format(self.cur_epoch + 1) + title + '.nii.gz')
            img_to_save = sitk.GetImageFromArray(img_to_save)
            sitk.WriteImage(img_to_save, fpath)