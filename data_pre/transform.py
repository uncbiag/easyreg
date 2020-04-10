import data_pre.transform_pool as bio_transform
import SimpleITK as sitk
import numpy as np

class Transform(object):
    def __init__(self,option, dim=3):
        self.transform_seq = []
        self.dim = dim
        self.option = option
        self.patch_size =option[('patch_size',[128, 128, 32], 'patch size')]
        self.transform_dic = {'balanced_random_crop':self.balanced_random_crop,
                              'my_random_crop': self.my_random_crop,
                              'my_balanced_random_crop': self.my_balanced_random_crop,
                              'random_rigid_transform':self.random_rigid_transform,
                              'bspline_transform':self.bspline_transform,
                              'gaussian_blur': self.gaussian_blur,
                              'bilateral_filter':self.bilateral_filter
                              }
        self.buffer={}


    def get_transform_seq(self, transform_name_seq):
        transform_name_seq = transform_name_seq
        if len(transform_name_seq) :
            self.transform_seq = [self.transform_dic[tf]() for tf in transform_name_seq]
        else:
            self.transform_seq = self.default_train_sched()
        return self.transform_seq



    def default_train_sched(self):
        self.transform_seq = [self.identity_trans()]
        option_default = self.option[('default', {}, 'get default transform setting')]
        using_bspline_deform = option_default[('using_bspline_deform',False, 'using bspline transform')]

        if using_bspline_deform:
            deform_target = option_default[('deform_target', 'padded','deform mode: global, local or padded')]
            deform_scale =  option_default[('deform_scale', 1.0, 'deform scale')]
            self.option['bspline_trans']['deform_scale'] = deform_scale

            if using_bspline_deform:
                if deform_target == 'global':
                    self.transform_seq.insert(0, self.bspline_transform())
                elif deform_target == 'local':
                    self.transform_seq.insert(1, self.bspline_transform())
                elif deform_target == 'padded':
                    self.transform_seq.insert(0, self.bspline_transform())
                    pre_padded_size = list(self.patch_size[i] + int(deform_scale) * 2 for i in range(self.dim))
                    pre_random_crop = bio_transform.BalancedRandomCrop(pre_padded_size)
                    post_crop = bio_transform.RandomCrop(self.patch_size, threshold=-1)
                    self.transform_seq = [pre_random_crop, self.bspline_transform(), post_crop]

        return self.transform_seq

    def balanced_random_crop(self):
        label_list = self.option['shared_info']['label_list']
        option_brc = self.option[('bal_rand_crop', {}, 'settings for balanced random crop')]
        sample_threshold = option_brc[('sample_threshold',[1]+[0.1]* (len(label_list)-1), 'sample threshold for each class')]
        balanced_random_crop = bio_transform.BalancedRandomCrop(self.patch_size, threshold=sample_threshold)
        return balanced_random_crop

    def identity_trans(self):

        identity_trans = bio_transform.IdentityTransform()
        return identity_trans


    def my_random_crop(self):
        option_mrc = self.option[('my_rand_crop', {}, 'settings for balanced random crop')]
        scale_ratio = option_mrc[('scale_ratio', 0.05, 'scale_ratio for patch sampling')]
        bg_label = option_mrc[('bg_label', 0, 'background label')]
        crop_bg_ratio = option_mrc[('crop_bg_ratio', 0.1, 'ratio of background crops')]
        label_density = self.option['shared_info']['label_density']
        img_size = self.option['shared_info']['img_size']
        from functools import reduce
        scale_dim = [img_size[i]/self.patch_size[i] for i in range(self.dim)]
        scale = reduce(lambda x,y:x*y, scale_dim)
        nbg_threshold = (1-label_density[bg_label]) * scale * scale_ratio
        my_random_crop = bio_transform.MyRandomCrop(self.patch_size, nbg_threshold, crop_bg_ratio=crop_bg_ratio, bg_label=bg_label)
        return my_random_crop



    def flicker_crop(self):
        option_fp = self.option[('flicker_crop', {}, 'settings for flicker_crop')]
        bg_label = option_fp[('bg_label', 0, 'background label')]
        adopt_bg_ratio = option_fp[('adopt_bg_ratio', 0.1, 'ratio of background crops')]
        img_size = self.option['shared_info']['img_size']
        from functools import reduce
        scale_dim = [img_size[i]/self.patch_size[i] for i in range(self.dim)]
        scale = reduce(lambda x,y:x*y, scale_dim)
        my_random_crop = bio_transform.FlickerCrop(self.patch_size, adopt_bg_ratio, bg_label=bg_label)
        return my_random_crop






    def my_balanced_random_crop(self):
        option_mbrc = self.option[('my_bal_rand_crop', {}, 'settings for balanced random crop')]
        scale_ratio = option_mbrc[('scale_ratio', 0.1, 'scale_ratio for patch sampling')]
        bg_th_ratio = option_mbrc[('bg_th_ratio', 0.0, 'th_ratio for bg ')]

        label_list = self.option['shared_info']['label_list']
        label_density = self.option['shared_info']['label_density']
        img_size = self.option['shared_info']['img_size']
        max_crop_num = self.option['shared_info']['num_crop_per_class_per_train_img']

        from functools import reduce
        scale_dim = [img_size[i]/self.patch_size[i] for i in range(self.dim)]
        scale = reduce(lambda x,y:x*y, scale_dim)
        sample_threshold = np.array(label_density) * scale * scale_ratio
        #np.clip(sample_threshold,0,0.06,out=sample_threshold)
        sample_threshold[0] = bg_th_ratio

        ########################################TODO ############################################
        #sample_threshold = sample_threshold*0.

        #################################################333
        my_balanced_random_crop = bio_transform.MyBalancedRandomCrop(self.patch_size, threshold=sample_threshold.tolist(),label_list =label_list,max_crop_num=max_crop_num )
        #print("Count init:", id(my_balanced_random_crop.np_coord_count))
        return my_balanced_random_crop









    def sitk_to_tensor(self):
        sitk_to_tensor = bio_transform.SitkToTensor()
        return sitk_to_tensor


    def random_rigid_transform(self):


        option_rrt = self.option[('rand_rigid_trans', {}, 'settins for random_rigid_transform')]
        transition = option_rrt[('transition',list([0.5]*self.dim), 'transtion for each dimension')]
        rotation = option_rrt[('rotation',list([0.0]*self.dim), 'rotation for each dimension')]
        rigid_ratio = option_rrt[('rigid_ratio',0.5, 'rigid ratio')]
        rigid_mode = option_rrt[('rigid_mode','both', 'three mode: both , img, seg')]
        rigid_transform = bio_transform.RandomRigidTransform(ratio=rigid_ratio, translation=transition,
                                                             rotation_angles=rotation, mode=rigid_mode)
        return rigid_transform


    def bspline_transform(self):
        # bspline setting
        option_bst = self.option[('bspline_trans', {}, 'settins for bspline_transform')]
        bspline_order = option_bst[('bspline_order',3, 'bspline order')]
        mesh_size = option_bst[('mesh_size',list([2])*self.dim, 'mesh size')]
        deform_ratio = option_bst[('deform_ratio',0.5, 'deform ratio')]
        deform_scale = option_bst[('deform_scale',1.0, 'deform scale')]
        interpolator = option_bst[('interpolator',"BSpline", 'interpolation sched, linear'
                                                             ' or Bspline')]


        bspline_transform = bio_transform.RandomBSplineTransform(
            mesh_size=mesh_size, bspline_order=bspline_order, deform_scale=deform_scale,
            ratio=deform_ratio, interpolator=sitk.sitkBSpline if interpolator == "BSpline" else sitk.sitkLinear)
        return bspline_transform

    def gaussian_blur(self):
        option_gb = self.option[('gaussian_blur', {}, 'settins for gaussian_blur')]
        blur_ratio = option_gb[('blur_ratio',1.0, 'blur ratio')]
        gaussian_var = option_gb[('gaussian_var',0.5, 'gaussian_var ')]
        gaussian_width = option_gb[('gaussian_width',1, 'gaussian_width')]
        maximumError = option_gb[('maximumError',0.9, 'maximumError')]

        gaussian_blur = bio_transform.GaussianBlur(
            variance=gaussian_var, maximumKernelWidth=gaussian_width, maximumError=maximumError, ratio=blur_ratio)
        return gaussian_blur


    def bilateral_filter(self):
        # Bilateral Filtering
        option_bf = self.option[('bilateral_filter', {}, 'settins for bilateral_filter')]
        bilateral_ratio = option_bf[('bilateral_ratio',1.0, 'bilateral_ratio ratio')]
        domain_sigma = option_bf[('domain_sigma',0.2, 'domain_sigma')]
        range_sigma = option_bf[('range_sigma',0.06, 'range_sigma')]
        number_of_range_gaussian_samples = option_bf[('number_of_range_gaussian_samples',50, 'number_of_range_gaussian_samples')]
        bilateral_filter = bio_transform.BilateralFilter(ratio=bilateral_ratio, domainSigma=domain_sigma,
                                                         rangeSigma=range_sigma,
                                                         numberOfRangeGaussianSamples=number_of_range_gaussian_samples)
        return bilateral_filter








