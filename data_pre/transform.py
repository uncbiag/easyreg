import data_pre.transform_pool as bio_transform
import SimpleITK as sitk


class Transform(object):
    def __init__(self,dim):
        self.transform_seq = []
        self.dim = dim
        self.transform_dic = {'balanced_random_crop':self.balanced_random_crop,
                              'random_rigid_transform':self.random_rigid_transform,
                              'bspline_transform':self.bspline_transform,
                              'gaussian_blur': self.gaussian_blur,
                              'bilateral_filter':self.bilateral_filter
                              }


    def get_transform_seq(self,transform_name_seq,option):
        if transform_name_seq is not None:
            self.transform_seq = [self.transform_dic[tf](option) for tf in transform_name_seq]
        else:
            self.transform_seq = self.default_train_sched(option)



    def default_train_sched(self, option):
        self.transform_seq = [self.balanced_random_crop(option)]
        option_default = option[('default', {}, 'get default transform setting')]
        using_bspline_deform = option_default[('using_bspline_deform',False, 'using bspline transform')]

        if using_bspline_deform:
            deform_target = option_default[('deform_target', 'padded','deform mode: global, local or padded')]
            deform_scale =  option[('deform_scale', 15.0, 'deform scale')]
            option['bspline_trans']['deform_scale'] = deform_scale

            if using_bspline_deform:
                if deform_target == 'global':
                    self.transform_seq.insert(0, self.bspline_transform(option))
                elif deform_target == 'local':
                    self.transform_seq.insert(1, self.bspline_transform(option))
                elif deform_target == 'padded':
                    self.transform_seq.insert(0, self.bspline_transform(option))
                    pre_padded_size = tuple(self.patch_size[i] + int(deform_scale) * 2 for i in range(self.dim))
                    pre_random_crop = bio_transform.BalancedRandomCrop(pre_padded_size)
                    post_crop = bio_transform.RandomCrop(self.patch_size, threshold=-1)
                    self.transform_seq = [pre_random_crop, self.bspline_transform(option), post_crop]

        return self.transform_seq


    def balanced_random_crop(self, option):
        num_label = option['shared_info']['num_label']
        option_brc = option[('bal_rand_crop', {}, 'settins for balanced random crop')]
        sample_threshold = option_brc[('sample_threshold',tuple([0.1]* num_label), 'sample threshold for each class')]
        balanced_random_crop = bio_transform.BalancedRandomCrop(self.patch_size, threshold=sample_threshold)
        return balanced_random_crop

    def sitk_to_tensor(self, option=None):
        sitk_to_tensor = bio_transform.SitkToTensor()
        return sitk_to_tensor


    def random_rigid_transform(self, option):


        option_rrt = option[('bal_rand_crop', {}, 'settins for random_rigid_transform')]
        transition = option_rrt[('transition',tuple([0.5]*self.dim), 'transtion for each dimension')]
        rotation = option_rrt[('rotation',tuple([0.0]*self.dim), 'rotation for each dimension')]
        rigid_ratio = option_rrt[('rigid_ratio',0.5, 'rigid ratio')]
        rigid_mode = option_rrt[('rigid_mode','both', 'three mode: both , img, seg')]
        rigid_transform = bio_transform.RandomRigidTransform(ratio=rigid_ratio, translation=transition,
                                                             rotation_angles=rotation, mode=rigid_mode)
        return rigid_transform


    def bspline_transform(self,option):
        # bspline setting
        option_bst = option[('bspline_transform', {}, 'settins for bspline_transform')]
        bspline_order = option_bst[('bspline_order',3, 'bspline order')]
        mesh_size = option_bst[('mesh_size',tuple([2])*self.dim, 'mesh size')]
        deform_ratio = option_bst[('deform_ratio',0.5, 'deform ratio')]
        deform_scale = option_bst[('deform_scale',15.0, 'deform scale')]
        interpolator = option_bst[('interpolator',"BSpline", 'interpolation sched, linear'
                                                             ' or Bspline')]


        bspline_transform = bio_transform.RandomBSplineTransform(
            mesh_size=mesh_size, bspline_order=bspline_order, deform_scale=deform_scale,
            ratio=deform_ratio, interpolator=sitk.sitkBSpline if interpolator == "BSpline" else sitk.sitkLinear)
        return bspline_transform

    def gaussian_blur(self):
        # Gaussian Blur
        blur_ratio = 1.0
        gaussian_var = 0.5
        gaussian_width = 1
        gaussian_blur = bio_transform.GaussianBlur(
            variance=gaussian_var, maximumKernelWidth=gaussian_width, maximumError=0.9, ratio=blur_ratio)
        return gaussian_blur


    def bilateral_filter(self):
        # Bilateral Filtering
        bilateral_ratio = 1.0
        domain_sigma = 0.2
        range_sigma = 0.06
        number_of_range_gaussian_samples = 50
        bilateral_filter = bio_transform.BilateralFilter(ratio=bilateral_ratio, domainSigma=domain_sigma,
                                                         rangeSigma=range_sigma,
                                                         numberOfRangeGaussianSamples=number_of_range_gaussian_samples)
        return bilateral_filter