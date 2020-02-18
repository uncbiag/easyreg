""" classes of transformations for 3d simpleITK image
"""
import threading

import SimpleITK as sitk
import numpy as np
import torch
import math
import random
import time
from math import floor


class Resample(object):
    """Resample the volume in a sample to a given voxel size

    Args:
        voxel_size (float or tuple): Desired output size.
        If float, output volume is isotropic.
        If tuple, output voxel size is matched with voxel size
        Currently only support linear interpolation method
    """

    def __init__(self, voxel_size):
        assert isinstance(voxel_size, (float, tuple))
        if isinstance(voxel_size, float):
            self.voxel_size = (voxel_size, voxel_size, voxel_size)
        else:
            assert len(voxel_size) == 3
            self.voxel_size = voxel_size

    def __call__(self, sample):
        img, seg = sample['image'], sample['label']

        old_spacing = img.GetSpacing()
        old_size = img.GetSize()

        new_spacing = self.voxel_size

        new_size = []
        for i in range(3):
            new_size.append(int(math.ceil(old_spacing[i] * old_size[i] / new_spacing[i])))
        new_size = tuple(new_size)

        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(1)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)

        # resample on image
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetOutputDirection(img.GetDirection())
        print("Resampling image...")
        sample['image'] = resampler.Execute(img)

        # resample on segmentation
        resampler.SetOutputOrigin(seg.GetOrigin())
        resampler.SetOutputDirection(seg.GetDirection())
        print("Resampling segmentation...")
        sample['label'] = resampler.Execute(seg)

        return sample

class Normalization(object):
    """Normalize an image by setting its mean to zero and variance to one."""

    def __call__(self, sample):
        self.normalizeFilter = sitk.NormalizeImageFilter()
        print("Normalizing image...")
        img, seg = sample['image'], sample['label']
        sample['image'] = self.normalizeFilter.Execute(img)

        return sample


class SitkToTensor(object):
    """Convert sitk image to 4D Tensors with shape(1, D, H, W)"""

    def __call__(self, sample):
        img, seg = sample['image'], sample['label']
        img_np = sitk.GetArrayFromImage(img)
        seg_np = sitk.GetArrayFromImage(seg)
        # threshold image intensity to 0~1
        img_np[np.where(img_np > 1.0)] = 1.0
        img_np[np.where(img_np < 0.0)] = 0.0

        img_np = np.float32(img_np)
        seg_np = np.uint8(seg_np)
        img_np = np.expand_dims(img_np, axis=0)  # expand the channel dimension

        sample['image'] = torch.from_numpy(img_np)
        sample['label'] = torch.from_numpy(seg_np)

        return sample


class RandomBSplineTransform(object):
    """
    Apply random BSpline Transformation to a 3D image
    check https://itk.org/Doxygen/html/classitk_1_1BSplineTransform.html for details of BSpline Transform
    """

    def __init__(self, mesh_size=(3,3,3), bspline_order=2, deform_scale=1.0, ratio=0.5, interpolator=sitk.sitkLinear,
                 random_mode = 'Normal'):
        self.mesh_size = mesh_size
        self.bspline_order = bspline_order
        self.deform_scale = deform_scale
        self.ratio = ratio  # control the probability of conduct transform
        self.interpolator = interpolator
        self.random_mode = random_mode

    def __call__(self, sample):
        random_state = np.random.RandomState(int(time.time()))

        if np.random.rand(1)[0] < self.ratio:
            img, seg = sample['image'], sample['label']

            # initialize a bspline transform
            bspline = sitk.BSplineTransformInitializer(img, self.mesh_size, self.bspline_order)

            # generate random displacement for control points, the deformation is scaled by deform_scale
            if self.random_mode == 'Normal':
                control_point_displacements = random_state.normal(0, self.deform_scale/2, len(bspline.GetParameters()))
            elif self.random_mode == 'Uniform':
                control_point_displacements = random_state.random(len(bspline.GetParameters())) * self.deform_scale

            control_point_displacements[0:int(len(control_point_displacements) / 3)] = 0  # remove z displacement
            bspline.SetParameters(control_point_displacements)

            # deform and resample image
            img_trans = resample(img, bspline, interpolator=self.interpolator, default_value=0.1)
            seg_trans = resample(seg, bspline, interpolator=sitk.sitkNearestNeighbor, default_value=0)

            sample['image'] = img_trans
            sample['label'] = seg_trans

        return sample

class RandomRigidTransform(object):
    """
    Apply random similarity Transformation to a 3D image
    """

    def __init__(self, ratio=1.0, rotation_center=None, rotation_angles=(0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0),
                 interpolator=sitk.sitkLinear, mode='both'):
        self.rotation_center = rotation_center
        self.rotation_angles = rotation_angles
        self.translation = translation
        self.interpolator = interpolator
        self.ratio = ratio
        self.mode = mode

    def __call__(self, sample):
        random_state = np.random.RandomState(int(time.time()))


        if random_state.rand(1)[0] < self.ratio:
            img, seg = sample['image'], sample['label']
            image_size = img.GetSize()
            image_spacing = img.GetSpacing()
            if self.rotation_center:
                rotation_center = self.rotation_center
            else:
                rotation_center = (np.array(image_size) // 2).tolist()


            rotation_center = img.TransformIndexToPhysicalPoint(rotation_center)

            rotation_radians_x = random_state.normal(0, self.rotation_angles[0]/2) * np.pi/180
            rotation_radians_y = random_state.normal(0, self.rotation_angles[1]/2) * np.pi/180
            rotation_radians_z = random_state.normal(0, self.rotation_angles[2]/2) * np.pi/180

            random_trans_x = random_state.normal(0, self.translation[0] / 2) * image_spacing[0]
            random_trans_y = random_state.normal(0, self.translation[1] / 2) * image_spacing[1]
            random_trans_z = random_state.normal(0, self.translation[2] / 2) * image_spacing[2]

            # initialize a bspline transform
            rigid_transform = sitk.Euler3DTransform(rotation_center, rotation_radians_x, rotation_radians_y, rotation_radians_z,
                                                    (random_trans_x, random_trans_y, random_trans_z))


            # deform and resample image

            if self.mode == 'both':
                img_trans = resample(img, rigid_transform, interpolator=self.interpolator, default_value=0.1)
                seg_trans = resample(seg, rigid_transform, interpolator=sitk.sitkNearestNeighbor, default_value=0)
            elif self.mode == 'image':
                img_trans = resample(img, rigid_transform, interpolator=self.interpolator, default_value=0.1)
                seg_trans = seg
            elif self.mode == 'label':
                img_trans = img
                seg_trans = resample(seg, rigid_transform, interpolator=sitk.sitkNearestNeighbor, default_value=0)
            else:
                raise ValueError('Wrong rigid transformation mode :{}!'.format(self.mode))

            sample['image'] = img_trans
            sample['label'] = seg_trans

        return sample

class IdentityTransform(object):
    """Identity transform that do nothing"""

    def __call__(self, sample):
        return sample


def resample(image, transform, interpolator=sitk.sitkBSpline, default_value=0.0):
    """Resample a transformed image"""
    reference_image = image
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)

class GaussianBlur(object):
    def __init__(self, variance=0.5, maximumKernelWidth=1, maximumError=0.9, ratio=1.0):
        self.ratio = ratio
        self.variance = variance
        self.maximumKernelWidth = maximumKernelWidth
        self.maximumError = maximumError

    def __call__(self, sample):
        random_state = np.random.RandomState(int(time.time()))
        if random_state.rand() < self.ratio:
            img, seg = sample['image'], sample['label']
            sample['image'] = sitk.DiscreteGaussian(
                img, variance=self.variance, maximumKernelWidth=self.maximumKernelWidth, maximumError=self.maximumError,
                useImageSpacing=False)
        return sample

class BilateralFilter(object):
    def __init__(self, domainSigma=0.5, rangeSigma=0.06, numberOfRangeGaussianSamples=50, ratio=1.0):
        self.domainSigma = domainSigma
        self.rangeSigma = rangeSigma
        self.numberOfRangeGaussianSamples = numberOfRangeGaussianSamples
        self.ratio = ratio

    def __call__(self, sample):
        random_state = np.random.RandomState(int(time.time()))
        if random_state.rand(1)[0] < self.ratio:
            img, _ = sample['image'], sample['label']
            sample['image'] = sitk.Bilateral(img, domainSigma=self.domainSigma, rangeSigma=self.rangeSigma,
                                             numberOfRangeGaussianSamples=self.numberOfRangeGaussianSamples)
        return sample

class RandomCrop(object):
    """Crop randomly the image in a sample. This is usually used for data augmentation

    Args:
        output_size (tuple or int): Desired output size. If int, cubic crop
            is made.
    """

    def __init__(self, output_size, threshold=-0, random_state=None):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size
        self.threshold = threshold
        if random_state:
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState()

    def __call__(self, sample):
        """
                the input patch sz and output_size is defined in itk coord

        :param sample:
        :return:
        """
        img, seg = sample['image'], sample['label']
        size_old = img.GetSize()
        size_new = self.output_size
        self.random_state = np.random.RandomState(int(time.time()))

        contain_label = False

        roiFilter = sitk.RegionOfInterestImageFilter()
        roiFilter.SetSize([size_new[0], size_new[1], size_new[2]])

        # print(sample['name'])
        while not contain_label:
            # get the start crop coordinate in ijk

            start_i = self.random_state.randint(0, size_old[0] - size_new[0])
            start_j = self.random_state.randint(0, size_old[1] - size_new[1])
            start_k = self.random_state.randint(0, size_old[2] - size_new[2])

            # start_i = torch.IntTensor(1).random_(0, size_old[0] - size_new[0])[0]
            # start_j = torch.IntTensor(1).random_(0, size_old[1] - size_new[1])[0]
            # start_k = torch.IntTensor(1).random_(0, size_old[2] - size_new[2])[0]

            # print(sample['name'], start_i, start_j, start_k)
            roiFilter.SetIndex([start_i, start_j, start_k])

            seg_crop = roiFilter.Execute(seg)

            # statFilter = sitk.StatisticsImageFilter()
            # statFilter.Execute(seg_crop)
            #
            # # will iterate until a sub volume containing label is extracted
            # if statFilter.GetSum() >= 1:
            #     contain_label = True

            seg_crop_np = sitk.GetArrayViewFromImage(seg_crop)
            # center_ind = np.array(seg_crop_np.shape)//2-1
            # if seg_crop_np[center_ind[0], center_ind[1], center_ind[2]] > 0:
            #     contain_label = True
            if np.sum(seg_crop_np)/seg_crop_np.size > self.threshold:
                contain_label = True

        img_crop = roiFilter.Execute(img)
        sample['image'] = img_crop
        sample['label'] = seg_crop

        return sample


class BalancedRandomCrop(object):
    """Crop randomly the image in a sample. This is usually used for data augmentation

    Args:
        output_size (tuple or int): Desired output size. If int, cubic crop
            is made.
    """

    def __init__(self, output_size, threshold=0.01, random_state=None):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

        assert isinstance(threshold, (float, tuple))
        if isinstance(threshold, float):
            self.threshold = (threshold, threshold, threshold)
        else:
            assert len(threshold) == 2
            self.threshold = threshold

        if random_state:
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState()

        self.current_class = 1  # tag that which class should be focused on currently


    def __call__(self, sample):
        """
                the input patch sz and output_size is defined in itk coord

        :param sample:
        :return:
        """
        img, seg = sample['image'], sample['label']
        size_old = img.GetSize()
        size_new = self.output_size
        self.random_state = np.random.RandomState(int(time.time()))

        contain_label = False

        roiFilter = sitk.RegionOfInterestImageFilter()
        roiFilter.SetSize([size_new[0], size_new[1], size_new[2]])

        contain_label = False

        if self.current_class == 0:  # random crop a patch
            start_i, start_j, start_k = random_3d_coordinates(np.array(size_old) - np.array(size_new), self.random_state)
            roiFilter.SetIndex([start_i, start_j, start_k])
            seg_crop = roiFilter.Execute(seg)

        elif self.current_class == 1:  # crop a patch where class 1 in main
            i = 0
            # print(sample['name'])
            while not contain_label:
                # get the start crop coordinate in ijk

                start_i, start_j, start_k = random_3d_coordinates(np.array(size_old) - np.array(size_new),
                                                                  self.random_state)
                roiFilter.SetIndex([start_i, start_j, start_k])

                seg_crop = roiFilter.Execute(seg)

                seg_crop_np = sitk.GetArrayViewFromImage(seg_crop)
                if np.sum(seg_crop_np==1) / seg_crop_np.size > self.threshold[0]:  # judge if the patch satisfy condition
                    contain_label = True
                i = i + 1

        else:  # crop a patch where class 2 in main
            # print(sample['name'])
            i = 0
            while not contain_label:
                # get the start crop coordinate in ijk

                start_i, start_j, start_k = random_3d_coordinates(np.array(size_old) - np.array(size_new),
                                                                  self.random_state)

                roiFilter.SetIndex([start_i, start_j, start_k])

                seg_crop = roiFilter.Execute(seg)

                seg_crop_np = sitk.GetArrayViewFromImage(seg_crop)
                if np.sum(seg_crop_np == 2) / seg_crop_np.size > self.threshold[1]:  # judge if the patch satisfy condition
                    contain_label = True
                i = i + 1
                # print(sample['name'], 'case: ', rand_ind, 'trying: ', i)
        # print([start_i, start_j, start_k])


        roiFilter.SetIndex([start_i, start_j, start_k])

        seg_crop = roiFilter.Execute(seg)
        img_crop = roiFilter.Execute(img)

        sample['image'] = img_crop
        sample['label'] = seg_crop
        sample['class'] = self.current_class

        # reset class tag
        self.current_class = self.current_class+1
        if self.current_class>3:
            self.current_class=0

        return sample



class MyRandomCrop(object):
    """Crop randomly the image in a sample. This is usually used for data augmentation

    Args:
        output_size (tuple or int): Desired output size. If int, cubic crop
            is made.
    """

    def __init__(self, output_size, nbg_threshold, crop_bg_ratio=0.1, bg_label=0,random_state=None):
        self.bg_label=  bg_label
        self.crop_bg_ratio = crop_bg_ratio
        """ expect ratio of crop backgound, assume background domain other labels"""
        self.nbg_threshold = nbg_threshold

        assert isinstance(output_size, (int, tuple,list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) >1
            self.output_size = output_size

        if random_state:
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState()



    def __call__(self, sample):
        """
                the input patch sz and output_size is defined in itk coord
        :param sample:
        :return:
        """
        img, seg = sample['image'], sample['label']
        size_old = img.GetSize()
        size_new = self.output_size
        roiFilter = sitk.RegionOfInterestImageFilter()
        roiFilter.SetSize(size_new)
        size_new = np.flipud(size_new)
        size_old = np.flipud(size_old)
        self.random_state = np.random.RandomState(int(time.time()))

        crop_once =  self.random_state.rand()< self.crop_bg_ratio
        seg_np = sitk.GetArrayViewFromImage(seg)
        contain_label = False
        start_coord = None
        nbg_ratio = 0 # ratio of non-bg label

        # print(sample['name'])
        while not contain_label :
            # get the start crop coordinate in ijk
            start_coord = random_nd_coordinates(np.array(size_old) - np.array(size_new),
                                                              self.random_state)
            seg_crop_np = cropping(seg_np,start_coord,size_new)
            bg_ratio = np.sum(seg_crop_np==self.bg_label) / seg_crop_np.size
            nbg_ratio =1.0- bg_ratio
            if nbg_ratio > self.nbg_threshold:  # judge if the patch satisfy condition
                contain_label = True
            elif crop_once:
                break

        start_coord = np.flipud(start_coord).tolist()
        roiFilter.SetIndex(start_coord)
        seg_crop = roiFilter.Execute(seg)
        if not isinstance(img,list):
            img_crop = roiFilter.Execute(img)
        else:
            img_crop = [roiFilter.Execute(im) for im in img]
        trans_sample={}
        trans_sample['image'] = img_crop
        trans_sample['label'] = seg_crop
        trans_sample['label_selected'] = -1
        trans_sample['start_coord']= tuple(start_coord)
        trans_sample['threshold'] = nbg_ratio

        return trans_sample









class FlickerCrop(object):
    """Crop randomly the image in a sample. This is usually used for data augmentation

    Args:
        output_size (tuple or int): Desired output size. If int, cubic crop
            is made.
    """

    def __init__(self, output_size, adopt_bg_ratio,  bg_label=0,random_state=None):
        self.bg_label=  bg_label
        self.adopt_bg_ratio = adopt_bg_ratio
        """ expect ratio of crop backgound, assume background domain other labels"""

        assert isinstance(output_size, (int, tuple,list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) >1
            self.output_size = output_size

        if random_state:
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState()



    def __call__(self, sample):
        """
                the input patch sz and output_size is defined in itk coord

        :param sample:
        :return:
        """
        img, seg = sample['image'], sample['label']
        if not isinstance(img,list):
            size_old = img.GetSize()
        else:
            size_old = img[0].GetSize()
        self.random_state = np.random.RandomState(int(time.time()))
        size_new = self.output_size
        roiFilter = sitk.RegionOfInterestImageFilter()
        roiFilter.SetSize(size_new)
        size_new = np.flipud(size_new)
        size_old = np.flipud(size_old)

        crop_once =  self.random_state.rand()< self.adopt_bg_ratio
        seg_np = sitk.GetArrayViewFromImage(seg)
        contain_label = False
        start_coord = None
        nbg_ratio = 0 # ratio of non-bg label

        # print(sample['name'])
        while not contain_label :
            # get the start crop coordinate in ijk
            start_coord = random_nd_coordinates(np.array(size_old) - np.array(size_new),
                                                              self.random_state)
            seg_crop_np = cropping(seg_np,start_coord,size_new)
            bg_ratio = np.sum(seg_crop_np==self.bg_label) / seg_crop_np.size
            nbg_ratio =1.0- bg_ratio
            if nbg_ratio > self.nbg_threshold:  # judge if the patch satisfy condition
                contain_label = True
            elif crop_once:
                break

        start_coord = np.flipud(start_coord).tolist()
        roiFilter.SetIndex(start_coord)
        seg_crop = roiFilter.Execute(seg)
        img_crop = roiFilter.Execute(img)
        trans_sample={}
        trans_sample['image'] = img_crop
        trans_sample['label'] = seg_crop
        trans_sample['label_selected'] = -1
        trans_sample['start_coord']= tuple(start_coord)
        trans_sample['threshold'] = nbg_ratio

        return trans_sample



class MyBalancedRandomCrop(object):
    """Crop randomly the image in a sample. This is usually used for data augmentation

    Args:
        output_size (tuple or int): Desired output size. If int, cubic crop
            is made.

    """

    def __init__(self, output_size, threshold, random_state=None, label_list=(),max_crop_num = -1):



        #assert max_crop_num==-1, "dataloader bugs, not fixed now"
        self.num_label=  len(label_list)
        self.label_list = label_list
        self.max_crop_on = max_crop_num>0
        self.max_crop_num = max_crop_num
        assert isinstance(output_size, (int, tuple,list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) >1
            self.output_size = output_size # the given outputsize is in numpy cord

        assert isinstance(threshold, (float, tuple,list))
        if isinstance(threshold, float):
            self.threshold = tuple([threshold]*self.num_label)
        else:
            #assert sum(np.array(threshold)!=0) == self.num_label
            self.threshold = threshold

        if random_state:
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState()
        self.cur_label_id = random.randint(0,self.num_label-1)
        ### if the max_crop_num>0, then a numpy of shape [], the coodinate is in numpy style
        print("Init my current transfrom: {}".format(threading.current_thread()))
        if self.max_crop_on:
            self.np_coord_buffer = np.zeros((self.num_label, self.max_crop_num, 3)).astype(np.int32)
            self.np_coord_count = np.zeros(self.num_label).astype(np.int32)

    def __call__(self, sample, rand_id=-1):
        """
        the input patch sz and output_size is defined in itk coord
        :param sample:if the img in sample is a list, then return a list of img list otherwise return single image
        :return:
        """
        if rand_id >=0:
            cur_label_id  = rand_id%self.num_label
        else:
            raise ValueError("should not happen in this case")
        #print("id(self): {}  , cur_label_id:{}".format(id(self),cur_label_id))
        is_numpy = False
        #self.random_state =np.random.RandomState(rand_id)
        self.random_state = np.random.RandomState(int(time.time()))
        # the size coordinate system here is according to the itk coordinate

        img, seg = sample['image'], sample['label']
        cur_label = int(self.label_list[cur_label_id])

        if isinstance(img,list):
            if not isinstance(img[0],np.ndarray):
                size_old = img[0].GetSize()
            else:
                is_numpy = True
                #cur_label = cur_label_id

                size_old = np.flipud(list(img[0].shape)) #itkcoord
        else:
            if not isinstance(img,np.ndarray):
                size_old = img.GetSize() # itkcoord
            else:
                is_numpy = True

        size_new = self.output_size #itk coord


        #cur_label_id = self.random_state.randint(self.num_label)
        if not is_numpy:
            roiFilter = sitk.RegionOfInterestImageFilter()
            roiFilter.SetSize(size_new)
            seg_np = sitk.GetArrayViewFromImage(seg)
        else:
            seg_np = seg.copy()


        label_ratio =0

        if self.max_crop_on and self.np_coord_count[cur_label_id] >= self.max_crop_num:
            ins_id = self.np_coord_count[cur_label_id] % self.max_crop_num
            start_coord = self.np_coord_buffer[cur_label_id, ins_id, :]
            size_new = np.flipud(size_new)
            #print("this is signal")
        else:
            # here the coordinate system transfer from the sitk to numpy
            size_new = np.flipud(size_new)
            size_old = np.flipud(size_old)

            # rand_ind = self.random_state.randint(3)  # random choose to focus on one class


            contain_label = False
            start_coord = None
            count = 0
            # print(sample['name'])
            while not contain_label:
                # get the start crop coordinate in ijk, the operation is did in the numpy coordinate
                start_coord = random_nd_coordinates(np.array(size_old) - np.array(size_new),
                                                                  self.random_state)
                seg_crop_np = cropping(seg_np,start_coord,size_new)
                label_ratio = np.sum(seg_crop_np==cur_label) / float(seg_crop_np.size)

                count += 1
                if count>10000:
                    print("Warning!!!!!, no crop")
                    print(cur_label_id)
                    print(self.label_list)

                if label_ratio >= self.threshold[cur_label]:  # judge if the patch satisfy condition
                    contain_label = True

            if self.max_crop_on:
                self.np_coord_buffer[cur_label_id, self.np_coord_count[cur_label_id], :] = start_coord
        if self.max_crop_on:
            #print("coord count: {}--{}--{}--{}".format(self.np_coord_count, threading.current_thread(), id(self.np_coord_count), id(self)))
            self.np_coord_count[cur_label_id] += 1




        if is_numpy:
            after_coord = [start_coord[i] + size_new[i] for i in range(len(size_new))]
            img_crop =[]
            if isinstance(img, list):
                for im in img:
                    img_crop += [im[start_coord[0]:after_coord[0], start_coord[1]:after_coord[1], start_coord[2]:after_coord[2]].copy()]
            seg_crop =seg_np[start_coord[0]:after_coord[0], start_coord[1]:after_coord[1],
                             start_coord[2]:after_coord[2]].copy()
            img_crop = np.stack(img_crop,0)
            seg_crop =np.expand_dims(seg_crop,0)
            # transfer the numpy coordinate into itk coordinate
            start_coord = np.flipud(start_coord).tolist()

        # now transfer the system into the sitk system
        else:
            start_coord = np.flipud(start_coord).tolist()
            roiFilter.SetIndex(start_coord)
            seg_crop = roiFilter.Execute(seg)
            if not isinstance(img,list):
                img_crop = roiFilter.Execute(img)
            else:
                img_crop = [roiFilter.Execute(im) for im in img]
        trans_sample={}
        trans_sample['image'] = img_crop
        trans_sample['label'] = seg_crop
        trans_sample['label_selected'] = cur_label
        trans_sample['start_coord']= tuple(start_coord) # is always given in itk coord
        trans_sample['threshold'] = label_ratio

        if rand_id<0:
            self.cur_label_id = cur_label_id + 1
            self.cur_label_id = self.cur_label_id if self.cur_label_id < self.num_label else 0

        return trans_sample



def cropping(img,start_coord,size_new):
    if len(start_coord)==2:
        return img[start_coord[0]:start_coord[0]+size_new[0],start_coord[1]:start_coord[1]+size_new[1]]
    elif len(start_coord)==3:
        return img[start_coord[0]:start_coord[0]+size_new[0],start_coord[1]:start_coord[1]+size_new[1],
                    start_coord[2]: start_coord[2] + size_new[2]]

def random_nd_coordinates(range_nd, random_state=None):
    # if not random_state:
    #     random_state = np.random.RandomState()
    dim = len(range_nd)
    return [random_state.randint(0, range_nd[i]) for i in range(dim)]




def random_3d_coordinates(range_3d, random_state=None):
    assert len(range_3d)==3
    if not random_state:
        random_state = np.random.RandomState()

    return [random_state.randint(0, range_3d[i]) for i in range(3)]




