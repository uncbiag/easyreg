""" classes of transformations for 3d simpleITK image 
"""
import SimpleITK as sitk
import numpy as np
import torch
import math


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
        img, seg = sample['image'], sample['segmentation']

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
        sample['segmentation'] = resampler.Execute(seg)

        return sample

class Normalization(object):
    """Normalize an image by setting its mean to zero and variance to one."""

    def __call__(self, sample):
        self.normalizeFilter = sitk.NormalizeImageFilter()
        print("Normalizing image...")
        img, seg = sample['image'], sample['segmentation']
        sample['image'] = self.normalizeFilter.Execute(img)

        return sample


class SitkToTensor(object):
    """Convert sitk image to 4D Tensors with shape(1, D, H, W)"""

    def __call__(self, sample):
        img, seg = sample['image'], sample['segmentation']
        img_np = sitk.GetArrayFromImage(img)
        seg_np = sitk.GetArrayFromImage(seg)
        # threshold image intensity to 0~1
        img_np[np.where(img_np > 1.0)] = 1.0
        img_np[np.where(img_np < 0.0)] = 0.0

        img_np = np.float32(img_np)
        seg_np = np.uint8(seg_np)
        img_np = np.expand_dims(img_np, axis=0)  # expand the channel dimension

        sample['image'] = torch.from_numpy(img_np)
        sample['segmentation'] = torch.from_numpy(seg_np)

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

        if np.random.rand(1)[0] < self.ratio:
            img, seg = sample['image'], sample['segmentation']

            # initialize a bspline transform
            bspline = sitk.BSplineTransformInitializer(img, self.mesh_size, self.bspline_order)

            # generate random displacement for control points, the deformation is scaled by deform_scale
            if self.random_mode == 'Normal':
                control_point_displacements = np.random.normal(0, self.deform_scale/2, len(bspline.GetParameters()))
            elif self.random_mode == 'Uniform':
                control_point_displacements = np.random.random(len(bspline.GetParameters())) * self.deform_scale

            control_point_displacements[0:int(len(control_point_displacements) / 3)] = 0  # remove z displacement
            bspline.SetParameters(control_point_displacements)

            # deform and resample image
            img_trans = resample(img, bspline, interpolator=self.interpolator, default_value=0.1)
            seg_trans = resample(seg, bspline, interpolator=sitk.sitkNearestNeighbor, default_value=0)

            sample['image'] = img_trans
            sample['segmentation'] = seg_trans

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

        if np.random.rand(1)[0] < self.ratio:
            img, seg = sample['image'], sample['segmentation']
            image_size = img.GetSize()
            image_spacing = img.GetSpacing()
            if self.rotation_center:
                rotation_center = self.rotation_center
            else:
                rotation_center = (np.array(image_size) // 2).tolist()


            rotation_center = img.TransformIndexToPhysicalPoint(rotation_center)

            rotation_radians_x = np.random.normal(0, self.rotation_angles[0]/2) * np.pi/180
            rotation_radians_y = np.random.normal(0, self.rotation_angles[1]/2) * np.pi/180
            rotation_radians_z = np.random.normal(0, self.rotation_angles[2]/2) * np.pi/180

            random_trans_x = np.random.normal(0, self.translation[0] / 2) * image_spacing[0]
            random_trans_y = np.random.normal(0, self.translation[1] / 2) * image_spacing[1]
            random_trans_z = np.random.normal(0, self.translation[2] / 2) * image_spacing[2]

            # initialize a bspline transform
            rigid_transform = sitk.Euler3DTransform(rotation_center, rotation_radians_x, rotation_radians_y, rotation_radians_z,
                                                    (random_trans_x, random_trans_y, random_trans_z))


            # deform and resample image

            if self.mode == 'both':
                img_trans = resample(img, rigid_transform, interpolator=self.interpolator, default_value=0.1)
                seg_trans = resample(seg, rigid_transform, interpolator=sitk.sitkNearestNeighbor, default_value=0)
            elif self.mode == 'img':
                img_trans = resample(img, rigid_transform, interpolator=self.interpolator, default_value=0.1)
                seg_trans = seg
            elif self.mode == 'seg':
                img_trans = img
                seg_trans = resample(seg, rigid_transform, interpolator=sitk.sitkNearestNeighbor, default_value=0)
            else:
                raise ValueError('Wrong rigid transformation mode :{}!'.format(self.mode))

            sample['image'] = img_trans
            sample['segmentation'] = seg_trans

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
        if np.random.rand(1)[0] < self.ratio:
            img, seg = sample['image'], sample['segmentation']
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
        if np.random.rand(1)[0] < self.ratio:
            img, _ = sample['image'], sample['segmentation']
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
        img, seg = sample['image'], sample['segmentation']
        size_old = img.GetSize()
        size_new = self.output_size

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
        sample['segmentation'] = seg_crop

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
        img, seg = sample['image'], sample['segmentation']
        size_old = img.GetSize()
        size_new = self.output_size

        contain_label = False

        roiFilter = sitk.RegionOfInterestImageFilter()
        roiFilter.SetSize([size_new[0], size_new[1], size_new[2]])

        # rand_ind = self.random_state.randint(3)  # random choose to focus on one class
        seg_np = sitk.GetArrayViewFromImage(seg)

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
        sample['segmentation'] = seg_crop
        sample['class'] = self.current_class

        # reset class tag
        self.current_class = self.current_class+1
        if self.current_class>3:
            self.current_class=0

        return sample



class MyBalancedRandomCrop(object):
    """Crop randomly the image in a sample. This is usually used for data augmentation

    Args:
        output_size (tuple or int): Desired output size. If int, cubic crop
            is made.
    """

    def __init__(self, output_size, threshold=0.01, random_state=None, num_label=3):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

        assert isinstance(threshold, (float, tuple))
        if isinstance(threshold, float):
            self.threshold = tuple([threshold]*num_label)
        else:
            assert len(threshold) == num_label
            self.threshold = threshold

        if random_state:
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState()

        self.num_label = num_label


    def __call__(self, sample):
        img, seg = sample['image'], sample['segmentation']
        size_old = img.GetSize()
        size_new = self.output_size
        cur_label = np.floor(self.random_state.rand()*self.num_label)

        roiFilter = sitk.RegionOfInterestImageFilter()
        roiFilter.SetSize([size_new[0], size_new[1], size_new[2]])

        # rand_ind = self.random_state.randint(3)  # random choose to focus on one class
        seg_np = sitk.GetArrayViewFromImage(seg)
        contain_label = False
        img_crop = None
        seg_crop = None

        # print(sample['name'])
        while not contain_label:
            # get the start crop coordinate in ijk

            start_i, start_j, start_k = random_3d_coordinates(np.array(size_old) - np.array(size_new),
                                                              self.random_state)
            roiFilter.SetIndex([start_i, start_j, start_k])

            seg_crop = roiFilter.Execute(seg)

            seg_crop_np = sitk.GetArrayViewFromImage(seg_crop)
            if np.sum(seg_crop_np==cur_label) / seg_crop_np.size > self.threshold[cur_label]:  # judge if the patch satisfy condition
                img_crop = roiFilter.Execute(img)
                contain_label = True


        sample['image'] = img_crop
        sample['segmentation'] = seg_crop
        sample['class'] = cur_label

        return sample









def random_3d_coordinates(range_3d, random_state=None):
    assert len(range_3d)==3
    if not random_state:
        random_state = np.random.RandomState()

    start_i = np.random.randint(0, range_3d[0])
    start_j = np.random.randint(0, range_3d[1])
    start_k = np.random.randint(0, range_3d[2])
    return start_i, start_j, start_k


class Partition(object):
    """partition a 3D volume into small 3D patches using the overlap tiling strategy described in paper:
    "U-net: Convolutional networks for biomedical image segmentation." by Ronneberger, Olaf, Philipp Fischer, 
    and Thomas Brox. In International Conference on Medical Image Computing and Computer-Assisted Intervention, 
    pp. 234-241. Springer, Cham, 2015.
    
    Note: BE CAREFUL about the order of dimensions for image:
            The simpleITK image are in order x, y, z
            The numpy array/torch tensor are in order z, y, x 
            
    
    :param tile_size (tuple of 3 or 1x3 np array): size of partitioned patches 
    :param self.overlap_size (tuple of 3 or 1x3 np array): the size of overlapping region at both end of each dimension
    :param padding_mode (tuple of 3 or 1x3 np array): the mode of numpy.pad when padding extra region for image
    :param mode: "pred": only image is partitioned; "eval": both image and segmentation are partitioned TODO
    """

    def __init__(self, tile_size, overlap_size, padding_mode='reflect', mode="pred"):
        self.tile_size = np.flipud(np.asarray(tile_size))  # flip the size order to match the numpy array(check the note)
        self.overlap_size = np.flipud(np.asarray(overlap_size))
        self.padding_mode = padding_mode
        self.mode = mode

    def __call__(self, sample):
        """
        :param image: (simpleITK image) 3D Image to be partitioned
        :param seg: (simpleITK image) 3D segmentation label mask to be partitioned
        :return: N partitioned image and label patches
            {'image': torch.Tensor Nx1xDxHxW, 'label': torch.Tensor Nx1xDxHxW, 'name': str } 
        """
        # get numpy array from simpleITK images
        image_np = sitk.GetArrayFromImage(sample['image'])
        seg_np = sitk.GetArrayFromImage(sample['segmentation'])
        self.image = sample['image']
        self.name = sample['name']
        self.image_size = np.array(image_np.shape)  # size of input image
        self.effective_size = self.tile_size - self.overlap_size * 2  # size effective region of tiles after cropping
        self.tiles_grid_size = np.ceil(self.image_size / self.effective_size).astype(int)  # size of tiles grid
        self.padded_size = self.effective_size * self.tiles_grid_size + self.overlap_size * 2 - self.image_size # size difference of padded image with original image


        image_padded = np.pad(image_np,
                              pad_width=((self.overlap_size[0], self.padded_size[0] - self.overlap_size[0]),
                                         (self.overlap_size[1], self.padded_size[1] - self.overlap_size[1]),
                                         (self.overlap_size[2], self.padded_size[2] - self.overlap_size[2])),
                              mode=self.padding_mode)

        if self.mode == 'eval':
            seg_padded = np.pad(seg_np,
                                  pad_width=((self.overlap_size[0], self.padded_size[0] - self.overlap_size[0]),
                                             (self.overlap_size[1], self.padded_size[1] - self.overlap_size[1]),
                                             (self.overlap_size[2], self.padded_size[2] - self.overlap_size[2])),
                                  mode=self.padding_mode)

        image_tile_list = []
        seg_tile_list = []
        for i in range(self.tiles_grid_size[0]):
            for j in range(self.tiles_grid_size[1]):
                for k in range(self.tiles_grid_size[2]):
                    image_tile_temp = image_padded[i * self.effective_size[0]:i * self.effective_size[0] + self.tile_size[0],
                                      j * self.effective_size[1]:j * self.effective_size[1] + self.tile_size[1],
                                      k * self.effective_size[2]:k * self.effective_size[2] + self.tile_size[2]]
                    image_tile_list.append(image_tile_temp)

                    if self.mode == 'eval':
                        seg_tile_temp = seg_padded[i * self.effective_size[0]:i * self.effective_size[0] + self.tile_size[0],
                                          j * self.effective_size[1]:j * self.effective_size[1] + self.tile_size[1],
                                          k * self.effective_size[2]:k * self.effective_size[2] + self.tile_size[2]]
                        seg_tile_list.append(seg_tile_temp)

        # sample['image'] = np.stack(image_tile_list, 0)
        # sample['segmentation'] = np.stack(seg_tile_list, 0)

        sample['image'] = torch.from_numpy(np.expand_dims(np.stack(image_tile_list, 0), axis=1))
        if self.mode == 'pred':
            sample['segmentation'] = torch.from_numpy(np.expand_dims(seg_np, axis=0))
        else:
            sample['segmentation'] = torch.from_numpy(np.expand_dims(np.stack(seg_tile_list, 0), axis=1))

        return sample

    def assemble(self, tiles, is_vote=False):
        """
        Assembles segmentation of small patches into the original size
        :param tiles: Nxhxdxw tensor contains N small patches of size hxdxw
        :param is_vote: 
        :return: a segmentation information
        """
        tiles = tiles.numpy()

        if is_vote:
            label_class= np.unique(tiles)
            seg_vote_array = np.zeros(np.insert(self.effective_size * self.tiles_grid_size + self.overlap_size * 2, 0, label_class.size), dtype=int)
            for i in range(self.tiles_grid_size[0]):
                for j in range(self.tiles_grid_size[1]):
                    for k in range(self.tiles_grid_size[2]):
                        ind = i * self.tiles_grid_size[1] * self.tiles_grid_size[2] + j * self.tiles_grid_size[2] + k
                        for label in label_class:
                            local_ind = np.where(tiles[ind]==label)  # get the coordinates in local patch of each class
                            global_ind = (local_ind[0] + i * self.effective_size[0],
                                          local_ind[1] + j * self.effective_size[1],
                                          local_ind[2] + k * self.effective_size[2])  # transfer into global coordinates
                            seg_vote_array[label][global_ind] += 1  # vote for each glass

            seg_reassemble = np.argmax(seg_vote_array, axis=0)[
                                       self.overlap_size[0]:self.overlap_size[0] + self.image_size[0],
                                       self.overlap_size[1]:self.overlap_size[1] + self.image_size[1],
                                       self.overlap_size[2]:self.overlap_size[2] + self.image_size[2]].astype(np.uint8)

            pass

        else:
            seg_reassemble = np.zeros(self.effective_size * self.tiles_grid_size)
            for i in range(self.tiles_grid_size[0]):
                for j in range(self.tiles_grid_size[1]):
                    for k in range(self.tiles_grid_size[2]):
                        ind = i * self.tiles_grid_size[1] * self.tiles_grid_size[2] + j * self.tiles_grid_size[2] + k
                        seg_reassemble[i * self.effective_size[0]:(i+1) * self.effective_size[0],
                        j * self.effective_size[1]:(j+1) * self.effective_size[1],
                        k * self.effective_size[2]:(k+1) * self.effective_size[2]] = \
                            tiles[ind][self.overlap_size[0]:self.tile_size[0]-self.overlap_size[0],
                              self.overlap_size[1]:self.tile_size[1]-self.overlap_size[1],
                              self.overlap_size[2]:self.tile_size[2]-self.overlap_size[2]]
            seg_reassemble = seg_reassemble[:self.image_size[0], :self.image_size[1], :self.image_size[2]]

        seg_image = sitk.GetImageFromArray(seg_reassemble)
        seg_image.CopyInformation(self.image)

        return seg_image

