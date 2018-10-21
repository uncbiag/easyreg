import os
import numpy as np
import time
import SimpleITK as sitk
import subprocess

from model_pool.nifty_reg_utils import nifty_reg_affine, nifty_reg_resample

record_path = '/playpen/zyshen/debugs/'
from model_pool.utils import factor_tuple


def smooth_and_resample(image, shrink_factor, smoothing_sigma):
    """
    Args:
        image: The image we want to resample.
        shrink_factor: A number greater than one, such that the new image's size is original_size/shrink_factor.
        smoothing_sigma: Sigma for Gaussian smoothing, this is in physical (image spacing) units, not pixels.
    Return:
        Image which is a result of smoothing the input and then resampling it using the given sigma and shrink factor.
    """
    smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigma)

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(sz / float(shrink_factor) + 0.5) for sz in original_size]
    new_spacing = [((original_sz - 1) * original_spc) / (new_sz - 1)
                   for original_sz, original_spc, new_sz in zip(original_size, original_spacing, new_size)]
    return sitk.Resample(smoothed_image, new_size, sitk.Transform(),
                         sitk.sitkLinear, image.GetOrigin(),
                         new_spacing, image.GetDirection(), 0.0,
                         image.GetPixelID())


def demons_registration(fixed_image_pth, moving_image_pth):
    fixed_image = sitk.ReadImage(fixed_image_pth)
    moving_image = sitk.ReadImage(moving_image_pth)
    registration_method = sitk.ImageRegistrationMethod()

    # Create initial identity transformation.
    transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
    # The image returned from the initial_transform_filter is transferred to the transform and cleared out.
    initial_transform = sitk.DisplacementFieldTransform(transform_to_displacment_field_filter.Execute(sitk.AffineTransform(3)))

    # Regularization (update field - viscous, total field - elastic).
    initial_transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=2.0)

    registration_method.SetInitialTransform(initial_transform)

    registration_method.SetMetricAsDemons(10)  # intensities are equal if the difference is less than 10HU

    # Multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8, 4, 0])

    registration_method.SetInterpolator(sitk.sitkLinear)
    # If you have time, run this code as is, otherwise switch to the gradient descent optimizer
    # registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=20,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # If corresponding points in the fixed and moving image are given then we display the similarity metric
    # and the TRE during the registration.

    return registration_method.Execute(fixed_image, moving_image)



def get_affined_moving_image(fixed_image_pth, moving_image_path,ml_path=None):
    affine_path =moving_image_path.replace('moving.nii.gz','affine.nii.gz')
    affine_txt = moving_image_path.replace('moving.nii.gz', 'affine_transform.txt')
    cmd = nifty_reg_affine(ref=fixed_image_pth, flo=moving_image_path, aff=affine_txt, res=affine_path)
    affine_label_path = None

    if ml_path is not None:
        affine_label_path =moving_image_path.replace('moving.nii.gz', 'warped_label.nii.gz')
        cmd += '\n' + nifty_reg_resample(ref=fixed_image_pth,flo=ml_path,trans=affine_txt, res=affine_label_path, inter= 0)
    process = subprocess.Popen(cmd, shell=True)
    process.wait()

    affine_image_cp = sitk.ReadImage(affine_path)
    affine_image_array = sitk.GetArrayFromImage(affine_image_cp)
    affine_image_array[np.isnan(affine_image_array)] = 0.
    affine_image = sitk.GetImageFromArray(affine_image_array)
    affine_image.SetSpacing(affine_image_cp.GetSpacing())
    affine_image.SetOrigin(affine_image_cp.GetOrigin())
    affine_image.SetDirection(affine_image_cp.GetDirection())
    sitk.WriteImage(affine_image, affine_path)


    return affine_path, affine_label_path


def multiscale_demons(registration_algorithm,
                      fixed_image_pth, moving_image_pth, initial_transform=None,
                      shrink_factors=None, smoothing_sigmas=None):
    """
    Run the given registration algorithm in a multiscale fashion. The original scale should not be given as input as the
    original images are implicitly incorporated as the base of the pyramid.
    Args:
        registration_algorithm: Any registration algorithm that has an Execute(fixed_image, moving_image, displacement_field_image)
                                method.
        fixed_image: Resulting transformation maps points from this image's spatial domain to the moving image spatial domain.
        moving_image: Resulting transformation maps points from the fixed_image's spatial domain to this image's spatial domain.
        initial_transform: Any SimpleITK transform, used to initialize the displacement field.
        shrink_factors: Shrink factors relative to the original image's size.
        smoothing_sigmas: Amount of smoothing which is done prior to resmapling the image using the given shrink factor. These
                          are in physical (image spacing) units.
    Returns:
        SimpleITK.DisplacementFieldTransform
    """
    # Create image pyramid.

    fixed_image = sitk.ReadImage(fixed_image_pth)
    moving_image = sitk.ReadImage(moving_image_pth)
    fixed_images = [fixed_image]
    moving_images = [moving_image]
    if shrink_factors:
        for shrink_factor, smoothing_sigma in reversed(list(zip(shrink_factors, smoothing_sigmas))):
            fixed_images.append(smooth_and_resample(fixed_images[0], shrink_factor, smoothing_sigma))
            moving_images.append(smooth_and_resample(moving_images[0], shrink_factor, smoothing_sigma))

    # Create initial displacement field at lowest resolution.
    # Currently, the pixel type is required to be sitkVectorFloat64 because of a constraint imposed by the Demons filters.
    if initial_transform:
        initial_displacement_field = sitk.TransformToDisplacementField(initial_transform,
                                                                       sitk.sitkVectorFloat64,
                                                                       fixed_images[-1].GetSize(),
                                                                       fixed_images[-1].GetOrigin(),
                                                                       fixed_images[-1].GetSpacing(),
                                                                       fixed_images[-1].GetDirection())
    else:
        initial_displacement_field = sitk.Image(fixed_images[-1].GetWidth(),
                                                fixed_images[-1].GetHeight(),
                                                fixed_images[-1].GetDepth(),
                                                sitk.sitkVectorFloat64)
        initial_displacement_field.CopyInformation(fixed_images[-1])

    # Run the registration.
    initial_displacement_field = registration_algorithm.Execute(fixed_images[-1],
                                                                moving_images[-1],
                                                                initial_displacement_field)
    # Start at the top of the pyramid and work our way down.
    for f_image, m_image in reversed(list(zip(fixed_images[0:-1], moving_images[0:-1]))):
        initial_displacement_field = sitk.Resample(initial_displacement_field, f_image)
        initial_displacement_field = registration_algorithm.Execute(f_image, m_image, initial_displacement_field)
    return sitk.DisplacementFieldTransform(initial_displacement_field)


def __read_and_clean_itk_info(path):
    return sitk.GetImageFromArray(sitk.GetArrayFromImage(sitk.ReadImage(path)))

def resize_input_img_and_save_it_as_tmp(img_pth, is_label=False,fname=None,debug_path=None):
        """
        :param img: sitk input, factor is the outputsize/patched_sized
        :return:
        """
        img_org = sitk.ReadImage(img_pth)
        img = __read_and_clean_itk_info(img_pth)
        resampler= sitk.ResampleImageFilter()
        dimension =3
        factor = np.flipud([0.5,0.5,0.5])
        img_sz = img.GetSize()
        affine = sitk.AffineTransform(dimension)
        matrix = np.array(affine.GetMatrix()).reshape((dimension, dimension))
        after_size = [int(img_sz[i]*factor[i]) for i in range(dimension)]
        after_size = [int(sz) for sz in after_size]
        matrix[0, 0] =1./ factor[0]
        matrix[1, 1] =1./ factor[1]
        matrix[2, 2] =1./ factor[2]
        affine.SetMatrix(matrix.ravel())
        resampler.SetSize(after_size)
        resampler.SetTransform(affine)
        if is_label:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkBSpline)
        img_resampled = resampler.Execute(img)
        fpth = os.path.join(debug_path,fname)
        img_resampled.SetSpacing(factor_tuple(img_org.GetSpacing(),1./factor))
        img_resampled.SetOrigin(factor_tuple(img_org.GetOrigin(),factor))
        img_resampled.SetDirection(img_org.GetDirection())
        sitk.WriteImage(img_resampled, fpth)
        return fpth

def sitk_grid_sampling(fixed,moving, displacement,is_label=False):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(displacement)
    out = resampler.Execute(moving)
    return out


debug_path = '/playpen/zyshen/debugs/demons/'

moving_img_path = '/playpen/zhenlinx/Data/OAI_segmentation/Nifti_rescaled/9047800_20050111_SAG_3D_DESS_LEFT_016610322306_image.nii.gz'
target_img_path = '/playpen/zhenlinx/Data/OAI_segmentation/Nifti_rescaled/9003406_20041118_SAG_3D_DESS_LEFT_016610296205_image.nii.gz'
ml_path = '/playpen/zhenlinx/Data/OAI_segmentation/Nifti_rescaled/9047800_20050111_SAG_3D_DESS_LEFT_016610322306_label_all.nii.gz'
tl_path = '/playpen/zhenlinx/Data/OAI_segmentation/Nifti_rescaled/9003406_20041118_SAG_3D_DESS_LEFT_016610296205_label_all.nii.gz'
#ml_path = '/playpen/zhenlinx/Data/OAI_segmentation/segmentations/images_6sets_right/Cascaded_2_AC_residual-1-s1_end2end_multi-out_UNet_bias_Nifti_rescaled_train1_patch_128_128_32_batch_2_sample_0.01-0.02_cross_entropy_lr_0.0005_scheduler_multiStep_02262018_013038/9991313_20050623_SAG_3D_DESS_RIGHT_10564014_prediction_step1_batch6_16_reflect.nii.gz'
#tl_path = '/playpen/zyshen/debugs/9002116_20060804_SAG_3D_DESS_RIGHT_11269909_prediction_step1_batch6_16_reflect.nii.gz'

af_warped_path = '/playpen/zyshen/debugs/af_warped_img.nii.gz'
af_warped_lpath = '/playpen/zyshen/debugs/af_warped_label.nii.gz'
demons_warped_path = '/playpen/zyshen/debugs/demons_warped_img.nii.gz'
demons_warped_lpath = '/playpen/zyshen/debugs/demons_warped_label.nii.gz'

moving_img_path = resize_input_img_and_save_it_as_tmp(moving_img_path,is_label=False,fname='moving.nii.gz',debug_path=debug_path)
target_img_path = resize_input_img_and_save_it_as_tmp(target_img_path,is_label=False,fname='target.nii.gz',debug_path=debug_path)
moving_label_path = resize_input_img_and_save_it_as_tmp(ml_path,is_label=True,fname='l_moving.nii.gz',debug_path=debug_path)
target_label_path = resize_input_img_and_save_it_as_tmp(tl_path,is_label=True,fname='l_target.nii.gz',debug_path=debug_path)

demons_filter =  sitk.FastSymmetricForcesDemonsRegistrationFilter()
demons_filter.SetNumberOfIterations(20)
# Regularization (update field - viscous, total field - elastic).
demons_filter.SetSmoothDisplacementField(True)
demons_filter.SetStandardDeviations(1.)



affine_tf = None


# # Run the registration.
#
# #tx = demons_registration(target_img_path,moving_img_path)
#
#
# fixed = sitk.ReadImage(target_img_path)
# moving = sitk.ReadImage(moving_img_path)
#
# affine_filter = sitk.CenteredTransformInitializerFilter()
# affine_tf = affine_filter.Execute(fixed, moving,sitk.AffineTransform(fixed.GetDimension()))
#
#
#
#
#
# initial_transform = sitk.CenteredTransformInitializer(fixed,
#                                                       moving,
#                                                       sitk.Euler3DTransform(),
#                                                       sitk.CenteredTransformInitializerFilter.GEOMETRY)
# registration_method = sitk.ImageRegistrationMethod()
# registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
# registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
# registration_method.SetMetricSamplingPercentage(0.01)
# registration_method.SetInterpolator(sitk.sitkLinear)
# # The order of parameters for the Euler3DTransform is [angle_x, angle_y, angle_z, t_x, t_y, t_z]. The parameter
# # sampling grid is centered on the initial_transform parameter values, that are all zero for the rotations. Given
# # the number of steps and their length and optimizer scales we have:
# # angle_x = 0
# # angle_y = -pi, 0, pi
# # angle_z = -pi, 0, pi
# registration_method.SetOptimizerAsExhaustive(numberOfSteps=[10,10,10,10,10,10], stepLength = np.pi)
# registration_method.SetOptimizerScales([1,1,1,1,1,1])
#
# #Perform the registration in-place so that the initial_transform is modified.
# registration_method.SetInitialTransform(initial_transform, inPlace=True)
# registration_method.Execute(fixed, moving)
#
# print('best initial transformation is: ' + str(initial_transform.GetParameters()))


moving_img_path, moving_label_path = get_affined_moving_image(target_img_path, moving_img_path, ml_path=moving_label_path)

tx = multiscale_demons(registration_algorithm=demons_filter,
                       fixed_image_pth = target_img_path,
                       moving_image_pth = moving_img_path,
                       shrink_factors =[4,2],
                       smoothing_sigmas = [4,2],
                       initial_transform=affine_tf)
warped_img = sitk_grid_sampling(sitk.ReadImage(target_img_path),sitk.ReadImage(moving_img_path),tx,is_label=False)
warped_label = sitk_grid_sampling(sitk.ReadImage(target_label_path),sitk.ReadImage(moving_label_path),tx,is_label=True)
#phi = sitk.GetArrayFromImage(tx.GetDisplacementField())
sitk.WriteImage(warped_img,demons_warped_path)
sitk.WriteImage(warped_label,demons_warped_lpath)
