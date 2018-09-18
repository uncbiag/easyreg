import os
import numpy as np
import time
import SimpleITK as sitk

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
        # img_resampled.SetSpacing(factor_tuple(img_org.GetSpacing(),1./factor))
        # img_resampled.SetOrigin(factor_tuple(img_org.GetOrigin(),factor))
        # img_resampled.SetDirection(img_org.GetDirection())
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

moving_img_path = '/playpen/zyshen/debugs/9002116_20050715_SAG_3D_DESS_RIGHT_10423916_image.nii.gz'
target_img_path = '/playpen/zyshen/debugs/9002116_20060804_SAG_3D_DESS_RIGHT_11269909_image.nii.gz'
ml_path = '/playpen/zyshen/debugs/9002116_20050715_SAG_3D_DESS_RIGHT_10423916_prediction_step1_batch6_16_reflect.nii.gz'
tl_path = '/playpen/zyshen/debugs/9002116_20060804_SAG_3D_DESS_RIGHT_11269909_prediction_step1_batch6_16_reflect.nii.gz'

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


# Run the registration.
tx = multiscale_demons(registration_algorithm=demons_filter,
                       fixed_image_pth = target_img_path,
                       moving_image_pth = moving_img_path,
                       shrink_factors = [4,2],
                       smoothing_sigmas = [4,2])
warped_img = sitk_grid_sampling(sitk.ReadImage(target_img_path),sitk.ReadImage(moving_img_path),tx,is_label=False)
warped_label = sitk_grid_sampling(sitk.ReadImage(target_label_path),sitk.ReadImage(moving_label_path),tx,is_label=True)
phi = sitk.GetArrayFromImage(tx.GetDisplacementField())
sitk.WriteImage(warped_img,demons_warped_path)
sitk.WriteImage(warped_label,demons_warped_lpath)
