import os
import numpy as np
import time
import SimpleITK as sitk
import subprocess
from .nifty_reg_utils import expand_batch_ch_dim, nifty_reg_affine

def smooth_and_resample(image, shrink_factor, smoothing_sigma):
    """
    :param image: The image we want to resample.
    :param shrink_factor: A number greater than one, such that the new image's size is original_size/shrink_factor.
    :param smoothing_sigma: Sigma for Gaussian smoothing, this is in physical (image spacing) units, not pixels.
    :return: Image which is a result of smoothing the input and then resampling it using the given sigma and shrink factor.
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



def get_initial_transform_nifty(nifty_bin, fixed_image_pth, moving_image_path,fname = None):
    """
    call the niftyreg affine transform, and set it in sitk form

    :param nifty_bin: tniftyreg execuable path
    :param fixed_image_pth: fixed/target image path
    :param moving_image_path: moving/source image path
    :param fname: name of image pair
    :return: sitk transform object
    """
    affine_path = moving_image_path.replace('moving.nii.gz', 'affine.nii.gz')
    affine_txt = moving_image_path.replace('moving.nii.gz', fname + '_af.txt')
    cmd = nifty_reg_affine(nifty_bin=nifty_bin, ref=fixed_image_pth, flo=moving_image_path, aff=affine_txt, res=affine_path)
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    affine_trans = get_affine_transform(affine_txt)
    return affine_trans

def get_initial_transform_sitk( fixed_image_pth, moving_image_pth):
    """
    call the SimpleITK affine transform

    :param fixed_image_pth: fixed/target image path
    :param moving_image_path: moving/source image path
    :return: sitk transform object
    """
    fixed_image = sitk.ReadImage(fixed_image_pth)
    moving_image = sitk.ReadImage(moving_image_pth)
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image,
                                                          sitk.AffineTransform(3),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    #registration_method.SetMetricAsCorrelation()
    #registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1, numberOfIterations=1000,
                                                      convergenceMinimumValue=1e-9, convergenceWindowSize=50)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)




    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                     moving_image.GetPixelID())
    affine_path = moving_image_pth.replace('moving.nii.gz', 'affine.nii.gz')
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    sitk.WriteImage(moving_resampled,affine_path)


    return final_transform







def get_initial_transform(fixed_image_pth, moving_image_pth, fname=None,nifty_bin_pth=None):
    """

    :param fixed_image_pth:
    :param moving_image_pth:
    :param fname:
    :param nifty_bin_pth:
    :return:
    """
    if os.path.exists(nifty_bin_pth):
        print("the niftyreg is detected, by default, the niftyreg will be called to get initial affine transformation")
        transform = get_initial_transform_nifty(nifty_bin_pth, fixed_image_pth,moving_image_pth,fname)
    else:
        print("the niftyreg is not detected, the simpleitk will be called to get initial affine transformation")
        print("we don't recommend using simpleitk initialization, since niftyreg provide easier and more stable affine result")
        transform = get_initial_transform_sitk( fixed_image_pth,moving_image_pth)
    return transform





def get_affine_transform(af_pth):
    """
    read the niftyreg affine txt to initialize the sitk object

    :param af_pth: path of affine txt
    :return: affine object
    """
    matrix, trans = read_nifty_reg_affine(af_pth)
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(matrix.ravel())
    affine.SetTranslation(trans)
    return affine


def read_nifty_reg_affine(affine_txt):
    """
    read the nifti affine results(RAS) form to sitk form(LPS)

    :param affine_txt:
    :return: affine matrix (3x3), translation (3x1)
    """
    res = np.loadtxt(affine_txt, delimiter=' ')
    matrix = res[:3,:3]
    matrix_cp = matrix.copy()
    matrix[0,2]=-matrix[0,2]
    matrix[1,2]=-matrix[1,2]
    matrix[2,0]=-matrix[2,0]
    matrix[2,1]=-matrix[2,1]
    matrix[0,0]= matrix_cp[0,0]
    matrix[1,1]= matrix_cp[1,1]
    trans = res[:3,3]
    trans_cp = trans.copy()
    trans[1] =-trans_cp[1]
    trans[0] =-trans_cp[0]
    return matrix, trans




def multiscale_demons(registration_algorithm,
                      fixed_image_pth, moving_image_pth, initial_transform=None,
                      shrink_factors=None, smoothing_sigmas=None,record_path=None,fname=None):
    """
        Run the given registration algorithm in a multiscale fashion. The original scale should not be given as input as the
    original images are implicitly incorporated as the base of the pyramid.

    :param registration_algorithm:  Any registration algorithm that has an Execute(fixed_image, moving_image, displacement_field_image) method.
    :param fixed_image_pth: Resulting transformation maps points from this image's spatial domain to the moving image spatial domain.
    :param moving_image_pth:  Resulting transformation maps points from the fixed_image's spatial domain to this image's spatial domain.
    :param initial_transform:  Any SimpleITK transform, used to initialize the displacement field.
    :param shrink_factors:  Shrink factors relative to the original image's size.
    :param smoothing_sigmas: Amount of smoothing which is done prior to resmapling the image using the given shrink factor. These are in physical (image spacing) units.
    :param record_path: Saving path
    :param fname: pair name
    :return: SimpleITK.DisplacementFieldTransform, jacobi of the transform
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
    if record_path is not None:
        sitk.WriteImage(initial_displacement_field, os.path.join(record_path, fname + '_disp.nii.gz'))
    jacobi_filter = sitk.DisplacementFieldJacobianDeterminantFilter()
    jacobi_image = jacobi_filter.Execute(initial_displacement_field)

    return sitk.DisplacementFieldTransform(initial_displacement_field),jacobi_image



def sitk_grid_sampling(fixed, moving, transform,is_label=False):
    """
    resample the fixed image though transformation map

    :param fixed: fixed or the target image
    :param moving: the moving or the source image
    :param transform: the transformation map
    :param is_label: nearestneighbor interpolation if is label else linear interpolation
    :return: warped image
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)
    out = resampler.Execute(moving)
    return out


def performDemonsRegistration(param, mv_path, target_path, registration_type='demons', record_path = None, ml_path=None,tl_path= None,fname=None):
    """
    call a symmetric forces demons  algorithm, which is provided by simple itk

    :param param: ParameterDict, demons related params
    :param mv_path: path of moving image
    :param target_path: path of target image
    :param registration_type: type of registration, support 'demons' for now
    :param record_path: path of saving results
    :param ml_path: path of label of moving image
    :param tl_path: path of label fo target image
    :param fname: pair name or saving name of the image pair
    :return: warped image, warped label, transformation map (None), jacobian map
    """

    start = time.time()

    print("start demons registration")
    assert registration_type =='demons'
    iter_num = param[('iter', 500,'num of the iteration') ]
    stand_dev = param['std',1.5 , 'the standard deviation in demon registration']
    nifty_bin = param['nifty_bin','','the path of the nifth reg binary file']
    shrink_factors = param['shrink_factors',[],'the multi-scale shrink factor in demons']
    shrink_sigma = param['shrink_sigma',[],'amount of smoothing which is done prior to resmapling the image using the given shrink factor']
    shrink_factors = shrink_factors if len(shrink_factors) else None
    shrink_sigma = shrink_sigma if len(shrink_sigma) else None
    output = None
    loutput = None

    #mv_path, ml_path = get_affined_moving_image(target_path, mv_path, ml_path=ml_path,fname=fname)
    initial_transform = get_initial_transform(target_path, mv_path, fname=fname,nifty_bin_pth=nifty_bin)

    demons_filter = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    demons_filter.SetNumberOfIterations(iter_num)
    # Regularization (update field - viscous, total field - elastic).
    demons_filter.SetSmoothDisplacementField(True)
    demons_filter.SetStandardDeviations(stand_dev)  #1,4

    # Run the registration.
    # param_in_demons = (2,1) # todo write into json settings
    # print("!!!!!!!!!!demons param{}".format(param_in_demons) )
    tx,jacobi_image = multiscale_demons(registration_algorithm=demons_filter,
                           fixed_image_pth=target_path,
                           moving_image_pth=mv_path,
                           shrink_factors=shrink_factors,#[4, 2],
                           smoothing_sigmas=shrink_sigma,
                                        initial_transform=initial_transform,
                                record_path=record_path,fname =fname) #[2,1],[4, 2]) (8,4)
    warped_img = sitk_grid_sampling(sitk.ReadImage(target_path), sitk.ReadImage(mv_path), tx,
                                    is_label=False)
    output = sitk.GetArrayFromImage(warped_img)
    if ml_path is not None and tl_path is not None:
        warped_label = sitk_grid_sampling(sitk.ReadImage(tl_path), sitk.ReadImage(ml_path), tx,
                                          is_label=True)
        loutput = sitk.GetArrayFromImage(warped_label)

    print('demons registration finished and takes: :', time.time() - start)


    jacobian_np = sitk.GetArrayFromImage(jacobi_image)
    # phi = sitk.GetArrayFromImage(tx.GetDisplacementField())
    # """ todo  check that whether the xyz order is the same as the interpolation assumed"""
    # phi = np.transpose(phi,(3,0,1,2))


    return expand_batch_ch_dim(output), expand_batch_ch_dim(loutput), None ,jacobian_np#np.expand_dims(phi,0)