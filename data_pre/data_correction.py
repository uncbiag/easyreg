import SimpleITK as sitk
import os
import numpy as np
from glob import glob
from multiprocessing import Pool, TimeoutError
from itertools import product
from functools import partial

# a function to get the center coordinate of a 3D bounding box give its starting position and size
def get_box_center(box):
    box = np.array(box)
    starting_pos = box[0:3]
    size = box[3:]

    return (starting_pos+size//2).astype(int)


# a function to get the up and lower bound of a region given its center index and region size
def get_region_bound(center, size):
    lower = center - (size//2)
    up = lower + size
    return lower, up


# # crop the image given a region, if the region is outside the image boundaries, padding it with zeros.
# def crop(image, start, size):
#     image_size = image.GetSize()
#     crop_size = np.array([size[i] if start[i]+size[i]<= image_size[i] else image_size[i] - start[i] for i in range(3)])
#     print(crop_size)
#     return sitk.RegionOfInterest(image, crop_size, start)


def crop_sample(image, label, roi_size):
    image_size = image.GetSize()

    label_shape_stat = sitk.LabelShapeStatisticsImageFilter()   # filter to analysis the label shape
    label_shape_stat.Execute(label)
    box = np.array(label_shape_stat.GetBoundingBox(1))

    # get the lower bounds and upper bounds coordinates
    low_bound, up_bound = get_region_bound(get_box_center(box), (roi_size * 1.1).astype(int))

    # get the crop size at lower bounds and limit them to be positive
    # then pad the dropped size after cropping
    crop_low = np.array([low_bound[i] if low_bound[i]>0 else 0 for i in range(3)], dtype=int)
    padding_low = np.array([-low_bound[i] if low_bound[i]<0 else 0 for i in range(3)], dtype=int)

    # get the crop size at upper bound and limit them to be smaller than image size,
    # then pad the dropped size after cropping
    up_bound_diff = image_size - up_bound
    crop_up = np.array([up_bound_diff[i] if up_bound_diff[i]>0 else 0 for i in range(3)], dtype=int)
    padding_up = np.array([-up_bound_diff[i] if up_bound_diff[i]<0 else 0 for i in range(3)], dtype=int)

    # crop and pad
    valid_size = image_size - crop_up - crop_low  # size of valid cropped region
    print("valid cropped size: {}".format(valid_size))
    image_crop = sitk.ConstantPad(sitk.Crop(image, crop_low.tolist(), crop_up.tolist()), padding_low.tolist(), padding_up.tolist())
    label_crop = sitk.ConstantPad(sitk.Crop(label, crop_low.tolist(), crop_up.tolist()), padding_low.tolist(), padding_up.tolist())

    return image_crop, label_crop


def image_normalize(image, window_min_perc, window_max_perc, output_min, output_max):
    window_rescale = sitk.IntensityWindowingImageFilter()
    image_array = sitk.GetArrayFromImage(image)
    window_min = np.percentile(image_array, window_min_perc)
    window_max = np.percentile(image_array, window_max_perc)
    return window_rescale.Execute(image, window_min, window_max, output_min, output_max)

def label2image(label_array, source_image):
    label_image = sitk.GetImageFromArray(label_array)
    label_image.CopyInformation(source_image)
    return label_image

def pre_process(image_list, if_corrected=True, if_crop=True, if_normalize=True, overwrite=False):
    """
    Preprocess images given the absolute path to them.
    Pre-processed images are saved with under the same root dir of the data folder with folder name + '_{$ops}'
    :param image_list: A list of absolute path of images to be preprocessed
    :param if_corrected: if do bias field correction
    :param overwrite: if overwrite existing image files
    :return: None
    """
    for image_file in image_list:

        print("Processing {}".format(image_file))
        data_dir = os.path.dirname(image_file)

        # read image and label file
        image = sitk.ReadImage(image_file)
        label_file = os.path.join(data_dir, '_'.join(os.path.basename(image_file).split("_")[:-1]+['label', 'all']) + ".nii.gz")
        label = sitk.ReadImage(label_file)
        image_name = os.path.basename(image_file).split(".")[0]

        # #code for calculate the ROI size
        # label_shape_stat = sitk.LabelShapeStatisticsImageFilter()   # filter to analysis the label shape
        # label_shape_stat.Execute(label)
        # if box == []:
        #     box = np.array(label_shape_stat.GetBoundingBox(1), ndmin=2)
        # else:
        #     box = np.concatenate((box, np.array(label_shape_stat.GetBoundingBox(1),ndmin=2)),axis=0)

        image = sitk.Cast(image, sitk.sitkFloat32)

        # bias field correction
        if if_corrected:
            print("Bias Correcting " + image_name)
            image_corrected_file_path = os.path.join(data_dir, image_name+'_all_corrected.nii.gz')
            if not overwrite and os.path.isfile(image_corrected_file_path):
                print("Bias corrected file found!")
                image_corrected = sitk.ReadImage(image_corrected_file_path)

            else:
                # label_comb = sitk.Threshold(label, lower=1, upper=2, outsideValue=0)  # use the label as correcting mask
                all_mask = label2image(np.ones(sitk.GetArrayFromImage(image).shape).astype(int), image) # if want to use all voxels
                image = sitk.Add(image, 1)
                image = sitk.N4BiasFieldCorrection(image, maskImage=all_mask)
                image = sitk.Subtract(image, 1)
                print("Saving corrected: " + image_name)
                sitk.WriteImage(image, image_corrected_file_path)

        # crop the ROI
        if if_crop:
            print("Cropping" + image_name)
            roi_size = np.array([228, 167, 139])  # this size is pre-calculated
            image, label = crop_sample(image, label, roi_size) # crop the ROI

        # rescale the intensity
        if if_normalize:
            print("Normalizing" + image_name)
            image = image_normalize(image, 0.1, 99.9, 0, 1)

        # save the images
        print("Saving:" + image_name)
        save_dir = data_dir + '{}{}{}'.format("_corrected" if if_corrected else "",
                                              "_cropped" if if_crop else "",
                                              "_rescaled" if if_normalize else "")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        sitk.WriteImage(image, os.path.join(save_dir, os.path.basename(image_file)))
        sitk.WriteImage(label, os.path.join(save_dir, os.path.basename(label_file)))


def main():
    number_of_workers = 2
    if_corrected = True # if do bias field correction
    if_crop = True
    if_normalize = True
    nifti_dir = os.path.realpath("../../data/OAI_segmentation/Nifti")  # repository raw nifti file

    image_files = glob(os.path.join(nifti_dir, "*_image.nii.gz"))  # get image files

    np.random.shuffle(image_files)
    image_file_patitions = np.array_split(image_files, number_of_workers)

    with Pool(processes=number_of_workers) as pool:
        res = pool.map(partial(pre_process, if_corrected=if_corrected, if_crop=if_crop, if_normalize=if_normalize),
                       image_file_patitions)

    # box = []  # ROI regions of each image
    # print(np.max(box,axis=0)[3:])  # get the max bounding box for ROI which gave [228 150 135]X [228 167 139]



if __name__ == '__main__':
    main()
