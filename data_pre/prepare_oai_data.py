import SimpleITK as sitk
import scipy.io
import os
import numpy as np
from glob import glob



def image_normalize(image, window_min_perc, window_max_perc, output_min, output_max):
    window_rescale = sitk.IntensityWindowingImageFilter()
    image_array = sitk.GetArrayFromImage(image)
    window_min = np.percentile(image_array, window_min_perc)
    window_max = np.percentile(image_array, window_max_perc)
    return window_rescale.Execute(image, window_min, window_max, output_min, output_max)

def main():

    nifti_dir = os.path.realpath("/playpen/zhenlinx/Data/OAI_segmentation/Nifti")  # repository raw nifti file
    normalized_dir = os.path.realpath("/playpen/zyshen/oai_data/Nifti_rescaled")  # repository to store cropped and rescaled
    if not os.path.exists(normalized_dir):
        os.mkdir(normalized_dir)

    image_files = glob(os.path.join(nifti_dir, "*_image.nii.gz"))  # get image files

    # box = []  # ROI regions of each image

    for image_file in image_files:

        print("Processing {}".format(image_file))

        # read image and label file
        image = sitk.ReadImage(image_file)
        label_file = os.path.join(nifti_dir, '_'.join(os.path.basename(image_file).split("_")[:-1]+['label', 'all']) + ".nii.gz")
        label = sitk.ReadImage(label_file)

        image = sitk.Cast(image, sitk.sitkFloat32)
        image_rescaled = image_normalize(image, 0.1, 99.9, 0., 1.)

        # save the images
        print("Saving")
        sitk.WriteImage(image_rescaled, os.path.join(normalized_dir, os.path.basename(image_file)))
        sitk.WriteImage(label, os.path.join(normalized_dir, os.path.basename(label_file)))

        # print(np.max(box,axis=0)[3:])  # get the max bounding box for ROI which gave [228 150 135]


if __name__ == '__main__':
    main()