Evaluate a trained model
========================================

.. _get_results:

Segmentation Model
#########

After training a model or obtaining a trained model, we can readily get our model to be used for forward pass. For this matter, we provide a script to get the results for a trained model.
We provide an example brain segmentation model for LPBA40 dataset in the following link:
https://drive.google.com/file/d/1QYj-5T2mxSKEXX_V3lzZaE5v4hA8TCtt/view?usp=sharing


Our split can be found here (absolute paths needs to be fixed):
https://drive.google.com/drive/folders/1b2oygVEFQ0xQb9-rBwUjO2H154D4YfcR?usp=sharing


.. code:: shell

        Arguments:
            -m : the path to model file, 
            -o  : the directory you want to have the results at.
            -i: absolute path to image
            -l: absolute path to label
            -ts : setting file for the evaluation
            

This script outputs the 3D segmentation to `{-o}/seg/res/records/3D`, under the name of {image_name}_test_iter_0_output.nii.gz.

An example arguments could be as following:

.. code:: shell

    python eval_seg.py -m MODEL_PATH -o outputs -i s22.nii.gz -l s22.nii.gz -ts settings_for_lpba/seg_test/

    OR 
    python eval_seg.py -m MODEL_PATH -o outputs -txt outputs/25case/test/file_path_list.txt -ts settings_for_lpba/seg_test/



Registration Model
#########

Registration models can be forward pass only by using the following commands:

.. code:: shell

        Arguments:
            -m : the path to model file, 
            -o : the directory you want to have the results at.
            -s: absolute path to source image
            -t: absolute path to target image
            -ls: absolute path to source labels
            -lt: absolute path to target labels
            -ts : setting file for the evaluation



.. code:: shell

    python eval_seg.py -o outputs3 -ts settings_for_lpba/eval_network_lddmm -s /playpen-raid/zyshen/data/lpba_seg_resize/resized_img/s12.nii.gz -t /playpen-raid/zyshen/data/lpba_seg_resize/resized_img/s1.nii.gz -ls /playpen-raid/zyshen/data/lpba_seg_resize/label_filtered/s12.nii.gz -lt /playpen-raid/zyshen/data/lpba_seg_resize/label_filtered/s11.nii.gz  -m outputs/pairwise/lddmm_test3/checkpoints/epoch_590_



{output_directory_path}/reg/res/records/3D contains registered image, jacobi map in Nifti format, as well as the warped labels.