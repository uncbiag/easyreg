Obtaining results from trained model
========================================

.. _get_results:


After training a model or obtaining a trained model, we can readily get our model to be used for forward pass. For this matter, we provide a script to get the results for a trained model.



Our framework requires files to be organized in a specific structure. However, it is possible to use the **prep_data.py** script in order to generate the appropriate folders and file lists. **seg_eval.py** takes the following arguments:

.. code:: shell

        Arguments:
            -m : the path to model file, 
            -o  : the directory you want to have the uploads at.
            -i: absolute path to image
            -l: absolute path to label
            -ts : setting file for the evaluation
            

This script outputs the 3D segmentation to `{-o}/seg/res/records/3D`, under the name of {image_name}_test_iter_0_output.nii.gz.

An example arguments could be as following:

.. code:: shell

python eval_seg.py -m outputs/25case/train_name/checkpoints/epoch_350_ -o outputs -i s22.nii.gz -l s22.nii.gz -ts settings_for_lpba/seg_test/
