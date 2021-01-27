Obtaining results from trained model
========================================

.. _get_results:


After training a model or obtaining a trained model, we can readily get our model to be used for forward pass. For this matter, we provide a script to get the results for a trained model.



Our framework requires files to be organized in a specific structure. However, it is possible to use the **prep_data.py** script in order to generate the appropriate folders and file lists. **seg_eval.py** takes the following arguments:

.. code:: shell

        Arguments:
            -m : the path to model file, 
            -o  : the directory you want to have the uploads at.
            --preprocess : if you would like to apply pre-processing (under development)
            --use_labels : if you have labels for given pairs
            --txt : the list of files in a txt document, which can be generated via prep_data.py
            --image_list: absolute path to images
            --label_list: absolute path to labels
            -ts : setting file for the evaluation
            either --txt or image/label list needs to be provided 

This script outputs the 3D segmentation to `-o/seg/res/records/3D`, under the name of {image_name}_test_iter_0_output.nii.gz.

An example arguments could be as following:

.. code:: shell

    python seg_eval.py -m /playpen-raid/olut/easyreg/demo/outputs/25case/train_name/checkpoints/epoch_350_ -o outputs -txt outputs/25case/test/file_path_list.txt -ts demo_settings/seg/lpba_seg_test

