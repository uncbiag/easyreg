Training a Segmentation model
========================================

.. _train_segmentation_model:



We will walk you through for training segmentation models.
All of the scripts that we refer can be found under `scripts/` folder. You can refer to *seg_train.py* script, however, it is totally optional.

1) Data Preprocessing & Data Organization   
########

**The sizes of images should be the factor of 16, if not, it can be set by settings, `img_after_resize`.**

The framework takes 3 names to give a unique identifier, namely --data_task_name, --output_root_path, --task_name.


* -- output_root_path defines the directory, example could be "outputs_lpba_segmentation"
* -- data_task_name defines the name of the dataset you are using, example could be "LPBA" 
* -- task_name defines the specific name of the experiment, example could be "segmentation_initial_task"

Our framework requires files to be organized in a specific structure. However, it is possible to use the **prepare_data.py** script in order to generate the appropriate folders and file lists. **prepare_data.py** takes the following arguments:

.. code:: shell

        Arguments:
        --dataset_path : the path of output folder
        --data_task_name : data task name i.e. lung_reg_task , oai_reg_task
        --output_root_path  : task name i.e. run_training_rdmm_task
        --preprocess : path of the folder where settings are saved,should include cur_task_setting.json
        --task_type: either reg or seg, should be seg 
        --seed: seed that you would like to use, 
        --train_size: percentage size for train set, if it is not pre-splitted
        --test_size: percentage size for test set, if it is not pre-splitted
        --val_size: percentage size for val set, if it is not pre-splitted


This script requires to have data splitted under dataset_path/train, dataset_path/test, dataset_path/val and in each subfolder, we need to have two folders, images and labels.
In specific, below is the desired structure, if the user wants to use their split:
::
     dataset_path
     ├── train          
     │   ├── images
     |   |    └── image1.nii.gz
     │   └── labels
     |        └── label1.nii.gz
     ├── test          
     │   ├── images
     |   |     └── image2.nii.gz
     │   └── labels
     |         └── label2.nii.gz
     ├── val          
     │   ├── images
     |   |     └── image3.nii.gz
     │   └── labels
     |         └── label3.nii.gz



Alternatively, you can also decide to leave splitting to our end, in which case you need to have the following structure:
::
     dataset_path
     ├── images          
     │   ├── image1.nii.gz
     │   └── image2.nii.gz
     ├── labels          
     │   ├── label1.nii.gz
     │   └── label2.nii.gz
  

**IMPORTANT: In order to match the labels with correct images, when their names are sorted, they should be in the same order, if they have same names, it will make them ordered exactly same.** 



    
Example script can be run as following, 

.. code-block:: shell

  python prepare_data.py --dataset_path DATASET_LOCATION --output_root_path segmentation_work --data_task_name lpba_segmentation --task_type seg


2) Segmentation Training Script and Settings
########

Below are the command line arguments that *seg_train.py* accepts. 

.. code:: shell

        Assume there is three-level folder, output_root_path/ data_task_folder/ task_folder
        Arguments:
            --output_root_path/ : the path of output folder
            --data_task_name/ : data task name i.e. lung_reg_task , oai_reg_task
            --task_name / : task name i.e. run_training_rdmm_task
            --setting_folder_path/ : path of the folder where settings are saved,should include cur_task_setting.json
            --gpu_id/ -g: on which gpu to run

**

For training, we can also utilize data augmentation for better segmentation score, which is enabled by default.
Also, this segmentation network is derivate of UNet, which uses residual connections and patch based training. Thus, it is important to set the `patch_size` parameter in the training settings.
If you would like to classify specific labels, you can determine that as well, using `interested_label_list` in the JSON file for settings.

It is possible to replicate our training process using our setting, which can be found under `scripts/settings_for_lpba/seg_train/curr_task_settings.json`.

In order to start training, you need to execute the following script:

.. code-block:: shell

    python train_seg.py -ts settings_for_lpba/seg_train/curr_task_settings.json --output_root_path lpba_segmentation --data_task_name lpba --task_name segmentation_with_unet


Resume the training
^^^^^^^^^^^^^^^^^^^^^^^

If the training needs to be resumed for further fine-tuning, the procedure below can be followed:

To do this, we need to change a few parameters in our settings JSON, which can be found under `--setting_folder_path`

* set "continue_train": true  and set "continue_train_lr"
* optional, if the epoch number needs to be reset into a given number, set "reset_train_epoch" and "load_model_but_train_from_epoch"
* set "model_path" as the path of the checkpoint

..  code:: shell

    python train_seg.py -ts settings_for_lpba/seg_train/curr_task_settings.json --output_root_path lpba_segmentation --data_task_name lpba --task_name segmentation_with_unet_resumed


Tracking the training
^^^^^^^^^^^^^^^^^^^^^^^

We can observe the training under output_root_path/data_task_name/task_name, which can be import to Tensorboard, as it saves in the .tfevents format.

