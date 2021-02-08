Training a Registration model
========================================

.. _train_reg_model:



We will walk you through for training registration models.
All of the scripts that we refer can be found under `/demo` folder. You can refer to *seg_train.py* script, however, it is totally optional.

1) Data Preprocessing & Data Organization   
########


The framework takes 3 names to give a unique identifier, namely --data_task_name, --output_root_path, --task_name.


* -- output_root_path defines the directory, example could be "outputs_lpba_reg"
* -- data_task_name defines the name of the dataset you are using, example could be "LPBA" 
* -- task_name defines the specific name of the experiment, example could be "reg_initial_task"

Our framework requires files to be organized in a specific structure. However, it is possible to use the **prep_data.py** script in order to generate the appropriate folders and file lists. **prep_data.py** takes the following arguments:

.. code:: shell

        Arguments:
        --dataset_path : the path of output folder
        --data_task_name : data task name i.e. lung_reg_task , oai_reg_task
        --output_root_path  : task name i.e. run_training_rdmm_task
        --preprocess : path of the folder where settings are saved,should include cur_task_setting.json
        --seed: seed that you would like to use
        --im2im: enables the flag for image-to-image (pairwise) registration
        --atlas: enables the flag for image-to-atlas registration
        --atlas_name: if you want to use one image as an atlas, example could be s1.nii.gz, which sets s1.nii.gz as atlas.
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

  python prep_data.py --dataset_path DATASET_LOCATION --output_root_path reg_work --data_task_name lpba_reg


2) Registration Training Script and Settings
########

Below are the command line arguments that *reg_train.py* accepts. 

.. code:: shell

        Assume there is three-level folder, output_root_path/ data_task_folder/ task_folder
        Arguments:
            --output_root_path/ : the path of output folder
            --data_task_name/ : data task name i.e. lung_reg_task , oai_reg_task
            --task_name / : task name i.e. run_training_rdmm_task
            --setting_folder_path/ : path of the folder where settings are saved,should include cur_task_setting.json
            --gpu_id/ -g: on which gpu to run

**

Also, this registration network (default setting) is derivate of VoxelMorph [ref], where we predict the down-scaled displacement field using U-Net. By the construction, it does not guarantee folding-free solution, however there is another models included in the framework with folding-free guarantees. One of which is the derivate of the VoxelMorph method [ref], that uses VAE-like model and step-by-step refinement for the displacement map that replicates the integration scheme. We also further provide LDDMM and momentum based models, the example settings could be found under `settings_for_lpba/reg_train`. Currently, we have limited support for LDDMM models but we will support it too.
It is really important to babysit the training if a new dataset is used, and the records can be found under `output_root_path/data_task_name/task_name/records`, we recommend to try different loss measures, such as Localized Cross Correlation, with different factors for regularization. The coefficient for similarity loss is set to 1, so you can tune the registration loss coefficient and the learning rate to tune the training.
Further, if labels for the dataset is provided, we measure the performance in terms of Dice and Jacobi distances with respect to registered labels.
It is possible to replicate our training process using our setting, which can be found under `scripts/settings_for_lpba/reg_train/curr_task_settings.json`.

In order to start training, you need to execute the following script:

.. code-block:: shell

    python train_reg.py -ts settings_for_lpba/reg_train/curr_task_settings.json --output_root_path lpba_reg --data_task_name lpba --task_name reg_with_unet


Pre-alignment with affine network
^^^^^^^^^^^^^^^^^^^^^^^
You can pre-align images using affine transformations, which can be enabled from settings. The affine transformations are predicted by a small neural network. It is handy and recommended for atlas-based registration, especially when an atlas from another dataset is utilized.


Resume the training
^^^^^^^^^^^^^^^^^^^^^^^

If the training needs to be resumed for further fine-tuning, the procedure below can be followed:

To do this, we need to change a few parameters in our settings JSON, which can be found under `--setting_folder_path`

* set "continue_train": true  and set "continue_train_lr"
* optional, if the epoch number needs to be reset into a given number, set "reset_train_epoch" and "load_model_but_train_from_epoch"
* set "model_path" as the path of the checkpoint

..  code:: shell

    python train_reg.py -ts settings_for_lpba/reg_train/curr_task_settings.json --output_root_path lpba_reg --data_task_name lpba --task_name reg_with_unet_resumed


Tracking the training
^^^^^^^^^^^^^^^^^^^^^^^

We can observe the training under output_root_path/data_task_name/task_name, which can be import to Tensorboard, as it saves in the .tfevents format. Also, it is recommended to check `output_root_path/data_task_name/task_name/records` folder to see intermediate result for specific images.
