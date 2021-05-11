Walkthrough Example for OASIS
========================================
.. _train_oasis:



We will walk you through for OASIS dataset in this tutorial. For this dataset, we will be using our default setting for OASIS dataset, which can be found under `scripts/`.
We will be using a preprocessed version of OASIS dataset, in which where we register the dataset to SRI24<link> atlas, and apply a brain extraction. Both the atlas and the processed data can be accessed using the following link.
After downloading the dataset, it should have two subfolders, namely `images` and `labels`. The folder contains the both is going to be the DATASET_PATH for our training scripts. You can access our own model using this link.




1) Data Preprocessing & Data Organization   
########

As the data is already pre-preprocessed, we only need to generate the training image lists. This can be done by using **prep_data.py** script, it will generate the desired structure for project. 

You need to provide 3 names to give a unique identifier, namely --data_task_name, --output_root_path, --task_name.

* -- output_root_path defines the directory, example could be "outputs_oasis"
* -- data_task_name defines the name of the dataset you are using, example could be "oasis_segmentation" 
* -- task_name defines the specific name of the experiment, example could be "segmentation_first_training"

    
Example script can be run as following, 

.. code-block:: shell

  python prep_data.py --dataset_path DATASET_PATH --output_root_path outputs_oasis --data_task_name oasis_segmentation


2) Segmentation Training Script and Settings
########

Below are the command line arguments that *train_seg.py* accepts. 

.. code:: shell

        Assume there is three-level folder, output_root_path/ data_task_folder/ task_folder
        Arguments:
            --output_root_path/ : the path of output folder
            --data_task_name/ : data task name i.e. lung_reg_task , oai_reg_task
            --task_name / : task name i.e. run_training_rdmm_task
            --setting_folder_path/ : path of the folder where settings are saved,should include cur_task_setting.json
            --gpu_id/ -g: on which gpu to run

**

It is possible to replicate our training process using our setting, which can be found under demo_settings/segmentation_lpba/curr_task_settings.json. The detailed explanation for settings will be provided.
For OASIS Segmentation, you can use the setting file under scripts/settings_for_oasis_seg/ folder.
In order to start training, you need to execute the following script:

.. code-block:: shell

    python scripts/train_seg.py -ts scripts/settings_for_oasis_seg/cur_task_settings.json --output_root_path outputs_oasis --data_task_name oasis_segmentation --task_name segmentation_first_training


3) Registration Training Script and Settings
########

Below are the command line arguments that *train_reg.py* accepts. 

.. code:: shell

        Assume there is three-level folder, output_root_path/ data_task_folder/ task_folder
        Arguments:
            --output_root_path/ : the path of output folder
            --data_task_name/ : data task name i.e. lung_reg_task , oai_reg_task
            --task_name / : task name i.e. run_training_rdmm_task
            --setting_folder_path/ : path of the folder where settings are saved,should include cur_task_setting.json
            --gpu_id/ -g: on which gpu to run

**

Same as segmentation, it is possible to replicate our training settings with the following script:

.. code-block:: shell
    python scripts/train_reg.py --output_root_path outputs_oasis --data_task_name oasis_registration --task_name reg_with_lddmm -ts scripts/settings_for_oasis_reg