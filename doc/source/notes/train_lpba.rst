Walkthrough Example for LPBA40
========================================
.. _train_lpba:



We will walk you through for LPBA dataset in this tutorial. For this dataset, we will be using our default setting, which can be found under default_settings/lpba_segmentation.

Firstly, you need to download the preprocessed dataset from https://drive.google.com/drive/folders/1b2oygVEFQ0xQb9-rBwUjO2H154D4YfcR?usp=sharing, after that, you can extract the dataset to DATASET_PATH.

1) Data Preprocessing & Data Organization   
########

As the data is already pre-preprocessed, we only need to generate the training image lists. This can be done by using **prep_data.py** script, it will generate the desired structure for project. 

You need to provide 3 names to give a unique identifier, namely --data_task_name, --output_root_path, --task_name.

* -- output_root_path defines the directory, example could be "outputs_lpba_segmentation"
* -- data_task_name defines the name of the dataset you are using, example could be "lpba_segmentation" 
* -- task_name defines the specific name of the experiment, example could be "segmentation_initial_task"

    
Example script can be run as following, 

.. code-block:: shell

  python prep_data.py --dataset_path DATASET_LOCATION --output_root_path outputs_lpba_segmentation --data_task_name lpba_segmentation


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
For LPBA Segmentation, you can use the setting file under demo_settings/segmentation_lpba/ folder.
In order to start training, you need to execute the following script:

.. code-block:: shell

    python start_segmentation_training.py -ts demo_settings/segmentation_lpba/curr_task_settings.json --output_root_path outputs_lpba_segmentation --data_task_name lpba_segmentation --task_name initial_lpba_segmentation


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

    python train_reg.py -ts scripts/settings_for_lpba/reg_lddmm_train/curr_task_settings.json --output_root_path lpba_reg --data_task_name lpba --task_name reg_with_lddmm
