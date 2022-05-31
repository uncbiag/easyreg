Walkthrough Example for lung dataset
========================================
.. _train_lung:

We will walk you through for COPD lung dataset in this tutorial.


Train registration network on Lung dataset
____________________________________

1) Prepare data
###############
The data preparation includes two step:
1) preprocess the data (intensity crop, image resampling and padding)
2) organize the data into easyreg input format

Now let's get into details

1) Here use a preprocessing function
data_pre/reg_preprocess_example/preprocessing_one_lung_data.py
It takes an image path and its segmentation (lung region) path as inputs and save proccessed outputs into OUTPUT_PREPROCESSED_FOLDER.
Please apply this function to each of the image and its segmentation, saving into the same OUTPUT_PREPROCESSED_FOLDER.


2) Please read *prepare data* part  :ref:`prepare-data-training-label` first before moving ahead.
You can prepare the data accordingly or here we provide an example, using the "RegDatasetPool" in script "reg_data_pool.py" to help prepare the data (please set your own YOUR_TASK_OUTPUT_PATH).

.. code-block:: shell

        data_path = OUTPUT_PREPROCESSED_FOLDER
        source_image_path_list = glob(os.path.join(data_path, "*EXP*img*"))
        target_image_path_list = [path.replace("_EXP_", "_INSP_") for path in source_image_path_list]
        coupled_pair_path_list = list(zip(source_image_path_list,target_image_path_list))
        divided_ratio = (0.985, 0.15, 0.0) # ratio for train val test (no need for this task, we have a separate test dataset with 10 COPD cases)
        dataset_type = 'custom'
        output_path = YOUR_TASK_OUTPUT_PATH + '/lung_reg'
        label_switch = ('_img', '_seg')
        lung = RegDatasetPool().create_dataset(dataset_type)
        lung.reg_coupled_pair=True
        lung.coupled_pair_list = coupled_pair_path_list
        lung.label_switch = label_switch
        lung.set_data_path(data_path)
        lung.set_output_path(output_path)
        lung.set_divided_ratio(divided_ratio)
        lung.prepare_data()

**

2) Manually add unpublic network and landmarks into easyreg
########################################################

Since the network we use here is not a public one, so we need to manually add the script into easyreg framework
To do that, we
1) Copy the model script "multiscale_net_improved.py" from  https://drive.google.com/file/d/18b7l1Z75vWBn5YLZi2H27HwUGQppiqAM/view?usp=sharing
and move it into easyreg/easyreg

2) uncomment "multiscale_net" in "model_pool" dictionary structure (easyreg/reg_net.py)

Besides, the dirlab landmarks (ground truth) is also not a public one, so we need to manually add it
download landmarks.zip from https://drive.google.com/file/d/1jcn9Xjd3eRYJitRQkHHnXOWe-_dhSurV/view?usp=sharing
and unzip it into  demo/lung_reg/landmarks


3) Train registration network
#############################################

The relevant training setting can be found at *demo/demo_settings/mermaid/train_network_multiscale_lung.
We then train registration network on lung dataset by

.. code:: shell

    python demo_for_easyreg_train.py -o YOUR_TASK_OUTPUT_PATH -dtn=lung_reg -tn=train_multiscale_debug -ts=./demo/demo_settings/mermaid/train_network_multiscale_lung -g=3


3) Evaluate registration network
####################################

We have ten separate DirLab cases (each with 300 landmarks) for evaluate the performance, simply run hack.py to evaluate the model performance.
