Walkthrough Example for LPBA40
========================================
.. _train_lpba:



We will walk you through for LPBA dataset in this tutorial.
A subset of LPBA40 is included at demo/lpba_examples



Train segmentation network on LPBA
____________________________________

1) Prepare data
###############
Please read *prepare data* part  :ref:`prepare-data-training-label` first before moving ahead.
You can prepare the data accordingly or here we use the "SegDatasetPool" in script "seg_data_pool.py" to help prepare the data.

.. code-block:: shell

        data_path = "../demo/lpba_examples/data"
        label_path = "../demo/lpba_examples/label"
        divided_ratio = (0.4, 0.4, 0.2) # ratio for train val test
        output_path = '../demo/demo_training_seg_net/lpba'
        lpba = SegDatasetPool().create_dataset(dataset_name='lpba',file_type_list=['*nii.gz'])
        lpba.set_data_path(data_path)
        lpba.set_label_path(label_path)
        lpba.set_output_path(output_path)
        lpba.set_divided_ratio(divided_ratio)
        lpba.prepare_data()

**


2) Train segmentation network
#############################################

The task setting file can be found at *demo/demo_settings/seg/lpba_seg_train*.
We can train segmentation on LPBA simply by

.. code:: shell

  python demo_for_seg_train.py -o=./demo_training_seg_net  -dtn=lpba -tn=training_seg  -ts=./demo_settings/seg/lpba_seg_train -g=0



Train registration network on LPBA
____________________________________

1) Prepare data
###############
Please read *prepare data* part  :ref:`prepare-data-training-label` first before moving ahead.
You can prepare the data accordingly or here we use the "RegDatasetPool" in script "reg_data_pool.py" to help prepare the data.

.. code-block:: shell

        data_path = "../demo/lpba_examples/data"
        label_path = "../demo/lpba_examples/label"
        divided_ratio = (0.4, 0.4, 0.2) # ratio for train val test
        file_type_list = ['*.nii.gz']
        dataset_type = 'custom'
        output_path = '../demo/demo_training_reg_net/lpba'
        lpba = RegDatasetPool().create_dataset(dataset_type)
        lpba.file_type_list = file_type_list
        lpba.set_data_path(data_path)
        lpba.set_output_path(output_path)
        lpba.set_divided_ratio(divided_ratio)
        lpba.set_label_path(label_path)
        lpba.prepare_data()

**


2) Train registration network
#############################################

We use the same setting file from our OAI registration demo to train the LPBA registration tasks, the setting can be found  at *demo/demo_settings/mermaid/training_on_3_cases_voxelmorph*.
We need to manually set *img_after_resize* in setting to a desired/current value [196, 164, 196] (numpy convention, not itk convention).
We then can train registration network on LPBA simply by

.. code:: shell

    python demo_for_easyreg_train.py  -o=./demo_training_reg_net -dtn=lpba -tn=training_vm_cvpr -ts=./demo_settings/mermaid/training_on_3_cases_voxelmorph -g=0
