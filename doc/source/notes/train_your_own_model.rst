Train your own model
========================================

In this tutorial, we would show how to train your own model.


The data part refers to prepare_data section.


The script *demo_for_easyreg_train.py* is for training new learning-based registration model.

::

        A training interface for learning methods.
        The method support list :  mermaid-related methods
        Assume there is three-level folder, output_root_path/ data_task_folder/ task_folder
        Arguments:
            --output_root_path/ -o: the path of output folder
            --data_task_name/ -dtn: data task name i.e. lung_reg_task , oai_reg_task
            --task_name / -tn: task name i.e. run_training_vsvf_task, run_training_rdmm_task
            --train_affine_first: train affine network first, then train non-parametric network
            --gpu_id/ -g: on which gpu to run

An example

..  code::

    python demo_for_easyreg_train.py -o=/playpen/zyshen/data -data_task_name=croped_for_reg_debug_3000_pair_oai_reg_inter -task_name=interface_rdmm -ts=/playpen/zyshen/reg_clean/demo/demo_settings/mermaid/training_network_rdmm -g=0



In the 'task_name' folder, three folder will be auto created, **log** for tensorboard, **checkpoints** for saving models,
**records** for saving running time results. Besides, two files will also be created. **task_settings.json** for recording settings of current tasks.
**logfile.log** for terminal output ( only flushed when task finished).


One thing to mention is that the affine-network involves the fully connection layer,  whose input channel number needs to be adjusted by the input image size.