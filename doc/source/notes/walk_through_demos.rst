Walk through Demos
========================================

Introduction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

EasyReg is developed for research propose, providing interfaces for various registration methods, including `AntsPy <https://github.com/ANTsX/ANTsPy>`_ , `NiftyReg <http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg>`_ and Demons (embedded in `SimpleITK <http://www.simpleitk.org/SimpleITK/resources/software.html>`_).
*NiftyReg requires users to install themselves. Once installed, please set the **nifty_bin** in 'cur_task_setting.json' to the path of the NiftyReg binary file.*

In this tutorial, we would show how to run the following demos:

1. Demos on toolkit methods (NiftyReg, ANTsPy, Demons)
2. Demos on optimization-based mermaid model (vSVF and RDMM)
3. Demos on evaluating pretrained learning-based mermaid model (vSVF and RDMM)
4. Demos on evaluating learning-based mermaid model(vSVF and RDMM)
5. Demos on training learning-based mermaid network
6. Demos on training VoxelMorph (cvpr and miccai version)
7. Demos on training `Brainstorm <https://arxiv.org/abs/1902.09383>`_
8. Demos on data augmentation (optimization and learnt version)

*Demos for RDMM on synthesis data is put in mermaid repository. if you are interested in that, please refer to mermaid tutorial.*


Download Examples and Pretrained Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To run the demo, we first need to download examples and pretrained model for learning based methods.

.. code:: shell

    gdown https://drive.google.com/uc?id=1m0dhCUYe3DTisHEV3FMy7LY0SIJlTjpa
    unzip demo.zip -d EASYREG_REPOSITORY_PATH

Now we are ready to play with demos. The repository *demo* lists the scripts and settings for demos.



The script *demo_for_easyreg_eval.py*  is for optimization-based or pretrained methods.

Let's first go through the document of the *demo_for_easyreg_eval.py*.
.. code:: shell

    An evaluation interface for optimization methods or learning methods with pre-trained models.
    Though the purpose of this script is to provide demo, it is a generalized interface for evaluating the following methods.
    The method support list :  mermaid-related ( optimizing/pretrained) methods, ants, demons, niftyreg
    The demo names supported by category are :
        mermaid: eval_network_rdmm/eval_network_vsvf/opt_vsvf/opt_rdmm/opt_rdmm_predefined
        ants: ants
        demons: demons
        niftyreg: niftyreg
    * eval_network_* refers to learning methods with pre-trained models
    * opt_* : refers to optimization based methods
    Arguments:
        demo related:
             --run_demo: run the demo
             --demo_name: eval_network_rdmm/eval_network_vsvf/opt_vsvf/opt_rdmm/opt_rdmm_predefined/ants/demons/niftyreg
        input related:two input styles are supported. For both cases, images should be first normalized into [0,1].
            1. given txt
             --pair_txt_path/-txt: the txt file list the pairs to be registered
            2. given image
            --source_list/ -s: the source path list,  s1 s2 s3..sn
            --target_list/ -t: the target path list,  t1 t2 t3..tn
            --lsource_list/ -ls: optional, the source label path list,  ls1 ls2 ls3..lsn
            --ltarget_list/ -lt: optional, the target label path list,  lt1 lt2 lt3..ltn
        other arguments:
             --setting_folder_path/ -ts :path of the folder where settings are saved,should include cur_task_setting.json, mermaid_affine_settings.json(optional) and mermaid_nonp_settings(optional)
             --task_output_path/ -o: the path of output folder
             --gpu_id/ -g: on which gpu to run



Demos on toolkit methods
^^^^^^^^^^^^^^^^^^^^^^^^
1. NiftyReg

.. code:: shell

    python demo_for_easyreg_eval.py  --run_demo --demo_name=nifty_reg  -txt=./oai_examples.txt -o=OUTPUT_PATH

2. Demons

.. code:: shell

    python demo_for_easyreg_eval.py  --run_demo --demo_name=demons -txt=./oai_examples.txt -o=OUTPUT_PATH

3. AntsPy

.. code:: shell

    python demo_for_easyreg_eval.py  --run_demo --demo_name=ants -txt=./oai_examples.txt -o=OUTPUT_PATH



Demos on optimization-based mermaid model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Vector momentum-parameterized Stationary Velocity Field (vSVF) [`link <https://arxiv.org/pdf/1903.08811.pdf>`_]

.. code:: shell

    python demo_for_easyreg_eval.py  --run_demo --demo_name=opt_vsvf -txt=./oai_examples.txt -g=0 -o=OUTPUT_PATH


2. Region-specific Diffeomorphic Metric Mapping (RDMM) with pre-defined regularizer [`link <https://arxiv.org/pdf/1906.00139.pdf>`_]

.. code:: shell

    python demo_for_easyreg_eval.py  --run_demo --demo_name=opt_rdmm_predefined -txt=./lung_examples.txt -g=0 -o=OUTPUT_PATH



Demos on evaluating pretrained learning-based mermaid model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Affine-vSVF-Mapping [`link <https://arxiv.org/pdf/1903.08811.pdf>`_]

.. code:: shell

    python demo_for_easyreg_eval.py  --run_demo --demo_name=eval_network_vsvf -txt=./oai_examples.txt -g=0 -o=OUTPUT_PATH


2. RDMM network with a learnt regularizer [`link <https://arxiv.org/pdf/1906.00139.pdf>`_]

.. code:: shell

    python demo_for_easyreg_eval.py  --run_demo --demo_name=eval_network_rdmm -txt=./oai_examples.txt -g=0 -o=OUTPUT_PATH



Demos on training mermaid network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
More training details, please refer to :ref:`train_your_own_model`

.. code:: shell

    python demo_for_easyreg_train.py -o=./demo_training_reg_net -dtn=oai -tn=training_on_3_cases -ts=./demo_settings/mermaid/training_on_3_cases --train_affine_first -g=0  --is_demo


Demos on training VoxelMorph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. The CVPR 2018 VoxelMorph [`link <https://arxiv.org/abs/1809.05231>`_]

.. code:: shell

    python demo_for_easyreg_train.py  -dtn=oai -tn=training_vm_cvpr -ts=./demo_settings/mermaid/training_on_3_cases_voxelmorph -g=0 -o=OUTPUT_PATH

2. The MICAAI 2018 Diffeomorphic Version [`link <https://arxiv.org/abs/1805.04605>`_]

.. code:: shell

    python demo_for_easyreg_train.py  -dtn=oai -tn=training_vm_miccai -ts=./demo_settings/mermaid/training_on_3_cases_voxelmorph_miccai -g=0 -o=OUTPUT_PATH



Demos on data augmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For more training details, please refer to (to be added).

1. Anatomical augmentation with mermaid optimization

.. code:: shell

    python demo_for_data_aug.py --run_demo --demo_name=opt_lddmm_lpba -g 0 1 2 3 0 1 2 3

2. Anatomical augmentation with learnt mermaid network

.. code:: shell

    python demo_for_data_aug.py --run_demo --demo_name=learnt_lddmm_oai -g 0

3. Random augmentation with Bspline

.. code:: shell

    python gen_aug_samples.py -t=./data_aug_demo_output/rand_bspline_lpba/input.txt --bspline  -as=./demo_settings/data_aug/rand_bspline_lpba/data_aug_setting.json -o=./data_aug_demo_output/rand_bspline_lpba/aug


4. Random augmentation with Fluid-based model

.. code:: shell

    python gen_aug_samples.py -t=./data_aug_demo_output/rand_lddmm_oai/input.txt -as=./demo_settings/data_aug/rand_lddmm_oai/data_aug_setting.json -ms=./demo_settings/data_aug/rand_lddmm_oai/mermaid_nonp_settings.json -o=./data_aug_demo_output/rand_lddmm_oai/aug



Demos on training BrainStorm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This demo need set input data and some additional steps to generate new data.
We didn't put data generation code here, but sample codes can be found in data_pre/reg_process_example/gen_from_brainstorm.py

1. The transformation network of Brainstorm [`link <https://arxiv.org/abs/1902.09383>`_]

.. code:: shell

    python demo_for_easyreg_train.py  -dtn=DATA_TASK_NAME -tn=training_brainstorm_tf -ts=./demo_settings/mermaid/training_brainstorm_transform -g=0 -o=OUTPUT_PATH

2. The appearance network of Brainstorm

.. code:: shell

    python demo_for_easyreg_train.py  -dtn=DATA_TASK_NAME -tn=training_brainstorm_ap -ts=./demo_settings/mermaid/training_brainstorm_appearance -g=0 -o=OUTPUT_PATH
