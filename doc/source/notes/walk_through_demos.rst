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
4. Demos on training learning-based mermaid model(vSVF and RDMM)

*Demos for RDMM on synthesis data is put in mermaid repository. if you are interested in that, please refer to mermaid tutorial.*


Download Examples and Pretrained Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To run the demo, we first need to download examples and pretrained model for learning based methods.

.. code:: shell

    gdown https://drive.google.com/uc?id=1RI7YevByrLAKy1JTv6KG4RSAnHIC7ybb
    unzip demo.zip -d EASYREG_REPOSITORY_PATH

Now we are ready to play with demos. The repository *demo* lists the scripts and settings for demos.

In this section, the script *demo_for_easyreg_eval.py* will be introduced which is for optimization-based or pretrained methods.

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
1. For NiftyReg

.. code:: shell

    python demo_for_easyreg_eval.py  --run_demo --demo_name=nifty_reg  -txt=./oai_examples.txt -o=OUTPUT_PATH

2. For Demons

.. code:: shell

    python demo_for_easyreg_eval.py  --run_demo --demo_name=demons -txt=./oai_examples.txt -o=OUTPUT_PATH

3. For AntsPy

.. code:: shell

    python demo_for_easyreg_eval.py  --run_demo --demo_name=ants -txt=./oai_examples.txt -o=OUTPUT_PATH



Demos on optimization-based mermaid model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. For Vector momentum-parameterized Stationary Velocity Field (vSVF) [`link <https://arxiv.org/pdf/1903.08811.pdf>`_]

.. code:: shell

    python demo_for_easyreg_eval.py  --run_demo --demo_name=opt_vsvf -txt=./oai_examples.txt -g=0 -o=OUTPUT_PATH


2. For Region-specific Diffeomorphic Metric Mapping (RDMM) with pre-defined regularizer [`link <https://arxiv.org/pdf/1906.00139.pdf>`_]

.. code:: shell

    python demo_for_easyreg_eval.py  --run_demo --demo_name=opt_rdmm_predefined -txt=./lung_examples.txt -g=0 -o=OUTPUT_PATH



Demos on evaluating pretrained learning-based mermaid model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. For Affine-vSVF-Mapping [`link <https://arxiv.org/pdf/1903.08811.pdf>`_]

.. code:: shell

    python demo_for_easyreg_eval.py  --run_demo --demo_name=eval_network_vsvf -txt=./oai_examples.txt -g=0 -o=OUTPUT_PATH


2. For RDMM network with a learnt regularizer [`link <https://arxiv.org/pdf/1906.00139.pdf>`_]

.. code:: shell

    python demo_for_easyreg_eval.py  --run_demo --demo_name=eval_network_rdmm -txt=./oai_examples.txt -g=0 -o=OUTPUT_PATH




