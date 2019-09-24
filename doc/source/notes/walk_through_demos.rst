Walk through Demos
========================================

EasyReg is developed for research propose, providing interfaces for various registration methods, including `AntsPy <https://github.com/ANTsX/ANTsPy>`_ , `NiftyReg <http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg>`_ and Demons (embedded in `SimpleITK <http://www.simpleitk.org/SimpleITK/resources/software.html>`_).

In this tutorial, we would step by step showing how to run following demos:

1. Demos on toolkit methods (NiftyReg, AntsPy, Demons)
2. Demos on optimization-based mermaid model (vSVF in CVPR paper and RDMM in NeurIPS paper)
3. Demos on evaluating pretrained learning-based mermaid model (vSVF in CVPR paper and RDMM in NeurIPS paper)
4. Demos on training learning-based mermaid model(vSVF in CVPR paper and RDMM in NeurIPS paper)


Download Examples and Pretrained Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To run the demo, we first need to download examples and pretrained model for learning based methdods.

.. code::

    gdown https://drive.google.com/open?id=1RjFV0lht4uQFc2jYmBYxmXtRrdYAzk8S
    unzip demo.zip -d EASYREG_PATH

Now we are ready to play with demos. The script for demos are in *demo*.

The script *demo_for_easyreg_eval* is for optimization-based or pretrained methods.

.. code::

    A evaluation interface for optimization methods or learning methods with pre-trained models.
    Though the purpose of this script is to provide demo, it is a generalized interface for evaluating the following methods.
    The method support list :  mermaid-related ( optimizing/pretrained) methods, ants, demons, niftyreg
    The demos supported by category are :
        mermaid: eval_network_rdmm/eval_network_vsvf/opt_vsvf/opt_rdmm/opt_rdmm_predefined
        ants: ants
        demons: demons
        niftyreg: niftyreg
    * network_* refers to learning methods with pre-trained models
    * opt_* : refers to optimization based methods
    Arguments:
        demo related:
             --run_demo: run the demo
             --demo_name: eval_network_rdmm/eval_network_vsvf/opt_vsvf/opt_rdmm/opt_rdmm_predefined/ants/demons/niftyreg
        input related:two input styles are supported. For both cases, images should be first normalized into [0,1].
            1. given txt
             --pair_txt_path/-txt: the txt file recording the pairs to registration
            2. given image
            --source_list/ -s: the source list,  s1 s2 s3..sn
            --target_list/ -t: the target list,  t1 t2 t3..tn
            --lsource_list/ -ls: optional, the source label list,  ls1,ls2,ls3..lsn
            --ltarget_list/ -lt: optional, the target label list,  lt1,lt2,lt3..ltn
        other arguments:
             --setting_folder_path/ -ts :path of the folder where settings are saved
             --task_output_path/ -o: the path of output folder
             --gpu_id/ -g: gpu_id to use



Demos on toolkit methods
^^^^^^^^^^^^^^^^^^^^^^^^
1. For NiftyReg

.. code::

    python demo_for_easyreg_eval.py  --run_demo --demo_name=nifty_reg  -o=OUTPUT_PATH

2. For Demons

.. code::

    python demo_for_easyreg_eval.py  --run_demo --demo_name=demons  -o=OUTPUT_PATH

3. For AntsPy

.. code::

    python demo_for_easyreg_eval.py  --run_demo --demo_name=ants  -o=OUTPUT_PATH



Demos on optimization-based mermaid model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. For Vector momentum-parameterized Stationary Velocity Field (vSVF) [`link <https://arxiv.org/pdf/1903.08811.pdf>`_]

.. code::

    python demo_for_easyreg_eval.py  --run_demo --demo_name=opt_vsvf -g=0 -o=OUTPUT_PATH


2. For Region-specific Diffeomorphic Metric Mapping (RDMM) with pre-defined regularizer [`link <https://arxiv.org/pdf/1906.00139.pdf>`_]

.. code::

    python demo_for_easyreg_eval.py  --run_demo --demo_name=opt_rdmm_predefined -txt=./lung_examples.txt -g=0 -o=OUTPUT_PATH



Demos on evaluating pretrained learning-based mermaid model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. For Affine-vSVF-Mapping [`link <https://arxiv.org/pdf/1903.08811.pdf>`_]

.. code::

    python demo_for_easyreg_eval.py  --run_demo --demo_name=eval_network_vsvf -g=0 -o=OUTPUT_PATH


2. For RDMM network with a learnt regularizer [`link <https://arxiv.org/pdf/1906.00139.pdf>`_]

.. code::

    python demo_for_easyreg_eval.py  --run_demo --demo_name=eval_network_rdmm -txt=./lung_examples.txt -g=0 -o=OUTPUT_PATH




