Prepare Data
=============

For either optimization- or learning- based methods, SimpleITK is used as the image IO and our test is based on "nii.gz" format.

\*TODO, SimpleITK will be replaced by ITK in the future release.\*



Non-Training Tasks
^^^^^^^^^^^^^^^^^^

For non-training tasks, we support two kind of input format.

- put image list in command line.
- put image list in .txt file, each line of the txt include 4 terms: s_pth t_pth ls_path lt_path

**s** refers to source, **t** refers to target, **ls** refers to label of source (leave empty or string "None" if not exist), **lt** refers to label of target (leave empty or string "None" if not exist)*

Example:

- command line:
.. code::

    python demo_for_easyreg_eval.py --setting_folder_path ./demo_settings/mermaid/opt_vsvf --gpu_id 0  --task_output_path ./demo_output/mermaid/opt_vsvf -s ./examples/9352883_20051123_SAG_3D_DESS_LEFT_016610798103_image.nii.gz ./examples/9761431_20051103_SAG_3D_DESS_RIGHT_016610945809_image.nii.gz -t ./examples/9403165_20060316_SAG_3D_DESS_LEFT_016610900302_image.nii.gz ./examples/9211869_20050131_SAG_3D_DESS_RIGHT_016610167512_image.nii.gz

- txt file:
.. code::

    python demo_for_easyreg_eval.py --setting_folder_path ./demo_settings/mermaid/opt_vsvf --gpu_id 0  --task_output_path ./demo_output/mermaid/opt_vsvf --pair_txt_path ./oai_examples.txt



Training Tasks
^^^^^^^^^^^^^^
we assume there is three level folder, **output_root_path**/ **data_task_folder**/ **task_folder**

* In **data_task_folder**, each folder refer to different preprocessing strategies or different datasets, i.e. lung tasks and brain tasks.
* In **task_folder**, each folder refer to a specific setting

so the folders would be created as *output_root_path/data_task_folder/your_current_task_name*

For training tasks, the data should be organized as following:

* **train**, **val**,  **test**, **debug** (subset of train data, to check overfit)  folder should be put under **output_root_path/data_task_folder**, each of the folder should  include **pair_path_list.txt** and **pair_name_list.txt**
* **pair_path_list.txt**: each line of the txt include 4 terms: s_pth t_pth ls_path lt_path
* **pair_name_list.txt**: each line of the txt include 1 term: the pair name  the file is line to line corresponded with pair_path_list.txt

