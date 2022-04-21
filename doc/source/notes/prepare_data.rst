.. _prepare-data-reg-training-label:
Prepare Data
=============

For all methods, SimpleITK is used as the image IO and our test is based on "nii.gz" format.





Training for registration tasks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We assume there is three level folder, **output_root_path**/ **data_task_name**/ **task_name**.
The output_root_path/data_task_name must be created and well organized as following before running the train script.

* In **data_task_name**, each folder refer to different preprocessing strategies or different datasets, i.e. lung tasks and brain tasks. It might as well could be different splits of the data too.
* In **task_name**, each folder refer to a specific setting, for a specific dataset setting.
Thus, the folders will be created as *output_root_path/data_task_name/task_name*.

For training tasks, the data should be organized as following:

* **train**, **val**,  **test**, **debug** (subset of train data, to check overfit)  folder should be put under **output_root_path/data_task_name**, each of the folder should  include **pair_path_list.txt** and **pair_name_list.txt**
* **pair_path_list.txt**: each line of the txt include 4 terms: source_path, target_path, labels_for_source_image_path, labels_for_target_image_path
* **pair_name_list.txt**: each line of the txt include 1 term: the pair name.  The file has line to line corresponded with pair_path_list.txt

Alternatively,  *prepare_data.py* script is capable of generating corresponding directory structure, just using it will be enough.




.. _prepare-data-reg-eval-label:
Inference/Optimization in registration tasks
^^^^^^^^^^^^^^^^^^

For inference (with pretrained model) or optimization tasks, we support two kind of input format.

- put image list in command line.
- put image list in a .txt file, each line of the txt include 4 terms: s_pth t_pth ls_path lt_path

**s** refers to source, **t** refers to target, **ls** refers to label of source (leave empty or string "None" if not exist), **lt** refers to label of target (leave empty or string "None" if not exist)*

Example:

- command line:
.. code:: shell

    python demo_for_easyreg_eval.py --setting_folder_path ./demo_settings/mermaid/opt_vsvf --gpu_id 0  --task_output_path ./demo_output/mermaid/opt_vsvf -s ./oai_examples/9352883_20051123_SAG_3D_DESS_LEFT_016610798103_image.nii.gz ./oai_examples/9761431_20051103_SAG_3D_DESS_RIGHT_016610945809_image.nii.gz -t ./oai_examples/9403165_20060316_SAG_3D_DESS_LEFT_016610900302_image.nii.gz ./oai_examples/9211869_20050131_SAG_3D_DESS_RIGHT_016610167512_image.nii.gz

- txt file:
.. code:: shell

    python demo_for_easyreg_eval.py --setting_folder_path ./demo_settings/mermaid/opt_vsvf --gpu_id 0  --task_output_path ./demo_output/mermaid/opt_vsvf --pair_txt_path ./oai_examples.txt



.. _prepare-data-seg-training-label:

Train for segmentation tasks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We assume there is three level folder, **output_root_path**/ **data_task_name**/ **task_name**. Our *prepare_data.py* script is capable of generating corresponding directory structure, just using it will be enough.
The output_root_path/data_task_name must be created and well organized as following before running the train script.

* In **data_task_name**, each folder refer to different preprocessing strategies or different datasets, i.e. lung tasks and brain tasks. It might as well could be different splits of the data too.
* In **task_name**, each folder refer to a specific setting, for a specific dataset setting.
Thus, the folders will be created as *output_root_path/data_task_name/task_name*.

For training tasks, the data should be organized as following:

* **train**, **val**,  **test**, **debug** (subset of train data, to check overfit)  folder should be put under **output_root_path/data_task_name**, each of the folder should  include **file_path_list.txt** and **file_name_list.txt**
* **file_path_list.txt**: each line of the txt include 2 terms: img_path, labels_path
* **file_name_list.txt**: each line of the txt include 1 term: the filename.  The file has line to line corresponded with file_path_list.txt




.. _prepare-data-seg-eval-label:
Inference in segmentation Tasks
^^^^^^^^^^^^^^^^^^

For inference (with pretrained model) or optimization tasks, we support two kind of input format.

- put image list in command line.
- put image list in a .txt file, each line of the txt include 4 terms: s_pth t_pth ls_path lt_path

**s** refers to source, **t** refers to target, **ls** refers to label of source (leave empty or string "None" if not exist), **lt** refers to label of target (leave empty or string "None" if not exist)*

Example:

- command line:
.. code:: shell

    python demo_for_seg_eval.py --setting_folder_path ./demo_settings/seg/lpba_seg_eval --gpu_id 0  --task_output_path ./demo_output/seg/lpba_seg_eval -i ./lpba_examples/s3.nii.gz ./lpba_examples/s27.nii.gz

- txt file:
.. code:: shell

    python demo_for_seg_eval.py --setting_folder_path ./demo_settings/seg/lpba_seg_eval --gpu_id 0  --task_output_path ./demo_output/seg/lpba_seg_eval --file_txt_path ./lpba_examples.txt





Data Augmentation Tasks
^^^^^^^^^^^^^^^^^^^^^^^^
We support two different data augmentation strategy, random augmentation and anatomical augmentation.

* For the random augmentation, we support Bspine augmentation and fluid-based random augmentation.
* For the anatomical augmentation, we support random sampling and data inter-/extra-polation.

Both tasks take a txt file recording file paths as input, items in the same line are separated by the space:

* For the random augmentation, the augmentation takes place among different lines, each line refers to a image and corresponding label (string "None" if not exist).
* For the anatomical augmentation, the augmentation takes place in a line, each line refers to a path of source image, paths of target images and the source label (string "None" if not exist), the labels of target images(None if not exist).


Additionally, an optional input is a txt file recording filename:

* For the random augmentation, each line include a image name.
* For the anatomical augmentation, each line include a source name and a series of target names.
