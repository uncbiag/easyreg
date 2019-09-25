Prepare Data
=============

To be consistent among methods, we suggest users to normalize data into [0,1] (toolkit methods like NiftyReg, Ants and Demons not requires that) and saved as nii.gz.


Non-Training Tasks
^^^^^^^^^^^^^^^^^^

For non-training tasks, if the number of input pairs are too large to input as command argument, it would be convenient to put the pairs into a txt file formated as following:

* each line of the txt include 4 terms: s_pth t_pth ls_path lt_path

*s refers to source, t refers to target, ls refers to label of source (string None if not exist), lt refers to label of target (string None if not exist)*






Training Tasks
^^^^^^^^^^^^^^
we assume there is three level folder, output_root_path/ data_task_folder/ task_folder

* In data_task_folder, each folder refer to different preprocessing strategy, i.e. resampling into different size,
* In task_folder, each folder refer to a specific setting

so the task folder would be created as *output_root_path/data_task_folder/your_current_task_name*

For training tasks, the data should be organized as following:

* **train**, **val**,  **test**, **debug** (subset of train data, to check overfit)  folder should be put under **output_root_path/data_task_folder**, each of the folder should  include **pair_path_list.txt** and **pair_name_list.txt**
* **pair_path_list.txt**: each line of the txt include 4 terms: s_pth t_pth ls_path lt_path
* **pair_name_list.txt**: each line of the txt include 1 term: the pair name  the file is line to line corresponded with pair_path_list.txt

