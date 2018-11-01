# registration_net
net for image registration


One Stop Mapping: Joint Networks for Longitudinal Image Registration

https://v2.overleaf.com/18560374xmmwkmtqmpzw

This is the concise version for registration network

# Guide
1 . prepare the data
 
Though the dataset processing code is provided, we recommend to write your own
code for specific dataset.
And take the following format

* image itself should be normalized into 0-1 and saved as nii.gz (or you can normalize it in dataloader)
* **train**, **val**,  **test**, **debug** (subset of train data, to check overfit)  folder should be put under **data_folder/data_task_folder**, each of the folder should
   include **'pair_path_list.txt'** and **'pair_name_list.txt'**\
   **'pair_path_list.txt'**: each line of the txt include 4 terms: s_pth t_pth ls_path lt_path\
    s refers to source, t refers to target, ls refers to label of source (put None if not exist)
    **'pair_name_list.txt'**: each line of the txt include 1 term: the pair name
    the file is line to line corresponded with 'pair_path_list.txt'

2 . settings in demo
* general settings for paths\
We assume different tasks would take the same data, so the task folder would be created as data_folder/data_task_folder/your_current_task_name\
```
dm.data_par['datapro']['dataset']['output_path'] = data_folder
dm.data_par['datapro']['dataset']['task_name'] = data_task_folder
tsm.task_par['tsk_set']['task_name'] = your_current_task_name
```

In the 'your_current_task_name' folder, three folder will be auto created, **log** for tensorboard, **checkpoints** for saving models,
**records** for saving running time results. Besides, two files will also be created. **task_settings.json** for recording settings of current tasks.
**logfile.log** for terminal output ( only flushed when task finished)


* general settings for tasks\
models support list: 'reg_net'  'mermaid_iter'  'ants'  'nifty_reg' 'demons'\
each model supports several methods
methods support by reg_net: 'affine_sim','affine_cycle','affine_sym', 'mermaid'\
methods support by mermaid_iter: 'affine','svf' ( including affine registration first)\
methods support by ants: 'affine','syn' ( including affine registration first)\
methods support by nifty_reg: 'affine','bspline' ( including affine registration first)\
methods support by demons: 'demons' ( including niftyreg affine registration first)\

**reg_net** refers to registration network. 'affine_sim' refers to single affine network, 'affine_cycle' refers to 
multi-step affine network, 'affine_sym' refers to multi-step affine symmetric network (s-t, t-s), 'mermaid' refers to the 
mermaid library ( including various fluid based registration methods, our implementation is based on velocity momentum based svf method)\
**mermaid_iter** refers to mermaid library, here we compare with 'affine' and 'svf' ( actually the velocity momentum based svf) method\
**ant** refers to AntsPy: https://github.com/ANTsX/ANTsPy\
** nifty_reg ** refers : http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg\
**demons** refers to deformably register two images using a symmetric forces demons
    algorithm, which is provided by simple itk
    
e.g setting reg_net with affine_sim
```
tsm.task_par['tsk_set']['network_name'] ='affine_sim'  #'mermaid' 'svf' 'syn' affine bspline
tsm.task_par['tsk_set']['model'] = 'reg_net'
```

 





    
