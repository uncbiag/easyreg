import matplotlib as matplt

matplt.use('Agg')
import os, sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../easy_reg'))

import tools.module_parameters as pars
from abc import ABCMeta, abstractmethod
from easyreg.piplines import run_one_task
from easyreg.reg_data_utils import write_list_into_txt, get_file_name, read_txt_into_list
import torch
torch.backends.cudnn.benchmark=True



class BaseTask():
    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def save(self):
        pass


class DataTask(BaseTask):
    """
    base module for data setting files (.json)
    """

    def __init__(self, name, path='../settings/base_data_settings.json'):
        super(DataTask, self).__init__(name)
        self.data_par = pars.ParameterDict()
        self.data_par.load_JSON(path)

    def save(self, path='../settings/data_settings.json'):
        self.data_par.write_ext_JSON(path)


class ModelTask(BaseTask):
    """
    base module for task setting files (.json)
    """

    def __init__(self, name, path='../settings/base_task_settings.json'):
        super(ModelTask, self).__init__(name)
        self.task_par = pars.ParameterDict()
        self.task_par.load_JSON(path)

    def save(self, path='../settings/task_settings.json'):
        self.task_par.write_ext_JSON(path)




def force_test_setting(dm, tsm, output_path):
    """
    To run in test mode, force set related param in datapro and tsk_set.
    The updated param are saved in output_path/cur_data_setting.json and output_path/cur_task_setting.json

    :param dm:  ParameterDict, settings for data proprecessing (disabled if the settings have already put in tsk_set)
    :param tsm: ParameterDict, settings for the task
    :param output_path:
    :return: None
    """
    if dm is not None:
        data_json_path = os.path.join(output_path, 'cur_data_setting.json')
        dm.data_par['datapro']['dataset']['prepare_data'] = False
        dm.data_par['datapro']['reg']['max_num_for_loading'] = [1, 1, -1, 1]
        dm.save(data_json_path)
    else:
        tsm.task_par['dataset']['max_num_for_loading'] = [1, 1, -1, 1]
    tsm.task_par['tsk_set']['train'] = False
    tsm.task_par['tsk_set']['continue_train'] = False
    tsk_json_path = os.path.join(output_path, 'cur_task_setting.json')
    tsm.save(tsk_json_path)


def init_test_env(setting_path, output_path, file_list,fname_list):
    """
    create test environment, the file list would be saved into output_path/reg/test/file_path_list.txt,
     a corresponding auto-parsed filename list would also be saved in output/path/reg/test/file_name_list.txt

    :param setting_path: the path to load 'cur_task_setting.json' and 'cur_data_setting.json' (optional if the related settings are in cur_task_setting)
    :param output_path: the output path of the task
    :param image_path_list: the image list, each item refers to the abstract path of the image
    :param l_path_list:optional, the label of image list, each item refers to the abstract path of the image
    :return: tuple of ParameterDict,  datapro (optional) and tsk_set
    """
    dm_json_path = os.path.join(setting_path, 'cur_data_setting.json')
    tsm_json_path = os.path.join(setting_path, 'cur_task_setting.json')
    assert os.path.isfile(tsm_json_path), "task setting not exists"
    dm = DataTask('task_reg', dm_json_path) if os.path.isfile(dm_json_path) else None
    tsm = ModelTask('task_reg', tsm_json_path)
    file_num = len(file_list)
    os.makedirs(os.path.join(output_path, 'seg/test'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'seg/res'), exist_ok=True)
    file_txt_path = os.path.join(output_path, 'seg/test/file_path_list.txt')
    fn_txt_path = os.path.join(output_path, 'seg/test/file_name_list.txt')
    has_label = len(file_list[0])==2
    if fname_list is None:
        if has_label:
            fname_list = [get_file_name(file_list[i][0]) for i in range(file_num)]
        else:
            fname_list = [get_file_name(file_list[i]) for i in range(file_num)]
    write_list_into_txt(file_txt_path, file_list)
    write_list_into_txt(fn_txt_path, fname_list)
    data_task_name = 'seg'
    cur_task_name = 'res'
    if dm is not None:
        dm.data_par['datapro']['dataset']['output_path'] = output_path
        dm.data_par['datapro']['dataset']['task_name'] = data_task_name
    tsm.task_par['tsk_set']['task_name'] = cur_task_name
    tsm.task_par['tsk_set']['output_root_path'] = os.path.join(output_path, data_task_name)
    return dm, tsm


def do_segmentation_eval(args, segmentation_file_list):
    """
    set running env and run the task

    :param args: the parsed arguments
    :param segmentation_file_list:  list of segmentation file list, [image_list, label_list]
    :return: None
    """
    task_output_path = args.task_output_path
    os.makedirs(task_output_path, exist_ok=True)
    setting_folder_path = args.setting_folder_path
    file_txt_path = ''
    if args.file_txt_path:
        file_txt_path = args.file_txt_path
        fname_txt_path = os.path.join(os.path.split(file_txt_path)[0],"file_name_list.txt")
        fname_list = read_txt_into_list(fname_txt_path) if os.path.isfile(fname_txt_path) else None
    else:
        print(segmentation_file_list)
        fname_list = [[f.split('/')[-1].split('.')[0] for f in segmentation_file_list[0]]]*2
    dm, tsm = init_test_env(setting_folder_path, task_output_path, segmentation_file_list, fname_list)
    tsm.task_par['tsk_set']['gpu_ids'] = args.gpu_id
    model_path= args.model_path
    if model_path is not None:
        assert os.path.isfile(model_path), "the model {} not exist".format_map(model_path)
        tsm.task_par['tsk_set']['model_path'] = model_path
    force_test_setting(dm, tsm, task_output_path)

    dm_json_path = os.path.join(task_output_path, 'cur_data_setting.json') if dm is not None else None
    tsm_json_path = os.path.join(task_output_path, 'cur_task_setting.json')
    run_one_task(tsm_json_path, dm_json_path)


if __name__ == '__main__':
    """
    A evaluation interface for segmentation network with pre-trained models.

    Arguments:
       
        input related:two input styles are supported,
            1. given txt
             --file_txt_path/-txt: the txt file recording the paths of images to segmentation
            2. given image
            --image_list/ -i: the image list,  s1 s2 s3..sn
            --limage_list/ -li: optional, the label list,  ls1,ls2,ls3..lsn
        other arguments:
             --setting_folder_path/-ts :path of the folder where settings are saved
             --task_output_path/ -o: the path of output folder
             --model_path/ -m: the path of pretrained model, can be set here or set in setting file
             --gpu_id/ -g: gpu_id to use


    """
    import argparse

    parser = argparse.ArgumentParser(description='An easy interface for evaluate various segmentation methods')
    parser.add_argument('-ts', '--setting_folder_path', required=False, type=str,
                        default=None,
                        help='path of the folder where settings are saved,should include cur_task_setting.json')
    parser.add_argument('-txt', '--file_txt_path', required=False, default=None, type=str,
                        help='the txt file recording the paths of images for segmentation')  # 2
    parser.add_argument('-i', '--image_list', nargs='+', required=False, default=None,
                        help='the image list,  s1 s2 s3..sn')
    parser.add_argument('-li', '--limage_list', nargs='+', required=False, default=None,
                        help='the image label list,  ls1,ls2,ls3..lsn')
    parser.add_argument('-o', "--task_output_path", required=True, default=None, help='the output path')
    parser.add_argument('-m', "--model_path", required=False, default=None, help='the path of trained model')
    parser.add_argument('-g', "--gpu_id", required=False, type=int, default=0, help='gpu_id to use')

    args = parser.parse_args()
    print(args)
    file_txt_path = args.file_txt_path
    image_list = args.image_list
    limage_list = args.limage_list
    image_label_list = []

    assert file_txt_path is not None or image_list is not None, "either file_txt_path or source/target_list should be provided"
    assert file_txt_path is None or image_list is None, " file_txt_path and source/target_list cannot be both provided"
    if file_txt_path is not None:
        image_label_list = read_txt_into_list(file_txt_path)
    
    if limage_list is not None:
        assert len(image_list) == len(limage_list), "the image_list and limage_list should be the same length"
        with open('file_path_list.txt', 'w+') as f:
            f.write('{}\t{}'.format(image_list[0], limage_list[0]))
        args.file_txt_path = 'file_path_list.txt'
        image_label_list = read_txt_into_list('file_path_list.txt')
        args.image_list = None
        args.limage_list = None

    do_segmentation_eval(args, image_label_list)
