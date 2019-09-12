import matplotlib as matplt
matplt.use('Agg')
import SimpleITK as sitk
import sys,os
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('../model_pool'))
sys.path.insert(0,os.path.abspath('../mermaid'))
print(sys.path)
import data_pre.module_parameters as pars
import subprocess
from abc import ABCMeta, abstractmethod
from model_pool.piplines import run_one_task
from data_pre.reg_data_utils import write_list_into_txt, get_file_name, loading_img_list_from_files

class BaseTask():
    __metaclass__ = ABCMeta
    def __init__(self,name):
        self.name = name

    @abstractmethod
    def save(self):
        pass

class DataTask(BaseTask):
    """
    base module for data setting files (.json)
    """
    def __init__(self,name,path='../settings/base_data_settings.json'):
        super(DataTask,self).__init__(name)
        self.data_par = pars.ParameterDict()
        self.data_par.load_JSON(path)


    def save(self, path='../settings/data_settings.json'):
        self.data_par.write_ext_JSON(path)

class ModelTask(BaseTask):
    """
    base module for task setting files (.json)
    """
    def __init__(self,name,path='../settings/base_task_settings.json'):
        super(ModelTask,self).__init__(name)
        self.task_par = pars.ParameterDict()
        self.task_par.load_JSON(path)

    def save(self,path= '../settings/task_settings.json'):
        self.task_par.write_ext_JSON(path)



def force_test_setting(dm, tsm,output_path):
    """
    To run in test mode, force set related param in datapro and tsk_set.
    The updated param are saved in output_path/cur_data_setting.json and output_path/cur_task_setting.json

    :param dm:  ParameterDict, settings for data proprecessing (disabled if the settings have already put in tsk_set)
    :param tsm: ParameterDict, settings for the task
    :param output_path:
    :return:
    """
    if dm is not None:
        data_json_path = os.path.join(output_path, 'cur_data_setting.json')
        dm.data_par['datapro']['dataset']['prepare_data'] = False
        dm.data_par['datapro']['reg']['max_pair_for_loading'] = [1, 1, -1, 1]
        dm.save(data_json_path)
        #dm.save()
    else:
        tsm.task_par['dataset']['max_pair_for_loading'] = [1, 1, -1, 1]
    tsm.task_par['tsk_set']['train'] = False
    tsm.task_par['tsk_set']['save_by_standard_label'] = True
    tsm.task_par['tsk_set']['continue_train'] = False
    tsm.task_par['tsk_set']['reg']['mermaid_net']['using_sym'] = False
    tsk_json_path = os.path.join(output_path, 'cur_task_setting.json')
    tsm.save(tsk_json_path)
    #tsm.save()





def init_test_env(setting_path,output_path, source_path_list, target_path_list, l_source_path_list=None, l_target_path_list=None):
    """
    create test environment, the pair list would be saved into output_path/reg/test/pair_path_list.txt,
     a corresponding auto-parsed filename list would also be saved in output/path/reg/test/pair_name_list.txt

    :param setting_path: the path to load 'cur_task_setting.json' and 'cur_data_setting.json' (optional if the related settings are in cur_task_setting)
    :param output_path: the output path of the task
    :param source_path_list: the source image list, each item refers to the abstract path of the image
    :param target_path_list: the target image list,  each item refers to the abstract path of the image
    :param l_source_path_list: optional, the label of source image list, each item refers to the abstract path of the image
    :param l_target_path_list:optional, the label of target image list, each item refers to the abstract path of the image
    :return: tuple of ParameterDict,  datapro (optional) and tsk_set
    """
    dm_json_path = os.path.join(setting_path, 'cur_data_setting.json')
    tsm_json_path = os.path.join(setting_path, 'cur_task_setting.json')
    assert os.path.isfile(tsm_json_path),"task setting not exists"
    dm = DataTask('task_reg',dm_json_path) if os.path.isfile(dm_json_path) else None
    tsm = ModelTask('task_reg',tsm_json_path)
    file_num = len(source_path_list)
    if l_source_path_list is not None and l_target_path_list is not None:
        file_list = [[source_path_list[i], target_path_list[i],l_source_path_list[i],l_target_path_list[i]] for i in range(file_num)]
    else:
        file_list = [[source_path_list[i], target_path_list[i]] for i in range(file_num)]
    os.makedirs(os.path.join(output_path,'reg/test'),exist_ok=True)
    os.makedirs(os.path.join(output_path,'reg/res'),exist_ok=True)
    pair_txt_path = os.path.join(output_path,'reg/test/pair_path_list.txt')
    fn_txt_path = os.path.join(output_path,'reg/test/pair_name_list.txt')
    fname_list = [get_file_name(file_list[i][0])+'_'+get_file_name(file_list[i][1]) for i in range(file_num)]
    write_list_into_txt(pair_txt_path,file_list)
    write_list_into_txt(fn_txt_path,fname_list)
    data_task_name = 'reg'
    cur_task_name = 'res'
    if dm is not None:
        dm.data_par['datapro']['dataset']['output_path'] = output_path
        dm.data_par['datapro']['dataset']['task_name'] = data_task_name
    tsm.task_par['tsk_set']['task_name'] = cur_task_name
    tsm.task_par['tsk_set']['data_folder'] = os.path.join(output_path,data_task_name)
    return dm, tsm




def do_registration_eval(args, registration_pair_list):
    """
    set running env and run the task

    :param args: the parsed arguments
    :param registration_pair_list:  list of registration pairs, [source_list, target_list, lsource_list, ltarget_list]
    :return: None
    """
    task_output_path = args.task_output_path
    os.makedirs(task_output_path, exist_ok=True)
    run_demo = args.run_demo
    if run_demo:
        demo_name = args.demo_name
        if demo_name not in['ants','demons','nifty_reg']:
            setting_folder_path = os.path.join('./demo_settings/mermaid',demo_name)
        else:
            setting_folder_path = os.path.join('./demo_settings', demo_name)
        assert os.path.isdir(setting_folder_path), "the {} is not in supported demo list, network_rdmm/network_vsvf/opt_vsvf/opt_rdmm/opt_rdmm_predefined/ants/demons/niftyreg"
        #task_output_path = os.path.join('./demo_output/mermaid',demo_name)
    else:
        setting_folder_path = args.setting_folder_path
    source_path_list, target_path_list, l_source_path_list, l_target_path_list = registration_pair_list
    dm, tsm = init_test_env(setting_folder_path,task_output_path,source_path_list,target_path_list,l_source_path_list,l_target_path_list)
    tsm.task_par['tsk_set']['gpu_ids'] = args.gpu_id
    if not tsm.task_par['tsk_set']['train']:
        force_test_setting(dm, tsm, task_output_path)

    dm_json_path = os.path.join(task_output_path, 'cur_data_setting.json') if dm is not None else None
    tsm_json_path = os.path.join(task_output_path, 'cur_task_setting.json')
    run_one_task(tsm_json_path, dm_json_path)





if __name__ == '__main__':
    """
    A evaluation interface for optimization methods or learning methods with pre-trained models.
    Though the purpose of this script is to provide demo, it is a generalized interface for evaluating the following methods.
    The method support list :  mermaid-related ( optimizing/pretrained) methods, ants, demons, niftyreg
    The demos supported by category are : 
        mermaid: network_rdmm/network_vsvf/opt_vsvf/opt_rdmm/opt_rdmm_predefined
        ants: ants
        demons: demons
        niftyreg: niftyreg
    * network_* refers to learning methods with pre-trained models
    * opt_* : refers to optimization based methods
    """
    import argparse

    parser = argparse.ArgumentParser(description='An easy interface for evaluate various registration methods')
    parser.add_argument("--run_demo",required=False,action='store_true',help='run demo')
    parser.add_argument('--demo_name',required=False, type=str, default='opt_vsvf',help='if run_demo, network_rdmm/network_vsvf/opt_vsvf/opt_rdmm/opt_rdmm_predefined')
    #---------------------------------------------------------------------------------------------------------------------------
    parser.add_argument('-ts','--setting_folder_path', required=False, type=str,
                        default=None,help='path to load settings')
    parser.add_argument('-txt','--pair_txt_path', required=False, default='./oai_examples_debug.txt', type=str,
                        help='the txt file recording the pairs to registration')  # 2
    parser.add_argument('-s','--source_list',nargs='+', required=False, default=None,
                        help='the source list,  s1 s2 s3..sn')
    parser.add_argument('-t','--target_list',nargs='+', required=False, default=None,
                        help='the target list,  t1 t2 t3..tn')
    parser.add_argument('-ls','--lsource_list',nargs='+', required=False, default=None,
                        help='the source label list,  ls1,ls2,ls3..lsn')
    parser.add_argument('-lt','--ltarget_list',nargs='+', required=False, default=None,
                        help='the target label list,  lt1,lt2,lt3..ltn')
    parser.add_argument('-o',"--task_output_path",required=True,default=None, help='the output path')
    parser.add_argument('-g',"--gpu_id",required=False,type=int,default=0,help='gpu_id to use')

    args = parser.parse_args()
    print(args)
    pair_txt_path = args.pair_txt_path
    source_list = args.source_list
    target_list = args.target_list
    lsource_list = args.lsource_list
    ltarget_list = args.ltarget_list

    assert pair_txt_path is not None or source_list is not None, "either pair_txt_path or source/target_list should be provided"
    assert pair_txt_path is None or source_list is None, " pair_txt_path and source/target_list cannot be both provided"
    if pair_txt_path is not None:
        source_list, target_list,lsource_list,ltarget_list = loading_img_list_from_files(pair_txt_path)
    if source_list is not None:
        assert len(source_list) == len(target_list), "the source and target list should be the same length"
    if lsource_list is not None:
        assert len(lsource_list)== len(source_list), "the lsource and source list should be the same length"
        assert len(lsource_list)== len(ltarget_list), " the lsource and ltarget list should be the same length"
    registration_pair_list= [source_list, target_list, lsource_list, ltarget_list]
    do_registration_eval(args, registration_pair_list)


    #python demo_for_easyreg_eval.py --run_demo --demo_name=demons -o=/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/new_interface_dev2/test_demons
    #python demo_for_easyreg_eval.py  --run_demo --demo_name=nifty_reg  -o=/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/new_interface_dev2/test_nifty_reg
    #python demo_for_easyreg_eval.py  --run_demo --demo_name=ants  -o=/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/new_interface_dev2/test_ants
    #python demo_for_easyreg_eval.py  --run_demo --demo_name=opt_rdmm_predefined -txt=./lung_examples.txt -g=3 -o=/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/new_interface_dev2/test_opt_rdmm_predefined
    #python demo_for_easyreg_eval.py  --run_demo --demo_name=opt_vsvf -g=2 -o=/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/new_interface_dev2/test_opt_vsvf



    #python demo_for_easyreg_eval.py --run_demo --demo_name=demons -txt=/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/test/pair_path_list.txt -o=/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/new_interface/test_demons
    #python demo_for_easyreg_eval.py  --run_demo --demo_name=nifty_reg -txt=/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/test/pair_path_list.txt -o=/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/new_interface/test_nifty_reg
    #python demo_for_easyreg_eval.py  --run_demo --demo_name=ants -txt=/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/test/pair_path_list.txt -o=/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/new_interface/test_ants
    #python demo_for_easyreg_eval.py  --run_demo --demo_name=opt_vsvf -txt=/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/test/pair_path_list.txt -g=3 -o=/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/new_interface/test_opt_vsvf
    #python demo_for_easyreg_eval.py  --run_demo --demo_name=opt_rdmm_predefined -txt=/playpen/zyshen/data/reg_lung_160/test/pair_path_list.txt -g=3 -o=/playpen/zyshen/data/reg_lung_160/new_interface/test_opt_rdmm_predefined
    #python demo_for_easyreg_eval.py  --run_demo --demo_name=network_vsvf -txt=/playpen/zyshen/debugs/get_val_and_debug_res/test.txt -g=3 -o=/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/new_interface/test_vsvf_net
    #python demo_for_easyreg_eval.py  --run_demo --demo_name=network_rdmm -txt=/playpen/zyshen/debugs/get_val_and_debug_res/test.txt -g=3 -o=/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/new_interface/test_rdmm_net
