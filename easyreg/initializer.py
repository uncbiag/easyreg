import tools.module_parameters as pars
from .data_manager import DataManager
import os
from tensorboardX import SummaryWriter
import sys


class Initializer():
    """
    The initializer for data manager,  log env and task settings
    """
    class Logger(object):
        """
        redirect the stdout into files
        """
        def __init__(self, task_path):
            self.terminal = sys.stdout
            self.log = open(os.path.join(task_path, "logfile.log"), "a",buffering=1)

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
            os.fsync(self.log.fileno())

        def flush(self):
            # this flush method is needed for python 3 compatibility.
            pass

    def init_task_option(self, setting_path='../settings/task_settings.json'):
        """
        load task settings from the task setting json
        some additional settings need to be done if the data manager is not initialized by the data processing json
        :param setting_path: the path of task setting json
        :return: ParameterDict, task settings
        """
        par_dataset = pars.ParameterDict()
        if self.task_root_path:
            par_dataset.load_JSON(os.path.join(self.task_root_path, 'cur_data_setting.json'))
        self.task_opt = pars.ParameterDict()
        self.task_opt.load_JSON(setting_path)
        #self.task_opt['tsk_set']['extra_info'] = self.get_info_dic()
        self.task_name = self.task_opt['tsk_set']['task_name']
        if self.task_root_path is None:
            self.task_root_path = self.task_opt['tsk_set']['output_root_path']
            par_dataset= self.task_opt['dataset']
            task_type = par_dataset[('task_type','reg',"task type can be 'reg' or 'seg'")]
            self.data_manager.set_task_type(task_type)
            self.data_manager.set_reg_option(par_dataset)
            self.data_manager.set_seg_option(par_dataset)
            self.data_manager.manual_set_task_root_path(self.task_root_path)
        else:
            self.task_opt['dataset'] = par_dataset
        return self.task_opt

    def get_task_option(self):
        """
        get current task settings
        :return:
        """
        return self.task_opt

    def initialize_data_manager(self, setting_path='../settings/data_settings.json', task_path=None):
        """
        if the data processing settings are given, then set the data manager according to the setting
        if not, then assume the settings are included in task settings, no further actions need to be set in data manager
        :param setting_path: the path of the data processing json
        :param task_path: the path of the task setting json (disabled)
        :return: None
        """
        if setting_path is not None:
            self.__initialize_data_manager(setting_path)
        else:
            self.task_root_path=None
            self.data_manager = DataManager(task_name=None, dataset_name=None)

    def __initialize_data_manager(self, setting_path='../settings/data_settings.json'):
        """
        if the data processing file is given , then data manager will be set according to the settings,
        the data manager prepares the data related environment including preprocessing the data into different sets ( train, val...)
        assigning  dataloaders for different phases( train,val.....)
        :param setting_path: the path of the settings of the data preprocessing json
        :return: None
        """
        par_dataset = pars.ParameterDict()
        par_dataset.load_JSON(setting_path)
        task_type = par_dataset['datapro'][('task_type','reg',"task type can be 'reg' or 'seg'")]
        # switch to exist task
        switch_to_exist_task = par_dataset['datapro']['switch']['switch_to_exist_task']
        task_root_path = par_dataset['datapro']['switch']['task_root_path']
    
        # work on current task
        dataset_name = par_dataset['datapro']['dataset']['dataset_name']
        data_pro_task_name = par_dataset['datapro']['dataset']['task_name']
        prepare_data = par_dataset['datapro']['dataset']['prepare_data']
        data_path = par_dataset['datapro']['dataset']['data_path']
        label_path = par_dataset['datapro']['dataset']['label_path']
        output_path = par_dataset['datapro']['dataset']['output_path']
        data_path = data_path if data_path.ext else None
        label_path = label_path if label_path.ext else None
        divided_ratio = par_dataset['datapro']['dataset']['divided_ratio']
    
    
        # settings for reg
        sched = par_dataset['datapro']['reg']['sched']
        reg_option = par_dataset['datapro']['reg']
        self.data_manager = DataManager(task_name=data_pro_task_name, dataset_name=dataset_name)
        self.data_manager.set_task_type(task_type)
        if switch_to_exist_task:
            self.data_manager.manual_set_task_root_path(task_root_path)
        else:
            self.data_manager.set_sched(sched)
            self.data_manager.set_data_path(data_path)
            self.data_manager.set_output_path(output_path)
            self.data_manager.set_label_path(label_path)
            self.data_manager.set_divided_ratio(divided_ratio)
            # reg
            self.data_manager.set_reg_option(reg_option)

            self.data_manager.generate_saving_path(auto=False)
            self.data_manager.generate_task_path()
            if prepare_data:
                self.data_manager.init_dataset()
                self.data_manager.prepare_data()
                par_dataset.load_JSON(setting_path)

            par_dataset.write_ext_JSON(os.path.join(self.data_manager.get_task_root_path(), 'data_settings.json'))
            task_root_path = self.data_manager.get_task_root_path()

        self.task_root_path = task_root_path
        self.data_pro_task_name = self.data_manager.get_full_task_name()

    def get_info_dic(self):
        """
        the info.json include the shared info about the dataset, like label density
        will be removed in the future release
        :return:
        """
        data_info = pars.ParameterDict()
        data_info.load_JSON(os.path.join(self.task_root_path, 'info.json'))
        return data_info['info']

    def get_data_loader(self):
        """
        get task related setttings for data manager
        """
        batch_size = self.task_opt['tsk_set'][('batch_sz', 1,'batch sz (only for mermaid related method, otherwise set to 1)')]
        is_train = self.task_opt['tsk_set'][('train',False,'train the model')]

        return self.data_manager.data_loaders(batch_size=batch_size,is_train=is_train)





    def setting_folder(self):
        for item in self.path:
            if not os.path.exists(self.path[item]):
                os.makedirs(self.path[item])


    
    def initialize_log_env(self,):
        """
        initialize log environment for the task.
        including
        task_path/checkpoints:  saved checkpoints for learning methods every # epoch
        task_path/logdir: saved logs for tensorboard
        task_path/records: saved 2d and 3d images for analysis
        :return: tensorboard writer
        """
        self.cur_task_path = os.path.join(self.task_root_path,self.task_name)
        logdir =os.path.join(self.cur_task_path,'log')
        check_point_path =os.path.join(self.cur_task_path,'checkpoints')
        record_path = os.path.join(self.cur_task_path,'records')
        model_path = self.task_opt['tsk_set'][('model_path', '', 'if continue_train, the model path should be given here')]
        self.task_opt['tsk_set'][('path',{},'record paths')]
        self.task_opt['tsk_set']['path']['expr_path'] =self.cur_task_path
        self.task_opt['tsk_set']['path']['logdir'] =logdir
        self.task_opt['tsk_set']['path']['check_point_path'] = check_point_path
        self.task_opt['tsk_set']['path']['model_load_path'] = model_path
        self.task_opt['tsk_set']['path']['record_path'] = record_path
        self.path = {'logdir':logdir,'check_point_path': check_point_path,'record_path':record_path}
        self.setting_folder()
        self.task_opt.write_ext_JSON(os.path.join(self.cur_task_path,'task_settings.json'))
        sys.stdout = self.Logger(self.cur_task_path)
        print('start logging:')
        self.writer = SummaryWriter(logdir, self.task_name)
        return self.writer