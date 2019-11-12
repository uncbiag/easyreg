from .initializer import Initializer
from .create_model import create_model
from .train_expr import train_model
from .test_expr import test_model


class Pipline():
    """
    Pipeline class,
    initialize  env : data_manager, log settings and task settings
    run_task : run training based model or evaluation based model
    """
    def initialize(self,task_setting_pth='../settings/task_settings.json',data_setting_pth='../settings/data_settings.json',data_loaders=None):
        """
        initialize task environment
        :param task_setting_pth: the path of current task setting file
        :param data_setting_pth:  the path of current data processing setting file (option if already set in task_setting_file)
        :param data_loaders: the dataloader for tasks
        :return: None
        """
        initializer = Initializer()
        initializer.initialize_data_manager(data_setting_pth)
        self.task_setting_pth = task_setting_pth
        self.tsk_opt = initializer.init_task_option(task_setting_pth)
        self.writer = initializer.initialize_log_env()
        self.tsk_opt = initializer.get_task_option()
        self.data_loaders = initializer.get_data_loader() if data_loaders is None else data_loaders
        self.model = create_model(self.tsk_opt)

    def clean_up(self):
        """
        clean the environment settings, but keep the dataloader
        :return: None
        """
        self.tsk_opt = None
        self.writer  = None
        self.model = None

    def run_task(self):
        """
        run training based model or evaluation based model
        :return: None
        """
        if self.tsk_opt['tsk_set']['train']:
            train_model(self.tsk_opt, self.model, self.data_loaders,self.writer)

        else:
            from easyreg.compare_sym import cal_sym
            test_model(self.tsk_opt, self.model, self.data_loaders)
            #cal_sym(self.tsk_opt,self.data_loaders)
        saving_comment_path = self.task_setting_pth.replace('.json','_comment.json')
        self.tsk_opt.write_JSON_comments(saving_comment_path)



def run_one_task(task_setting_pth='../settings/task_settings.json',data_setting_pth='../settings/data_settings.json',data_loaders=None):
    pipline = Pipline()
    pipline.initialize(task_setting_pth,data_setting_pth,data_loaders)
    pipline.run_task()
    return pipline


if __name__ == '__main__':
    pipline= Pipline()
    pipline.initialize()
    pipline.run_task()



