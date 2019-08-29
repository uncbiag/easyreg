import os
import sys
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../model_pool'))


from model_pool.initializer import Initializer
from model_pool.create_model import create_model
from model_pool.train_expr import train_model
from model_pool.test_expr import test_model


class Pipline():
    def initialize(self,task_setting_pth='../settings/task_settings.json',data_setting_pth='../settings/data_settings.json',data_loaders=None):
        initializer = Initializer()
        initializer.initialize_data_manager(data_setting_pth)
        self.tsk_opt = initializer.init_task_option(task_setting_pth)
        self.writer = initializer.initialize_log_env()
        self.tsk_opt = initializer.get_task_option()
        self.data_loaders = initializer.get_data_loader() if data_loaders is None else data_loaders
        self.model = create_model(self.tsk_opt)

    def clean_up(self):
        self.tsk_opt = None
        self.writer  = None
        self.model = None

    def run_task(self):
        if self.tsk_opt['tsk_set']['train']:
            train_model(self.tsk_opt, self.model, self.data_loaders,self.writer)

        else:
            from model_pool.compare_symmetric import cal_sym
            test_model(self.tsk_opt, self.model, self.data_loaders)
            #cal_sym(self.tsk_opt,self.data_loaders)



def run_one_task(task_setting_pth='../settings/task_settings.json',data_setting_pth='../settings/data_settings.json',data_loaders=None):
    pipline = Pipline()
    pipline.initialize(task_setting_pth,data_setting_pth,data_loaders)
    pipline.run_task()
    return pipline


if __name__ == '__main__':
    pipline= Pipline()
    pipline.initialize()
    pipline.run_task()



