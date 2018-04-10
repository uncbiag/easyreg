import os
import sys
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../model_pool'))


from model_pool.initializer import Initializer
from model_pool.create_model import create_model
from model_pool.train_expr import train_model
from model_pool.test_expr import test_model
from model_pool.debug_expr import debug_model


class Pipline():
    def initialize(self):
        initializer = Initializer()
        initializer.initialize_data_manager()
        self.tsk_opt = initializer.init_task_option()
        self.writer = initializer.initialize_log_env()
        self.tsk_opt = initializer.get_task_option()
        self.data_loaders = initializer.get_data_loader()
        self.model = create_model(self.tsk_opt)

    def run_task(self):
        if self.tsk_opt['tsk_set']['train']:
            train_model(self.tsk_opt, self.model, self.data_loaders,self.writer)

        else:
            debug_model(self.tsk_opt, self.model, self.data_loaders)
            #test_model(self.tsk_opt,self.model,self.data_loaders)
        # test_model(self.tsk_opt,self.model,self.data_loaders)
        # else:
        #     test_expr(self.tsk_opt, self.data_loaders)


def run_one_task():
    pipline = Pipline()
    pipline.initialize()
    pipline.run_task()


if __name__ == '__main__':
    pipline= Pipline()
    pipline.initialize()
    pipline.run_task()



