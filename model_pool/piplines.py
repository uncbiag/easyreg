from model_pool.initializer import Initializer
from pipLine.train_model_reg import train_model
from model_pool.create_model import create_model
from model_pool.train_expr import train_model


class pipline():
    def initialize(self):
        initializer = Initializer()
        initializer.initialize_data_manager(task_type='seg')
        self.tsk_opt = initializer.init_task_option()
        self.writer = initializer.initialize_log_env()
        self.tsk_opt = initializer.get_task_option()
        self.data_loaders = initializer.get_data_loader()
        self.model = create_model(self.tsk_opt)

    def run_task(self):
        if self.tsk_opt['tsk_set']['train']:
            train_model(self.tsk_opt, self.model, self.data_loaders,self.writer)
        else:
            test_expr(self.tsk_opt, self.data_loaders)


if __name__ == '__main__':
    pipline= pipline()
    pipline.initialize()
    pipline.run_task()




