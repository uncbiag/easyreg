import sys,os
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../model_pool'))

import data_pre.module_parameters as pars
from abc import ABCMeta, abstractmethod
from model_pool.piplines import run_one_task
class BaseTask():
    __metaclass__ = ABCMeta
    def __init__(self,name):
        self.name = name

    @abstractmethod
    def save(self):
        pass

class DataTask(BaseTask):
    def __init__(self,name):
        super(DataTask,self).__init__(name)
        self.data_par = pars.ParameterDict()
        self.data_par.load_JSON('../settings/base_data_settings.json')


    def save(self):
        self.data_par.write_ext_JSON('../settings/data_settings.json')

class ModelTask(BaseTask):
    def __init__(self,name):
        super(ModelTask,self).__init__(name)
        self.task_par = pars.ParameterDict()
        self.task_par.load_JSON('../settings/base_task_settings.json')

    def save(self):
        self.task_par.write_ext_JSON('../settings/task_settings.json')


# #######################  Task 1  #########
# tsm = ModelTask('tsk1')
# tsm.task_par['tsk_set']['task_name'] = 'focal_loss_t1'
# tsm.task_par['tsk_set']['loss'] = 'focal_loss'
# tsm.task_par['tsk_set']['gpu_ids'] = 0
# tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,5,1]
# tsm.save()
# run_one_task()

# ################# Task 2  #############
# tsm = ModelTask('tsk2')
# tsm.task_par['tsk_set']['task_name'] = 'big_batch_t2'
# tsm.task_par['tsk_set']['loss'] = 'ce'
# tsm.task_par['tsk_set']['criticUpdates'] = 5
# tsm.task_par['tsk_set']['optim']['lr'] = 0.001/4
# tsm.save()
# run_one_task()


################# Task 3  #############
tsm = ModelTask('tsk3')

tsm.task_par['tsk_set']['task_name'] = 'big_batch_t3'
tsm.task_par['tsk_set']['model'] = 'unet'
tsm.task_par['tsk_set']['loss'] = 'focal_loss'
tsm.task_par['tsk_set']['criticUpdates'] = 5
tsm.task_par['tsk_set']['gpu_ids'] = 0
tsm.task_par['tsk_set']['max_batch_num_per_epoch'] = [200,5,1]
tsm.task_par['tsk_set']['optim']['lr'] = 0.001/4

tsm.save()
run_one_task()

############## Task 4 #############


