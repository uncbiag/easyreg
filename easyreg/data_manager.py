from easyreg.reg_data_utils import *
from torchvision import transforms
import torch
from easyreg import reg_data_loader_onfly as reg_loader_of
from easyreg import seg_data_loader_onfly as seg_loader_of
from easyreg.reg_data_loader_onfly import ToTensor
# todo reformat the import style
class DataManager(object):
    def __init__(self, task_name, dataset_name):
        """
        the class for data management
        including two part: 1. preprocess data (disabled)   2. set dataloader
        todo the preprocess data is disabled for the current version, so the data should be prepared ahead.
        1. preprocess data,  currently support lpba, ibsr, oasis2d, cumc, oai
           the data will be saved in output_path/auto_generated_name/train|val|test|debug
        2. dataloader, pytorch multi-thread dataloader
            return a dict, each train/val/test/debug phase has its own dataloader
        The path is organized as  output_path/task_name with regard to data processing/ train|val|test|debug

        :param task_name: the name can help recognize the data
        :param dataset_name: the name of the the dataset
        :param sched: 'inter' or 'intra',  sched is can be used only when the dataset includes interpersonal and intrapersonal results

        """
        self.task_name = task_name
        """name for task"""
        self.task_type = None
        """" the type of task 'reg','seg'(disabled)'"""
        self.full_task_name = None
        """name of the output folder"""
        self.dataset_name = dataset_name
        """ name of the dataset i.e. lpba, ibsr, oasis """
        self.sched = ''
        """reg: inter, intra    seg:'patched' 'nopatched'"""
        self.data_path = None
        """path of the dataset"""
        self.output_path= None
        """path of the processed data"""
        self.task_root_path = None
        """output_path/full_task_name, or can be set manual"""
        self.label_path = None
        """path of the labels of the dataset"""
        self.divided_ratio = [0.7,0.1,0.2]
        """divided data into train, val, and test sets"""
        self.dataset = None
        self.task_path =None
        """train|val|test: dic task_root_path/train|val|test"""
        self.seg_option = None
        """ settings for seg task"""
        self.reg_option = None
        """ settings for reg task"""
        self.phases =['train','val','test','debug']
        """ phases, the debug here refers to the subtraining set to check overfitting"""


    def set_task_type(self,task_type):
        """ set task type, 'reg' or 'seg' """
        self.task_type = task_type

    def set_data_path(self, data_path):
        """ set data path, here refers to the folder path of dataset"""
        self.data_path = data_path

    def set_output_path(self,output_path):
        """ set the output path for propocessed data"""
        self.output_path = output_path

    def set_label_path(self, label_path):
        """set the label path, here refers to the folder path of labels"""
        self.label_path = label_path


    def set_sched(self,sched):
        """ set the sched, can be 'inter','intra'"""
        self.sched = sched

    def set_divided_ratio(self,divided_ratio):
        """ set the divide raito, the divide ratio of the dataset with regard to train, val and test"""
        self.divided_ratio = divided_ratio

    def set_full_task_name(self, full_task_name):
        """
        todo disabled for the cur version
        the task name that combined settings
        :param full_task_name:
        :return:
        """
        self.full_task_name = full_task_name

    def set_reg_option(self,option):
        """ set the registrion settings"""
        self.reg_option = option

    def set_seg_option(self,option):
        """ set the registrion settings"""
        self.seg_option = option

    def get_data_path(self):
        """ return the data path"""
        return self.data_path

    def get_full_task_name(self):
        """ return the full task name"""
        return os.path.split(self.task_root_path)[1]


    def generate_saving_path(self, auto=True):
        self.task_root_path =  os.path.join(self.output_path,self.task_name)

    def generate_task_path(self):
        """ the saving path for proprocessed data is output_path/task_name/train|val|test|debug"""
        self.task_path = {x:os.path.join(self.task_root_path,x) for x in ['train','val', 'test','debug']}
        return self.task_path

    def get_task_root_path(self):
        """ return the task root path, refers to output_path/task_name"""
        return self.task_root_path

    def manual_set_task_root_path(self, task_root_path):
        """
        if switch the task into existed task, this is the only setting need to be set
        :param task_path: given existed task_root_path
        :return:
        """
        self.task_root_path = task_root_path
        self.task_path = {x:os.path.join(task_root_path,x) for x in ['train','val', 'test','debug']}
        return self.task_path

    def init_dataset(self):
        """ preprocess the dataset"""
        if self.task_type == 'reg':
            self.init_reg_dataset()
        else:
            raise(ValueError,"not implemented")




    def init_reg_dataset(self):
        """ preprocess the registration dataset"""
        import data_pre.reg_data_pool as reg_pool
        self.dataset = reg_pool.RegDatasetPool().create_dataset(self.dataset_name,self.sched, self.full_comb)
        self.dataset.set_data_path(self.data_path)
        self.dataset.set_output_path(self.task_root_path)
        self.dataset.set_divided_ratio(self.divided_ratio)


    def prepare_data(self):
        """
        preprocess data into h5py
        :return:
        """
        self.dataset.prepare_data()

    def init_dataset_type(self):
        self.cur_dataset = reg_loader_of.RegistrationDataset if self.task_type=='reg' else seg_loader_of.SegmentationDataset

    def init_dataset_loader(self,transformed_dataset,batch_size):
        """
        initialize the data loaders: set work number, set work type( shuffle for trainning, order for others)
        :param transformed_dataset:
        :param batch_size: the batch size of each iteration
        :return: dict of dataloaders for train|val|test|debug
        """

        def _init_fn(worker_id):
            np.random.seed(12 + worker_id)
        num_workers_reg ={'train':16,'val':0,'test':0,'debug':0}#{'train':0,'val':0,'test':0,'debug':0}#{'train':8,'val':4,'test':4,'debug':4}
        shuffle_list ={'train':True,'val':False,'test':False,'debug':False}
        batch_size = [batch_size]*4 if not isinstance(batch_size, list) else batch_size
        batch_size = {'train': batch_size[0],'val':batch_size[1],'test':batch_size[2],'debug':batch_size[3]}
        dataloaders = {x: torch.utils.data.DataLoader(transformed_dataset[x], batch_size=batch_size[x],
                                                  shuffle=shuffle_list[x], num_workers=num_workers_reg[x],worker_init_fn=_init_fn) for x in self.phases}
        return dataloaders


    def data_loaders(self, batch_size=20,is_train=True):
        """
        get the data_loaders for the train phase and the test phase
        :param batch_size: the batch size for each iteration
        :param is_train: in train mode or not
        :return: dict of dataloaders for train phase or the test phase
        """
        if is_train:
            self.phases = ['train', 'val','debug']
        else:
            self.phases = ['test']
        composed = transforms.Compose([ToTensor()])
        self.init_dataset_type()
        option = self.seg_option if self.task_type=="seg" else self.reg_option
        transformed_dataset = {x: self.cur_dataset(data_path=self.task_path[x],phase=x,transform=composed,option=option) for x in self.phases}
        dataloaders = self.init_dataset_loader(transformed_dataset, batch_size)
        dataloaders['data_size'] = {x: len(dataloaders[x]) for x in self.phases}
        dataloaders['info'] = {x: transformed_dataset[x].name_list for x in self.phases}
        print('dataloader is ready')

        return dataloaders


if __name__ == "__main__":
    from tools.module_parameters import ParameterDict
    prepare_data = True

    task_root_path = '/home/zyshen/proj/local_debug/brain_seg'
    task_type = 'seg'
    dataset_name = 'lpba'
    task_name = 'debugging'
    settings = ParameterDict()
    settings.load_JSON('/home/zyshen/proj/easyreg/debug/settings/data_setting.json')
    seg_option = settings["datapro"]
    data_manager = DataManager(task_name, dataset_name)
    data_manager.set_task_type('seg')
    data_manager.manual_set_task_root_path(task_root_path)
    data_manager.generate_task_path()
    data_manager.seg_option = seg_option
    dataloaders = data_manager.data_loaders(batch_size=3)
    for data in dataloaders['train']:
        pass



