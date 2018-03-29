from data_pre.reg_data_utils import *
from torchvision import transforms
import torch
import data_pre.reg_data_loader as  reg_loader
from data_pre.seg_data_loader import ToTensor
import data_pre.module_parameters as pars
import  data_pre.reg_data_pool as reg_pool
import data_pre.seg_data_loader as  seg_loader
import data_pre.seg_data_loader_online as seg_loader_ol
import data_pre.seg_data_loader_online_old as seg_loader_ol_old
import data_pre.seg_data_loader_offline as seg_loader_fl
import  data_pre.seg_data_pool as seg_pool



class DataManager(object):
    def __init__(self, task_name, dataset_name):
        """
        the class for easy data management
        including two part: 1. preprocess data   2. set dataloader
        1. preprocess data,  currently support lpba, ibsr, oasis2d, cumc,
           the data will be saved in output_path/auto_generated_name/train|val|test
           files are named by the pair_name, source_target.h5py
        2. dataloader, pytorch multi-thread dataloader
            return the results in dic

        :param task_name: the name can help recognize the data
        :param dataset_name: the name of the the dataset
        :param sched: 'inter' or 'intra',  sched is can be used only when the dataset includes interpersonal and intrapersonal results,

        """
        self.task_name = task_name
        """name for easy recognition"""
        self.task_type = None
        """" the type of task 'reg','seg','mixed(to implement)'"""
        self.full_task_name = None
        """name of the output folder"""
        self.dataset_name = dataset_name
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
        self.full_comb = False     #e.g [1,2,3,4] True:[1,2],[2,3],[3,4],[1,3],[1,4],[2,4]  False: [1,2],[2,3],[3,4]
        """use all possible combination"""
        self.divided_ratio = [0.7,0.1,0.2]
        """divided data into train, val, and test sets"""
        self.slicing = -1
        """get the  nth slice(depend on image size) from the 3d volume"""
        self.axis = -1
        """slice the nth axis(1,2,3) of the 3d volume"""
        self.dataset = None
        self.task_path =None
        """train|val|test: dic task_root_path/train|val|test"""
        self.transform_seq = []
        self.seg_option = None


    def set_task_type(self,task_type):
        self.task_type = task_type

    def set_data_path(self, data_path):
        self.data_path = data_path

    def set_output_path(self,output_path):
        self.output_path = output_path

    def set_label_path(self, label_path):
        self.label_path = label_path

    def set_transform_seq(self,transform_seq):
        self.transform_seq = transform_seq

    def set_sched(self,sched):
        self.sched = sched

    def set_slicing(self, slicing, axis):
        self.slicing = slicing
        self.axis = axis

    def set_full_comb(self, full_comb):
        self.full_comb = full_comb

    def set_divided_ratio(self,divided_ratio):
        self.divided_ratio = divided_ratio

    def set_full_task_name(self, full_task_name):
        self.full_task_name = full_task_name

    def set_seg_option(self,option):
        self.seg_option = option
    def get_data_path(self):
        return self.data_path

    def get_full_task_name(self):
        return os.path.split(self.task_root_path)[1]

    def get_default_dataset_path(self,is_label):
        default_data_path = {'lpba':'/playpen/data/quicksilver_data/testdata/LPBA40/brain_affine_icbm_hist_oasis',
                             'oasis2d': '/playpen/zyshen/data/oasis',
                             'cumc':'/playpen/data/quicksilver_data/testdata/CUMC12/brain_affine_icbm',
                             'ibsr': '/playpen/data/quicksilver_data/testdata/IBSR18/brain_affine_icbm',
                             'oai':'/playpen/zyshen/unet/data/OAI_segmentation/Nifti_corrected_rescaled',
                             'brats':'/playpen/zyshen/data/miccia_brats'}

        default_label_path = {'lpba': '/playpen/data/quicksilver_data/testdata/LPBA40/label_affine_icbm',
                             'oasis2d': 'None',
                             'cumc': '/playpen/data/quicksilver_data/testdata/CUMC12/label_affine_icbm',
                             'ibsr': '/playpen/data/quicksilver_data/testdata/IBSR18/label_affine_icbm',
                             'brats': ''}
        if is_label:
            return default_label_path[self.dataset_name]
        else:
            return default_data_path[self.dataset_name]

    def get_default_seg_option(self):
        default_setting_path = {'oai': './settings/oai.json',
                               'lpba': './settings/lpba.json',
                               'brats': './settings/brats.json',
                               'cumc': './settings/cumc.json',
                               'ibsr': './settings/ibsr.json',}

        return default_setting_path[self.dataset_name]




    def generate_saving_path(self):
        slicing_info = '_slicing_{}_axis_{}'.format(self.slicing, self.axis) if self.slicing>0 else ''
        comb_info = '_full_comb' if self.full_comb else ''
        reg_info = slicing_info+comb_info
        transfrom_info=''
        from functools import reduce
        if len(self.transform_seq):
            transfrom_info = reduce((lambda x,y: x+y),self.transform_seq)
        extend_info = reg_info if self.task_type=='reg' else transfrom_info
        full_task_name = self.task_name+'_'+self.dataset_name+ '_'+ self.task_type+'_'+self.sched+extend_info
        self.set_full_task_name(full_task_name)
        self.task_root_path = os.path.join(self.output_path,full_task_name)

    def generate_task_path(self):
        self.task_path = {x:os.path.join(self.task_root_path,x) for x in ['train','val', 'test','debug']}
        return self.task_path

    def get_task_root_path(self):
        return self.task_root_path

    def manual_set_task_root_path(self, task_root_path):
        """
        switch the task into existed task, this is the only setting need to be set
        :param task_path: given existed task_root_path
        :return:
        """
        self.task_root_path = task_root_path
        self.task_path = {x:os.path.join(task_root_path,x) for x in ['train','val', 'test','debug']}
        return self.task_path

    def init_dataset(self):
        if self.task_type == 'reg':
            self.init_reg_dataset()
        elif self.task_type == 'seg':
            self.init_seg_dataset()
        else:
            raise(ValueError,"not implemented")



    def init_reg_dataset(self):
        if self.data_path is None:
            self.data_path = self.get_default_dataset_path(is_label=False)
        if self.label_path is None:
            self.label_path = self.get_default_dataset_path(is_label=True)

        self.dataset = reg_pool.RegDatasetPool().create_dataset(self.dataset_name,self.sched, self.full_comb)
        if 'set_slicing' in dir(self.dataset):
            self.dataset.set_slicing(self.slicing, self.axis)
        if 'set_label_path' in dir(self.dataset):
            self.dataset.set_label_path(self.label_path)

        self.dataset.set_data_path(self.data_path)
        self.dataset.set_output_path(self.task_root_path)
        self.dataset.set_divided_ratio(self.divided_ratio)




    def init_seg_dataset(self):
        if self.data_path is None:
            self.data_path = self.get_default_dataset_path(is_label=False)
        if self.label_path is None:
            self.label_path = self.get_default_dataset_path(is_label=True)
        if self.seg_option is None:
            option = pars.ParameterDict()
        else:
            option = self.seg_option
        self.dataset = seg_pool.SegDatasetPool().create_dataset(self.dataset_name,option, self.sched)
        self.dataset.set_label_path(self.label_path)

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
        self.cur_dataset = reg_loader.RegistrationDataset if self.task_type=='reg' else seg_loader_ol.SegmentationDataset

    def init_dataset_loader(self,transformed_dataset,batch_size):
        if self.task_type=='reg':
            dataloaders = {x: torch.utils.data.DataLoader(transformed_dataset[x], batch_size=batch_size,
                                                      shuffle=False, num_workers=4) for x in ['train','val','test','debug']}
        elif self.task_type=='seg':
            dataloaders = {'train':   torch.utils.data.DataLoader(transformed_dataset['train'],
                                                                  batch_size=batch_size,shuffle=True, num_workers=8),
                           'val': torch.utils.data.DataLoader(transformed_dataset['val'],
                                                                batch_size=1, shuffle=False, num_workers=1),
                           'test': torch.utils.data.DataLoader(transformed_dataset['test'],
                                                                batch_size=1, shuffle=False, num_workers=1),
                           'debug': torch.utils.data.DataLoader(transformed_dataset['debug'],
                                                               batch_size=1, shuffle=False, num_workers=1)
                           }
        elif self.task_type=='seg_pq':
            dataloaders={}
        return dataloaders


    def data_loaders(self, batch_size=20):
        """
        pytorch dataloader
        :param batch_size:  set the batch size
        :return:
        """
        phases = ['train', 'val','test','debug']
        composed = transforms.Compose([ToTensor()])
        self.init_dataset_type()
        transformed_dataset = {x: self.cur_dataset(data_path=self.task_path[x],phase=x,transform=composed,option=self.seg_option) for x in phases}
        dataloaders = self.init_dataset_loader(transformed_dataset, batch_size)
        dataloaders['data_size'] = {x: len(dataloaders[x]) for x in phases}
        dataloaders['info'] = {x: transformed_dataset[x].name_list for x in phases}
        print('dataloader is ready')


        return dataloaders

# TODO: support generic path names here

if __name__ == "__main__":

    prepare_data = True

    task_path = '/playpen/zyshen/data/lpba__slicing90'
    task_type = 'seg'

    dataset_name = 'lpba'
    task_name = 'debugging'
    full_comb = False
    output_path = '/playpen/zyshen/data/'
    divided_ratio = (0.6, 0.2, 0.2)
    slicing = -1
    sched ='patched'
    axis = 1
    switch_to_exist_task = False
    prepare_data = True

    if switch_to_exist_task:
        data_manager = DataManager(task_name=task_name, dataset_name=dataset_name)
        data_manager.set_task_type(task_type)
        data_manager.manual_set_task_root_path(task_path)
    else:

        data_manager = DataManager(task_name=task_name, dataset_name=dataset_name)
        data_manager.set_task_type(task_type)
        data_manager.set_sched(sched)
        data_manager.set_output_path(output_path)
        data_manager.set_full_comb(full_comb)
        data_manager.set_slicing(slicing, axis)
        data_manager.set_divided_ratio(divided_ratio)
        data_manager.generate_saving_path()
        data_manager.generate_task_path()

        data_manager.init_dataset()
        if prepare_data:
            data_manager.prepare_data()



    dataloaders = data_manager.data_loaders(batch_size=3)
    for data in dataloaders['test']:
        pass



