from data.data_utils import *
from os.path import join
import torch
from data.dataset import  *

def prepare_data(save_path, img_type, path='./data',skip=True, sched='intra'):
    '''
    default:
    path: './data'
    img_type: '*a.mhd'
    skip: True

     '''
    pair_list = list_pairwise(path, img_type, skip,sched)
    img_pair_list, info = load_as_data(pair_list)
    save_to_h5py(save_path, img_pair_list, info)



class DataManager(object):
    def __init__(self, sched='intra'):
        self.train_data_path = '/home/hbg/cs_courses/2d_data_code/data/train'
        self.val_data_path = '/home/hbg/cs_courses/2d_data_code/data/val'
        self.test_data_path = '/home/hbg/cs_courses/2d_data_code/data/test'
        self.raw_data_path=[self.train_data_path,self.val_data_path,self.test_data_path]
        self.sched = sched


        self.skip = True  # only in intra, True: (6month, 9 month)  (6month 12month)  (9 month 12month) False: (6 month 9 month) (9month 12month)
        if sched == 'intra':
            self.raw_img_type= ['*a.mhd']
        elif sched == 'inter':
            self.raw_img_type = ['*_1_a.mhd', '*_2_a.mhd', '*_3_a.mhd']
        else:
            raise ValueError('the sampling schedule should be intra or inter')
        self.train_h5_path = '../data/train_data_' + sched + '.h5py'
        self.val_h5_path = '../data/val_data_' + sched + '.h5py'
        self.test_h5_path = '../data/test_data_' + sched + '.h5py'
        self.save_h5_path = [self.train_h5_path,self.val_h5_path,self.test_h5_path]


    def prepare_data(self):
        for idx, path in enumerate(self.raw_data_path):
            prepare_data(self.save_h5_path[idx], self.raw_img_type, self.raw_data_path[idx], sched=self.sched)
            dic = read_file(self.save_h5_path[idx])
            print('data saved: {}'.format(self.save_h5_path[idx]))
            print(dic['info'])
            print(dic['data'].shape)
            print('finished')


    def dataloaders(self, batch_size=20):
        train_data_path = self.train_h5_path
        val_data_path = self.val_h5_path
        test_data_path= self.test_h5_path
        composed = transforms.Compose([ToTensor()])
        sess_sel = {'train': train_data_path, 'val': val_data_path, 'test': test_data_path}
        transformed_dataset = {x: RegistrationDataset(data_dir=sess_sel[x], transform=composed) for x in sess_sel}
        dataloaders = {x: torch.utils.data.DataLoader(transformed_dataset[x], batch_size=batch_size,
                                                shuffle=True, num_workers=4) for x in sess_sel}
        dataloaders['data_size'] = {x: len(transformed_dataset[x]) for x in ['train', 'val']}
        dataloaders['info'] = transformed_dataset['train'].info

        return dataloaders




if __name__ == "__main__":

    path = '/home/hbg/cs_courses/2d_data_code/data'
    img_type = ['*a.mhd']
    skip = True
    sched = 'intra'
    save_path = '../data/data_'+ sched +'.h5py'


    prepare_data(save_path, img_type, path, skip, sched)
    dic= read_file(save_path)
    print(dic['info'])
    print(dic['data'].shape)
    print('finished')

    img_type = ['*_1_a.mhd', '*_2_a.mhd', '*_3_a.mhd']
    sched='inter'
    save_path = '../data/data_' + sched + '.h5py'
    prepare_data(save_path, img_type, path, skip, sched)
    dic = read_file(save_path)
    print(dic['info'])
    print(dic['data'].shape)
    print('finished')

