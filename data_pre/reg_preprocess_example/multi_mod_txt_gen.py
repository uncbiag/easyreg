import os
from easyreg.reg_data_utils import get_file_path_list,find_corr_map,str_concat
class MultiTxtGen(object):
    def __init__(self, mode_list, data_path, output_path, img_type='.nii',label_replace=''):
        self.mode_list = mode_list
        self.data_path = data_path
        self.output_path = output_path
        self.img_type = img_type
        self.label_replace = label_replace
        self.file_path_list_dic = {}
        self.ratio = (0.8, 0.05)
        self.sesses = ['train','val','test']
        self.file_num = -1
        self.divide_ind={}

    def set_divide_ratio(self,ratio):
        self.ratio =  ratio


    def get_file_list(self):
        mode_0 = self.mode_list[0]
        file_path_list_dic_for_check={}
        for mode in self.mode_list:
            img_type = ['*'+ mode+ self.img_type]
            file_path_list_dic_for_check[mode]=get_file_path_list(self.data_path,img_type)
        mode0_checked_list = self.check_file_list(file_path_list_dic_for_check)
        n_before_check = len(file_path_list_dic_for_check[mode_0])
        n_after_check  =len(mode0_checked_list)
        self.file_num = n_after_check
        print("total {} of {} file has been removed during checking".format(n_before_check-n_after_check,n_before_check))
        self.file_path_list_dic[mode_0] = mode0_checked_list
        for mode in self.mode_list[1:]:
            self.file_path_list_dic[mode] = [path.replace(mode_0,mode) for path in self.file_path_list_dic[mode_0]]

    def check_file_list(self,dic_to_check):
        mode_0 = self.mode_list[0]
        to_rm=[]
        for path in dic_to_check[mode_0]:
            for mode in self.mode_list[1:]:
                is_in= path.replace(mode_0,mode) in dic_to_check[mode]
                if not is_in:
                    print("Warning, file{} is not exist".format(path.replace(mode_0,mode)))
                    to_rm.append(path)
                    break
        # save rm
        list_checked = dic_to_check[mode_0]
        for path in to_rm:
            list_checked.remove(path)
        return list_checked

    def get_label_list(self):
        mode_0 = self.mode_list[0]
        label_list = find_corr_map(self.file_path_list_dic[mode_0], None, label_switch = (mode_0,self.label_replace))
        self.file_path_list_dic['label']=label_list

    def gen_divide_index(self):
        ind = list(range(self.file_num))
        import random
        random.shuffle(ind)
        train_num = int(self.ratio[0] * self.file_num)
        val_num = int(self.ratio[1] * self.file_num)
        test_num = self.file_num - train_num - val_num
        self.divide_ind['train'] = ind[:train_num]
        self.divide_ind['val'] = ind[train_num:train_num + val_num]
        self.divide_ind['test'] = ind[train_num + val_num:]


    def write_into_txt(self):
        for sess in self.sesses:
            with open(os.path.join(self.output_path,sess+'.txt'), 'w') as f:
                for ind in self.divide_ind[sess]:
                    mode_list = [self.file_path_list_dic[self.mode_list[0]][ind],self.file_path_list_dic['label'][ind]]
                    concat_info = str_concat(mode_list,linker=',')
                    f.write(concat_info+'\n')

    def gen_text(self):
        self.get_file_list()
        self.get_label_list()
        self.gen_divide_index()
        self.write_into_txt()






#
# # Brat dataset
# model_list = ['flair','t1','t1ce','t2']
# data_path = '/playpen/zyshen/data/MICCAI_BraTS17_Data_Training'
# output_path = '/playpen/zyshen/data/MICCAI_BraTS17_Data_Training/debug'
# brat = MultiTxtGen(mode_list=model_list, data_path=data_path, output_path=output_path, img_type='.nii.gz',label_replace='seg')
# brat.gen_text()
#



class MultiTxtGenByFolder(object):
    def __init__(self, mode_list, data_path_list, output_path, img_type='.nii',label_replace=''):
        self.mode_list = mode_list
        self.data_path_list = data_path_list
        self.output_path = output_path
        self.img_type = img_type
        self.label_replace = label_replace
        self.file_path_list_dic = {}
        self.sesses = ['train','val','test']
        self.ratio = (0.8,0.2)
        self.file_num = -1
        self.divide_ind={}


    def set_divide_ratio(self,ratio):
        """
        :param ratio: a tuple including ratio for train and val
        :return:
        """
        self.ratio =  ratio

    def set_divide_folder(self,folder_list):
        self.divide_folder_path = folder_list

    def get_file_list(self,data_path):
        mode_0 = self.mode_list[0]
        file_path_list_dic_for_check={}
        for mode in self.mode_list:
            img_type = ['*'+ mode+ self.img_type]
            file_path_list_dic_for_check[mode]=get_file_path_list(data_path,img_type)
        mode0_checked_list = self.check_file_list(file_path_list_dic_for_check)
        n_before_check = len(file_path_list_dic_for_check[mode_0])
        n_after_check  =len(mode0_checked_list)
        self.file_num = n_after_check
        print("total {} of {} file has been removed during checking".format(n_before_check-n_after_check,n_before_check))
        self.file_path_list_dic[mode_0] = mode0_checked_list
        for mode in self.mode_list[1:]:
            self.file_path_list_dic[mode] = [path.replace(mode_0,mode) for path in self.file_path_list_dic[mode_0]]

    def check_file_list(self,dic_to_check):
        mode_0 = self.mode_list[0]
        to_rm=[]
        for path in dic_to_check[mode_0]:
            for mode in self.mode_list[1:]:
                is_in= path.replace(mode_0,mode) in dic_to_check[mode]
                if not is_in:
                    print("Warning, file{} is not exist".format(path.replace(mode_0,mode)))
                    to_rm.append(path)
                    break
        # save rm
        list_checked = dic_to_check[mode_0]
        for path in to_rm:
            list_checked.remove(path)
        return list_checked

    def get_label_list(self):
        mode_0 = self.mode_list[0]
        label_list = find_corr_map(self.file_path_list_dic[mode_0], None, label_switch = (mode_0,self.label_replace))
        self.file_path_list_dic['label']=label_list

    def gen_divide_index(self):
        ind = list(range(self.file_num))
        import random
        random.shuffle(ind)
        train_num = int(self.ratio[0] * self.file_num)
        val_num = self.file_num - train_num
        self.divide_ind['train'] = ind[:train_num]
        self.divide_ind['val'] = ind[train_num:train_num + val_num]


    def gen_test_list(self,test_path):
        self.get_file_list(test_path)

    def write_test_into_txt(self):
        with open(os.path.join(self.output_path, 'test' + '.txt'), 'w') as f:
            for path in self.file_path_list_dic[self.mode_list[0]]:
                concat_info = path+',no_label'
                f.write(concat_info+'\n')

    def write_into_txt(self):
        for sess in ['train','val']:
            with open(os.path.join(self.output_path,sess+'.txt'), 'w') as f:
                for ind in self.divide_ind[sess]:
                    mode_list = [self.file_path_list_dic[self.mode_list[0]][ind],self.file_path_list_dic['label'][ind]]
                    concat_info = str_concat(mode_list,linker=',')
                    f.write(concat_info+'\n')

    def gen_text(self):
        self.get_file_list(self.data_path_list[0])
        self.get_label_list()
        self.gen_divide_index()
        self.write_into_txt()
        self.gen_test_list(self.data_path_list[1])
        self.write_test_into_txt()

# Brat dataset
model_list = ['flair','t1','t1ce','t2']
data_path = ['/playpen/raid/zyshen/data/miccia_brats/train','/playpen/raid/zyshen/data/miccia_brats/test']
output_path_list = '/playpen/raid/zyshen/data/miccia_brats'
brat = MultiTxtGenByFolder(mode_list=model_list, data_path_list=data_path, output_path=output_path_list, img_type='.nii.gz',label_replace='seg')
brat.gen_text()