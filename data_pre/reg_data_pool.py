from __future__ import print_function
import progressbar as pb

from torch.utils.data import Dataset

from data_pre.reg_data_utils import *

from data_pre.oasis_longitude_reg import *
import copy

sesses = ['train', 'val', 'test', 'debug']
number_of_workers = 10
warning_once = True

class BaseRegDataSet(object):

    def __init__(self, dataset_type, file_type_list, sched=None):
        """
        :param name: name of data set
        :param dataset_type: ''mixed' like oasis including inter and  intra person  or 'custom' like LPBA40, only includes inter person
        :param file_type_list: the file types to be filtered, like [*1_a.bmp, *2_a.bmp]
        :param data_path: path of the dataset
        """

        self.data_path = None
        """path of the dataset"""
        self.output_path = None
        """path of the output directory"""
        self.pro_data_path = None
        self.pair_name_list = []
        self.pair_path_list = []
        self.file_type_list = file_type_list
        self.save_format = 'h5py'
        """currently only support h5py"""
        self.sched = sched
        """inter or intra, for inter-personal or intra-personal registration"""
        self.dataset_type = dataset_type
        self.saving_h5py=False
        """custom or mixed"""
        self.normalize= False
        """ settings for normalization, currently not used"""
        self.divided_ratio = (0.7, 0.1, 0.2)
        """divided the data into train, val, test set"""

    def generate_pair_list(self):
        pass


    def set_data_path(self, path):
        self.data_path = path

    def set_output_path(self, path):
        self.output_path = path
        make_dir(path)

    def set_normalize_sched(self,sched):
        self.normalize_sched = sched

    def set_divided_ratio(self,ratio):
        self.divided_ratio = ratio

    def get_file_num(self):
        return len(self.pair_path_list)

    def get_pair_name_list(self):
        return self.pair_name_list

    def read_file(self, file_path, is_label=False):
        """
        currently, default using file_io, reading medical format
        :param file_path:
        :param  is_label: the file_path is label_file
        :return:
        """
        try:
            img, info = file_io_read_img(file_path,normalize_intensities=self.normalize, is_label=is_label)
        except:
            global warning_once
            if warning_once:
                print("not default file reading function is used, no normalization is applied")
                warning_once = False
            img = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
            info =None
        return img, info

    def extract_pair_info(self, info1, info2):
        return info1

    def save_shared_info(self,info):
        save_sz_sp_to_json(info, self.output_path)

    def save_pair_to_txt(self,info=None):
        pass
    def save_pair_to_h5py(self,info=None):
        pass
    def gen_pair_dic(self):
        pass

    def save_file_to_h5py(self,info=None):
        pass

    def initialize_info(self):
        pass



    def prepare_data(self):
        """
        preprocessig  data for each dataset
        :return:
        """
        print("starting preapare data..........")
        print("the output file path is: {}".format(self.output_path))
        self.initialize_info()

        info=self.gen_pair_dic()
        self.save_pair_to_txt(copy.deepcopy(info))
        if self.saving_h5py:
            self.save_pair_to_h5py(copy.deepcopy(info))
        try:
            print("the total num of pair is {}".format(self.get_file_num()))
        except:
            pass
        print("data preprocessing finished")



class UnlabeledDataSet(BaseRegDataSet):
    """
    This class only compatible with dataset_type "mixed" and "custom"
    unlabeled dataset
    """
    def __init__(self,dataset_type, file_type_list, sched=None):
        BaseRegDataSet.__init__(self,dataset_type, file_type_list,sched)

    def save_pair_to_txt(self, info=None):
        """
        save the file into h5py
        :param pair_path_list: N*2  [[full_path_img1, full_path_img2],[full_path_img2, full_path_img3]
        :param pair_name_list: N*1 for 'mix': [patientName1_volumeName1_patientName2_volumeName2, .....]  for custom: [volumeName1_volumeName2, .....]
        :param ratio:  divide dataset into training val and test, based on ratio, e.g [0.7, 0.1, 0.2]
        :param saving_path_list: N*1 list of path for output files e.g [ouput_path/train/volumeName1_volumeName2.h5py,.........]
        :param info: dic including pair information
        :param normalized_sched: normalized the image
        """
        random.shuffle(self.pair_path_list)
        self.pair_name_list = generate_pair_name(self.pair_path_list, sched=self.dataset_type)
        self.num_pair = len(self.pair_path_list)
        sub_folder_dic, file_id_dic = divide_data_set(self.output_path, self.num_pair, self.divided_ratio)
        divided_path_and_name_dic = get_divided_dic(file_id_dic, self.pair_path_list, self.pair_name_list)
        saving_pair_info(sub_folder_dic,divided_path_and_name_dic)



    def save_file_to_h5py(self,info=None):
        file_path_list =  get_file_path_list(self.data_path, self.file_type_list)
        file_name_list = [get_file_name(pt) for pt in file_path_list]
        self.pro_data_path = os.path.join(self.output_path,'data')
        pro_data_path_list = [fp.replace(self.data_path,self.pro_data_path) for fp in file_path_list]
        self.pro_data_path_list =[os.path.join(os.path.split(fp)[0],file_name_list[idx]+'.h5py') for idx, fp in enumerate(pro_data_path_list)]
        make_dir(self.pro_data_path)
        img_size = ()
        info = None
        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(file_path_list)).start()
        for i, file in enumerate(file_path_list):
            img, info = self.read_file(file)
            f_name = file_name_list[i]
            if i == 0:
                img_size = img.shape
            else:
                check_same_size(img, img_size)
            saving_path = self.pro_data_path_list[i]
            make_dir(os.path.split(saving_path)[0])
            save_to_h5py(saving_path, img, None, f_name, info, verbose=False)
            pbar.update(i + 1)
        pbar.finish()




class LabeledDataSet(BaseRegDataSet):
    """
    This class only compatible with dataset_type "mixed" and "custom"

    labeled dataset
    """
    def __init__(self, dataset_type, file_type_list, sched=None):
        BaseRegDataSet.__init__(self,dataset_type, file_type_list,sched)
        self.label_switch = ('','')
        self.label_path = None
        self.pair_label_path_list=[]


    def set_label_switch(self,label_switch):
        self.label_switch = label_switch


    def set_label_path(self, path):
        self.label_path = path

    def convert_to_standard_label_map(self,label_map,file_path):

        cur_label_list = list(np.unique(label_map))
        num_label = len(cur_label_list)
        if self.num_label != num_label:  # s37 in lpba40 has one more label than others
            print("Warnning!!!!, The num of classes {} are not the same in file{}".format(num_label, file_path))

        for l_id in cur_label_list:
            if l_id in self.standard_label_index:
                st_index = self.standard_label_index.index(l_id)
            else:
                # assume background label is 0
                st_index = 0
                print("warning label: is not in standard label index, and would be convert to 0".format(l_id))
            label_map[np.where(label_map==l_id)]=st_index

    def initialize_info(self):
        file_path_list = get_file_path_list(self.data_path, self.file_type_list)
        file_label_path_list = find_corr_map(file_path_list, self.label_path,self.label_switch)
        label, linfo = self.read_file(file_label_path_list[0], is_label=True)
        label_list = list(np.unique(label))
        num_label = len(label_list)
        self.standard_label_index = tuple([int(item) for item in label_list])
        print('the standard label index is :{}'.format(self.standard_label_index))
        print('the num of the class: {}'.format(num_label))
        self.num_label = num_label
        linfo['num_label'] = num_label
        linfo['standard_label_index']= self.standard_label_index
        self.save_shared_info(linfo)


    def save_pair_to_txt(self):
        """
        save the file into h5py
        :param pair_path_list: N*2  [[full_path_img1, full_path_img2],[full_path_img2, full_path_img3]
        :param pair_name_list: N*1 for 'mix': [patientName1_volumeName1_patientName2_volumeName2, .....]  for custom: [volumeName1_volumeName2, .....]
        :param ratio:  divide dataset into training val and test, based on ratio, e.g [0.7, 0.1, 0.2]
        :param saving_path_list: N*1 list of path for output files e.g [ouput_path/train/volumeName1_volumeName2.h5py,.........]
        :param info: dic including pair information
        :param normalized_sched: normalized the image
        """
        random.shuffle(self.pair_path_list)
        self.pair_name_list = generate_pair_name(self.pair_path_list, sched=self.dataset_type)
        self.num_pair = len(self.pair_path_list)
        sub_folder_dic, file_id_dic = divide_data_set(self.output_path, self.num_pair, self.divided_ratio)
        divided_path_and_name_dic = get_divided_dic(file_id_dic, self.pair_path_list, self.pair_name_list)
        saving_pair_info(sub_folder_dic,divided_path_and_name_dic)



    def save_file_to_h5py(self=None):
        file_path_list =  get_file_path_list(self.data_path, self.file_type_list)
        file_label_path_list = find_corr_map(file_path_list, self.label_path,self.label_switch)
        file_name_list = [get_file_name(pt) for pt in file_path_list]
        self.pro_data_path = os.path.join(self.output_path,'data')
        pro_data_path_list = [fp.replace(self.data_path, self.pro_data_path) for fp in file_path_list]
        self.pro_data_path_list = [os.path.join(os.path.split(fp)[0], file_name_list[idx] + '.h5py') for idx, fp in
                                   enumerate(pro_data_path_list)]
        make_dir(self.pro_data_path)
        img_size = ()
        info = None
        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(file_path_list)).start()
        for i, file in enumerate(file_path_list):
            img, info = self.read_file(file)
            f_name = file_name_list[i]
            label, linfo = self.read_file(file_label_path_list[i], is_label=True)
            self.convert_to_standard_label_map(label, file_label_path_list[i])
            if i == 0:
                img_size = img.shape
                check_same_size(label, img_size)
            else:
                check_same_size(img, img_size)
                check_same_size(label, img_size)
            saving_path = self.pro_data_path_list[i]
            make_dir(os.path.split(saving_path)[0])
            save_to_h5py(saving_path, img, label, f_name, info, verbose=False)
            pbar.update(i + 1)
        pbar.finish()



    def save_file_by_default_type(self):
        file_path_list =  get_file_path_list(self.data_path, self.file_type_list)
        file_label_path_list = find_corr_map(file_path_list, self.label_path,self.label_switch)
        file_name_list = [get_file_name(pt) for pt in file_path_list]
        self.pro_data_path = os.path.join(self.output_path,'data')
        pro_data_path_list = [fp.replace(self.data_path, self.pro_data_path) for fp in file_path_list]
        self.pro_data_path_list = pro_data_path_list
        make_dir(self.pro_data_path)






class CustomDataSet(BaseRegDataSet):
    """
    dataset format that orgnized as data_path/slice1, slic2, slice3 .......
    """
    def __init__(self,dataset_type, file_type_list,full_comb=False):
        BaseRegDataSet.__init__(self, dataset_type, file_type_list)
        self.full_comb = full_comb


    def generate_pair_list(self, sched=None):
        """
        :param sched:
        :return:
        """
        pair_path_list = inter_pair(self.pro_data_path,  ['*.h5py'], self.full_comb, mirrored=True)
        return pair_path_list


class VolumetricDataSet(BaseRegDataSet):
    """
    3D dataset
    """
    def __init__(self,dataset_type, file_type_list,sched=None):
        BaseRegDataSet.__init__(self, dataset_type=dataset_type, file_type_list=file_type_list,sched=sched)
        self.slicing = -1
        self.axis = -1

    def set_slicing(self, slicing, axis):
        if slicing >0 and axis>0:
            print("slcing is set on , the slice of {} th dimension would be sliced ".format(slicing))
        self.slicing = slicing
        self.axis = axis


    def read_file(self, file_path, is_label=False, verbose=False):
        """

        :param file_path:the path of the file
        :param is_label:the file is label file
        :param verbose:
        :return:
        """
        if self.slicing != -1:
            if verbose:
                print("slicing file: {}".format(file_path))
            img, info = file_io_read_img_slice(file_path, self.slicing, self.axis, is_label=is_label)
        else:
            img, info= BaseRegDataSet.read_file(self,file_path,is_label=is_label)
        return img, info




class MixedDataSet(BaseRegDataSet):
    """
     include inter-personal and intra-personal data, which is orgnized as oasis2d, root/patient1_folder/slice1,slice2...
    """
    def __init__(self, dataset_type, file_type_list, sched, full_comb=False):
        BaseRegDataSet.__init__(self, dataset_type, file_type_list, sched=sched)
        self.full_comb = full_comb


    def generate_pair_list(self):
        """
         return the list of  paths of the paired image  [N,2]
        :param file_type_list: filter and get the image of certain type
        :param full_comb: if full_comb, return all possible pairs, if not, return pairs in increasing order
        :param sched: sched can be inter personal or intra personal
        :return:
        """
        if self.sched == 'intra':
            dic_list = list_dic(self.data_path)
            pair_path_list = intra_pair(self.pro_data_path, dic_list, ['*.h5py'], self.full_comb, mirrored=True)
        elif self.sched == 'inter':
            pair_path_list = inter_pair(self.pro_data_path,  ['*.h5py'], self.full_comb, mirrored=True)
        else:
            raise ValueError("schedule should be 'inter' or 'intra'")

        return pair_path_list





class PatientStructureDataSet(VolumetricDataSet):
    """

    The data in self.data_root_path would be loaded,
    """

    def __init__(self, dataset_type, file_type_list, sched):
        VolumetricDataSet.__init__(self, dataset_type, file_type_list,sched)
        self.patients = []
        self.only_test_set=False
        """all available data would be regarded as test data, no training and validation set would be generated"""
        self.data_root_path = None

    def initialize_info(self):
        self.__init_patients()


    def set_data_root_path(self,data_root_path):
        self.data_root_path = data_root_path

    def __init_patients(self):
        if self.data_root_path is None:
            if not self.only_test_set:
                root_path ="/playpen/zyshen/summer/oai_registration/reg_0623/data"
            else:
                root_path ="/playpen/zyshen/summer/oai_registration/reg_0820/data"
        else:
            root_path = self.data_root_path

        Patient_class = Patients(full_init=True, root_path=root_path)
        self.patients= Patient_class.get_filtered_patients_list(has_complete_label=True, len_time_range=[1, 10], use_random=False)
        print("total {} of  paitents are selected".format(len(self.patients)))

    def __divide_into_train_val_test_set(self,root_path, patients, ratio):
        num_patient = len(patients)
        train_ratio = ratio[0]
        val_ratio = ratio[1]
        sub_path = {x: os.path.join(root_path, x) for x in ['train', 'val', 'test', 'debug']}
        nt = [make_dir(sub_path[key]) for key in sub_path]
        if sum(nt):
            raise ValueError("the data has already exist, due to randomly assignment schedule, the program block\n"
                             "manually delete the folder to reprepare the data")

        train_num = int(train_ratio * num_patient)
        val_num = int(val_ratio * num_patient)
        sub_patients_dic = {}
        sub_patients_dic['train'] = patients[:train_num]
        sub_patients_dic['val'] = patients[train_num: train_num + val_num]
        sub_patients_dic['test'] = patients[train_num + val_num:]
        sub_patients_dic['debug'] = patients[: val_num]
        return sub_path,sub_patients_dic


    def __initialize_info(self,one_label_path):
        label = sitk.ReadImage(one_label_path)
        label = sitk.GetArrayFromImage(label)
        label_list = list(np.unique(label))
        num_label = len(label_list)
        self.standard_label_index = tuple([int(item) for item in label_list])
        print('the standard label index is :{}'.format(self.standard_label_index))
        print('the num of the class: {}'.format(num_label))
        self.num_label = num_label
        linfo={}
        linfo['num_label'] = num_label
        linfo['standard_label_index']= self.standard_label_index
        linfo['img_size'] = label.shape
        linfo['spacing']=np.array([-1])
        self.save_shared_info(linfo)



    def __gen_intra_pair_list(self,patients, pair_num_limit = 1000):
        intra_pair_list = []
        for patient in patients:
            for modality in patient.modality:
                for specificity in patient.specificity:
                    intra_image_list = patient.get_slice_path_list(modality,specificity)
                    intra_label_list = patient.get_label_path_list(modality,specificity)
                    num_images = len(intra_image_list)
                    for i, image in enumerate(intra_image_list):
                        for j in range(i+1, num_images):
                            intra_pair_list.append([intra_image_list[i],intra_image_list[j],
                                                    intra_label_list[i],intra_label_list[j]])
                            # intra_pair_list.append([intra_image_list[j], intra_image_list[i], used in old code
                            #                         intra_label_list[j], intra_label_list[i]])
            # if pair_num_limit>=0 and len(intra_pair_list)> 5*pair_num_limit:
            #     break
        if len(patients)>0:
            self.__initialize_info(patients[0].get_label_path_list()[0])
        if pair_num_limit >= 0:
            random.shuffle(intra_pair_list)
            return intra_pair_list[:pair_num_limit]
        else:
            return intra_pair_list


    def __gen_inter_pair_list(self, patients,pair_num_limit = 1000):
        """
        here we only use the first time period for inter image registration
        :param patients:
        :param pair_num_limit:
        :return:
        """
        inter_pair_list = []
        num_patients = len(patients)
        if pair_num_limit==0:
            return inter_pair_list
        while True:
            rand_pair_id = [int(num_patients * random.random()), int(num_patients * random.random())]
            patient_a = patients[rand_pair_id[0]]
            patient_b = patients[rand_pair_id[1]]
            modality_list = patient_a.modality
            specificity_list = patient_a.specificity
            modality_id = int(len(modality_list)*random.random())
            specificity_id = int(len(specificity_list)*random.random())
            patient_a_slice = patient_a.get_slice_path_list(modality_list[modality_id],specificity_list[specificity_id])
            patient_a_label = patient_a.get_label_path_list(modality_list[modality_id],specificity_list[specificity_id])
            patient_b_slice = patient_b.get_slice_path_list(modality_list[modality_id],specificity_list[specificity_id])
            patient_b_label = patient_b.get_label_path_list(modality_list[modality_id],specificity_list[specificity_id])

            slice_id_a = int(len(patient_a_slice)*random.random())
            slice_id_b = int(len(patient_b_slice)*random.random())
            if modality_list[modality_id] in patient_b.modality:
                if specificity_list[specificity_id] in patient_b.specificity:
                    pair = [patient_a_slice[slice_id_a], patient_b_slice[slice_id_b],
                            patient_a_label[slice_id_a], patient_b_label[slice_id_b]]
                    inter_pair_list.append(pair)
                    if len(inter_pair_list)> pair_num_limit:
                        break

        self.__initialize_info(patients[0].get_label_path_list()[0])
        return inter_pair_list

    def __gen_path_and_name_dic(self, pair_list_dic):

        divided_path_and_name_dic={}
        divided_path_and_name_dic['pair_path_list'] = pair_list_dic
        divided_path_and_name_dic['pair_name_list'] = self.__gen_pair_name_list(pair_list_dic)
        return divided_path_and_name_dic


    def __gen_pair_name_list(self,pair_list_dic):
        return {sess:[get_file_name(path[0])+'_'+get_file_name(path[1]) for path in pair_list_dic[sess]] for sess in sesses}







    def gen_pair_dic(self):
        if self.only_test_set:
            self.divided_ratio = [0.,0.,1.] ##############################################
            num_pair_limit = 150 # -1 if self.sched=='intra' else 300 used in old code

        else:
            num_pair_limit = 2000  #-1 used in old code
        sub_folder_dic, sub_patients_dic =self.__divide_into_train_val_test_set(self.output_path,self.patients,self.divided_ratio)
        gen_pair_list_func = self. __gen_intra_pair_list if self.sched=='intra' else self.__gen_inter_pair_list
        max_ratio = {'train':self.divided_ratio[0],'val':self.divided_ratio[1],'test':self.divided_ratio[2],'debug':self.divided_ratio[1]}
        pair_list_dic ={sess: gen_pair_list_func(sub_patients_dic[sess],int(num_pair_limit*max_ratio[sess])) for sess in sesses}
        divided_path_and_name_dic = self.__gen_path_and_name_dic(pair_list_dic)
        return (sub_folder_dic,divided_path_and_name_dic)



    def save_pair_to_txt(self,info=None):
        sub_folder_dic, divided_path_and_name_dic = info

        if not self.saving_h5py:
            saving_pair_info(sub_folder_dic, divided_path_and_name_dic)
        else:
            for sess in sesses:
                h5py_data_folder = sub_folder_dic[sess]
                pair_path_list = [[os.path.join(h5py_data_folder,get_file_name(fps[i])+'.h5py') for i in [0,1]] for fps in divided_path_and_name_dic['pair_path_list'][sess]]
                divided_path_and_name_dic['pair_path_list'][sess] = pair_path_list
            saving_pair_info(sub_folder_dic, divided_path_and_name_dic)



    def save_pair_to_h5py(self,info=None):
        sub_folder_dic, divided_path_and_name_dic =info
        for sess in sesses:
            self.pro_data_path = sub_folder_dic[sess]
            pair_path_list_part = np.array_split(divided_path_and_name_dic['pair_path_list'][sess],number_of_workers)
            with Pool(processes=number_of_workers) as pool:
                pool.map(self.save_file_to_h5py, pair_path_list_part)





    def save_file_to_h5py(self,info=None):
        file_path_list = info
        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(file_path_list)).start()
        for i, fps in enumerate(file_path_list):
            for j in range(2):
                f_path = os.path.join(self.pro_data_path,get_file_name(fps[j])+'.h5py')
                if not os.path.exists(f_path):
                    img_np, info = self.read_file(fps[j], is_label=False)
                    label_np, linfo = self.read_file(fps[j+2], is_label=True)
                    img_np = img_np.astype(np.float32)
                    label_np = label_np.astype(np.float32)
                    save_to_h5py(f_path, img_np, label_np, get_file_name(fps[j]), info, verbose=False)
                pbar.update(i + 1)
        pbar.finish()


    def generate_pair_list(self):
        """
        for compatiblity
        :return:
        """
        return None





class RegDatasetPool(object):
    def create_dataset(self,dataset_name, sched, full_comb):
        self.dataset_dic = {'oasis2d':Oasis2DDataSet,
                                   'lpba':VolLabCusDataSet,
                                    'ibsr':VolLabCusDataSet,
                                     'cumc':VolLabCusDataSet,
                                    'oai':OaiDataSet}

        dataset = self.dataset_dic[dataset_name](sched, full_comb)
        return dataset

class Oasis2DDataSet(UnlabeledDataSet, MixedDataSet):
    """"
    sched:  'inter': inter_personal,  'intra': intra_personal
    """
    def __init__(self, sched, full_comb=False):
        file_type_list = ['*a.mhd'] if sched == 'intra' else ['*_1_a.mhd', '*_2_a.mhd', '*_3_a.mhd']
        UnlabeledDataSet.__init__(self, 'mixed', file_type_list, sched)
        MixedDataSet.__init__(self, 'mixed', file_type_list, sched, full_comb)




class VolLabCusDataSet(VolumetricDataSet, LabeledDataSet, CustomDataSet):
    def __init__(self, sched, full_comb=True):
        VolumetricDataSet.__init__(self, 'custom', ['*.nii'])
        LabeledDataSet.__init__(self, 'custom', ['*.nii'])
        CustomDataSet.__init__(self,'custom', ['*.nii'], full_comb)

class OaiDataSet(PatientStructureDataSet):
    def __init__(self, sched, full_comb=True):
        img_poster= ['*image.nii.gz']
        PatientStructureDataSet.__init__(self,None,None,sched)







if __name__ == "__main__":
    print('debugging')
    # #########################       OASIS TESTING           ###################################3
    #
    # path = '/playpen/zyshen/data/oasis'
    # name = 'oasis'
    # divided_ratio = (0.6, 0.2, 0.2)
    #
    # ###################################################
    # #oasis  intra testing
    # full_comb = True
    # sched= 'intra'
    #
    # output_path = '/playpen/zyshen/data/'+ name+'_pre_'+ sched
    # oasis = Oasis2DDataSet(name='oasis',sched=sched, full_comb=True)
    # oasis.set_data_path(path)
    # oasis.set_output_path(output_path)
    # oasis.set_divided_ratio(divided_ratio)
    # oasis.prepare_data()


    # ###################################################
    # # oasis inter testing
    # sched='inter'
    # full_comb = False
    # output_path = '/playpen/zyshen/data/' + name + '_pre_' + sched
    # oasis = Oasis2DDataSet(name='oasis', sched=sched, full_comb=full_comb)
    # oasis.set_data_path(path)
    # oasis.set_output_path(output_path)
    # oasis.set_divided_ratio(divided_ratio)
    # oasis.prepare_data()




    # ###########################       LPBA TESTING           ###################################
    # path = '/playpen/data/quicksilver_data/testdata/LPBA40/brain_affine_icbm'
    # label_path = '/playpen/data/quicksilver_data/testdata/LPBA40/label_affine_icbm'
    # file_type_list = ['*.nii']
    # full_comb = False
    # name = 'lpba'
    # output_path = '/playpen/zyshen/data/' + name + '_pre'
    # divided_ratio = (0.6, 0.2, 0.2)
    #
    # ###################################################
    # #lpba testing
    #
    # sched= 'intra'
    #
    # lpba = LPBADataSet(name=name, full_comb=full_comb)
    # lpba.set_data_path(path)
    # lpba.set_output_path(output_path)
    # lpba.set_divided_ratio(divided_ratio)
    # lpba.set_label_path(label_path)
    # lpba.prepare_data()


    # ###########################       LPBA Slicing TESTING           ###################################
    # path = '/playpen/data/quicksilver_data/testdata/LPBA40/brain_affine_icbm'
    # label_path = '/playpen/data/quicksilver_data/testdata/LPBA40/label_affine_icbm'
    # full_comb = False
    # name = 'lpba'
    # output_path = '/playpen/zyshen/data/' + name + '_pre_slicing'
    # divided_ratio = (0.6, 0.2, 0.2)
    #
    # ###################################################
    # #lpba testing
    #
    #
    # lpba = VolLabCusDataSet(sched='', full_comb=full_comb)
    # lpba.set_slicing(90,1)
    # lpba.set_data_path(path)
    # lpba.set_output_path(output_path)
    # lpba.set_divided_ratio(divided_ratio)
    # lpba.set_label_path(label_path)
    # lpba.prepare_data()
    oasis =  PatientStructureDataSet('reg',['nii.gz'],'inter')
    data_root_path = "/playpen/zyshen/summer/oasis_registration/reg_0220/data"
    output_path = '/playpen/zyshen/data/' + "reg_debug_3000_pair_oasis3_reg_inter"
    divided_ratio = [0.7,0.1,0.2]
    oasis.set_data_root_path(data_root_path)
    oasis.set_output_path(output_path)
    oasis.set_divided_ratio(divided_ratio)
    oasis.prepare_data()







