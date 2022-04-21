from __future__ import print_function
import progressbar as pb

from easyreg.reg_data_utils import *

from data_pre.reg_preprocess_example.oasis_longitude_reg import *
import copy

sesses = ['train', 'val', 'test', 'debug']
number_of_workers = 10
warning_once = True

class BaseRegDataSet(object):

    def __init__(self, dataset_type, sched=None):
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
        self.file_type_list = None
        self.max_used_train_samples = -1
        self.max_pairs = -1

        self.sever_switch = None
        self.save_format = 'h5py'
        """currently only support h5py"""
        self.sched = sched
        """inter or intra, for inter-personal or intra-personal registration"""
        self.dataset_type = dataset_type
        """custom or mixed"""
        self.saving_h5py=False
        """if true, save the preprocessed results as h5py"""
        self.normalize= False
        """ settings for normalization, currently not used"""
        self.divided_ratio = (0.7, 0.1, 0.2)
        """divided the data into train, val, test set"""

    def generate_pair_list(self):
        pass


    def set_data_path(self, path):
        self.data_path = path

    def set_label_path(self, path):
        self.label_path = path

    def set_output_path(self, path):
        self.output_path = path
        make_dir(path)


    def set_divided_ratio(self,ratio):
        self.divided_ratio = ratio

    def get_file_num(self):
        return len(self.pair_path_list)

    def get_pair_name_list(self):
        return self.pair_name_list


    def save_pair_to_txt(self,info=None):
        pass

    def gen_pair_dic(self):
        pass



    def prepare_data(self):
        """
        preprocessig  data for each dataset
        :return:
        """
        print("starting preapare data..........")
        print("the output file path is: {}".format(self.output_path))

        info=self.gen_pair_dic()
        self.save_pair_to_txt(copy.deepcopy(info))
        print("data preprocessing finished")

class CustomDataSet(BaseRegDataSet):
    """
    This class only compatible with dataset_type "mixed" and "custom"
    unlabeled dataset
    """
    def __init__(self,dataset_type, sched=None):
        BaseRegDataSet.__init__(self,dataset_type,sched)
        self.aug_test_for_seg_task = False
        self.reg_coupled_pair = False
        self.coupled_pair_list = []
        self.find_corr_label = find_corr_map
        self.label_switch = ('', '')
        self.label_path = None


    def __gen_path_and_name_dic(self, pair_list_dic):

        divided_path_and_name_dic = {}
        divided_path_and_name_dic['pair_path_list'] = pair_list_dic
        divided_path_and_name_dic['pair_name_list'] = self.__gen_pair_name_list(pair_list_dic)
        return divided_path_and_name_dic

    def __gen_pair_name_list(self, pair_list_dic):
        return {sess: [generate_pair_name([path[0],path[1]]) for path in pair_list_dic[sess]] for sess
                in sesses}


    def __gen_pair_list(self,img_path_list, pair_num_limit = 1000):
        img_pair_list = []
        label_path_list  = self.find_corr_label(img_path_list, self.label_path, self.label_switch)
        num_img = len(img_path_list)
        for i in range(num_img):
            count_max=15 #15
            img_pair_list_tmp =[]
            for j in range(num_img):
                if i!=j:
                    if self.label_path is not None:
                        img_pair_list_tmp.append([img_path_list[i],img_path_list[j],
                                            label_path_list[i],label_path_list[j]])
                    else:
                        img_pair_list_tmp.append([img_path_list[i], img_path_list[j]])
            if len(img_pair_list_tmp)>count_max:
                img_pair_list_tmp= random.sample(img_pair_list_tmp,count_max)
            img_pair_list += img_pair_list_tmp
        if pair_num_limit >= 0:
            random.shuffle(img_pair_list)
            return img_pair_list[:pair_num_limit]
        else:
            return img_pair_list



    def __gen_pair_list_from_two_list(self,img_path_list_1, img_path_list_2, pair_num_limit = 1000):
        img_pair_list = []
        label_path_list_1 = self.find_corr_label(img_path_list_1, self.label_path, self.label_switch)
        label_path_list_2 = self.find_corr_label(img_path_list_2, self.label_path, self.label_switch)
        num_img_1 = len(img_path_list_1)
        num_img_2 = len(img_path_list_2)
        for i in range(num_img_1):
            count_max=15 #15
            img_pair_list_tmp =[]
            for j in range(num_img_2):
                if self.label_path is not None:
                    img_pair_list_tmp.append([img_path_list_1[i],img_path_list_2[j],
                                        label_path_list_1[i],label_path_list_2[j]])
                else:
                    img_pair_list_tmp.append([img_path_list_1[i], img_path_list_2[j]])
            if len(img_pair_list_tmp)>count_max:
                img_pair_list_tmp= random.sample(img_pair_list_tmp,count_max)
            img_pair_list += img_pair_list_tmp
        if pair_num_limit >= 0:
            random.shuffle(img_pair_list)
            return img_pair_list[:pair_num_limit]
        else:
            return img_pair_list


    def __gen_pair_list_with_coupled_list(self,pair_path_list, pair_num_limit = 1000):
        img_pair_list = []
        img_path_list_1 =  [pair_path[0] for pair_path in pair_path_list]
        img_path_list_2 =  [pair_path[1] for pair_path in pair_path_list]
        label_path_list_1 = self.find_corr_label(img_path_list_1, self.label_path, self.label_switch)
        label_path_list_2 = self.find_corr_label(img_path_list_2, self.label_path, self.label_switch)
        has_label = [os.path.exists(p1) and os.path.exists(p2) for p1, p2 in zip(label_path_list_1,label_path_list_2)]
        num_img_1 = len(img_path_list_1)
        num_img_2 = len(img_path_list_2)
        assert num_img_1 == num_img_2
        for i in range(num_img_1):
            if has_label[i]:
                img_pair_list.append([img_path_list_1[i],img_path_list_2[i],
                                    label_path_list_1[i],label_path_list_2[i]])
            else:
                img_pair_list.append([img_path_list_1[i], img_path_list_2[i]])
        if pair_num_limit >= 0:
            random.shuffle(img_pair_list)
            return img_pair_list[:pair_num_limit]
        else:
            return img_pair_list
        
        
    def __gen_across_file_pair_dic(self):
        num_pair_limit = self.max_pairs  # -1 
        img_path_list = get_file_path_list(self.data_path, self.file_type_list)
        if self.sever_switch is not None:
            img_path_list = [img_path.replace(self.sever_switch[0], self.sever_switch[1]) for img_path in img_path_list]
        sub_folder_dic, sub_patients_dic = self.__divide_into_train_val_test_set(self.output_path, img_path_list,
                                                                                 self.divided_ratio)
        gen_pair_list_func = self.__gen_pair_list
        max_ratio = {'train': self.divided_ratio[0], 'val': self.divided_ratio[1], 'test': self.divided_ratio[2],
                     'debug': self.divided_ratio[1]}
        if self.max_used_train_samples>-1:
            sub_patients_dic['train'] = sub_patients_dic['train'][:self.max_used_train_samples]
        if not self.aug_test_for_seg_task:
            pair_list_dic = {sess: gen_pair_list_func(sub_patients_dic[sess], int(
                num_pair_limit * max_ratio[sess]) if num_pair_limit > 0 else -1) for
                             sess in sesses}
        else:
            pair_list_dic = {sess: gen_pair_list_func(sub_patients_dic[sess], int(
                num_pair_limit * max_ratio[sess]) if num_pair_limit > 0 else -1) for
                             sess in ['train', 'val', 'debug']}
            pair_list_dic['test'] = self.__gen_pair_list_from_two_list(sub_patients_dic['test'],
                                                                       sub_patients_dic['train'][:10],
                                                                       int(num_pair_limit * max_ratio[
                                                                           "test"]) if num_pair_limit > 0 else -1)

        divided_path_and_name_dic = self.__gen_path_and_name_dic(pair_list_dic)
        return (sub_folder_dic, divided_path_and_name_dic)

    def __gen_pair_dic_with_given_pair(self):
        num_pair_limit = self.max_pairs  # -1 
        pair_path_list =self.coupled_pair_list
        if self.sever_switch is not None:
            pair_path_list = [[img_path.replace(self.sever_switch[0], self.sever_switch[1]) for img_path in pair_path] for pair_path in pair_path_list]
        sub_folder_dic, sub_patients_dic = self.__divide_into_train_val_test_set(self.output_path, pair_path_list,
                                                                                 self.divided_ratio)
        gen_pair_list_func = self.__gen_pair_list_with_coupled_list
        max_ratio = {'train': self.divided_ratio[0], 'val': self.divided_ratio[1], 'test': self.divided_ratio[2],
                     'debug': self.divided_ratio[1]}
        if self.max_used_train_samples > -1:
            sub_patients_dic['train'] = sub_patients_dic['train'][:self.max_used_train_samples]

        pair_list_dic = {sess: gen_pair_list_func(sub_patients_dic[sess], int(
            num_pair_limit * max_ratio[sess]) if num_pair_limit > 0 else -1) for
                         sess in sesses}

        divided_path_and_name_dic = self.__gen_path_and_name_dic(pair_list_dic)
        return (sub_folder_dic, divided_path_and_name_dic)
        

    def gen_pair_dic(self):
        if not self.reg_coupled_pair:
            return self.__gen_across_file_pair_dic()
        else:
            return self.__gen_pair_dic_with_given_pair()

       



    def __divide_into_train_val_test_set(self,root_path, img_path_list, ratio):
        num_img = len(img_path_list)
        train_ratio = ratio[0]
        val_ratio = ratio[1]
        sub_path = {x: os.path.join(root_path, x) for x in ['train', 'val', 'test', 'debug']}
        nt = [make_dir(sub_path[key]) for key in sub_path]
        if sum(nt):
            raise ValueError("the data task has already exist and has been created via random strategy, to avoid running by accident, the program blocks\n"
                             " To aovid blocking, you may need to manually delete the output folder and rerun the program")

        train_num = int(train_ratio * num_img)
        val_num = int(val_ratio * num_img)
        sub_patients_dic = {}
        sub_patients_dic['train'] = img_path_list[:train_num]
        sub_patients_dic['val'] = img_path_list[train_num: train_num + val_num]
        sub_patients_dic['test'] = img_path_list[train_num + val_num:]
        sub_patients_dic['debug'] = img_path_list[: val_num]
        return sub_path,sub_patients_dic



    def save_pair_to_txt(self, info=None):
        sub_folder_dic, divided_path_and_name_dic = info
        saving_pair_info(sub_folder_dic, divided_path_and_name_dic)








class PatientStructureDataSet(BaseRegDataSet):
    """

    The data in self.data_root_path would be loaded,
    """

    def __init__(self, dataset_type, sched):
        BaseRegDataSet.__init__(self, dataset_type,sched)
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

        return inter_pair_list

    def __gen_path_and_name_dic(self, pair_list_dic):

        divided_path_and_name_dic={}
        divided_path_and_name_dic['pair_path_list'] = pair_list_dic
        divided_path_and_name_dic['pair_name_list'] = self.__gen_pair_name_list(pair_list_dic)
        return divided_path_and_name_dic


    def __gen_pair_name_list(self,pair_list_dic):
        return {sess:[generate_pair_name([path[0],path[1]]) for path in pair_list_dic[sess]] for sess in sesses}




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
                h5py_output_root_path = sub_folder_dic[sess]
                pair_path_list = [[os.path.join(h5py_output_root_path,get_file_name(fps[i])+'.h5py') for i in [0,1]] for fps in divided_path_and_name_dic['pair_path_list'][sess]]
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






class RegDatasetPool(object):
    def create_dataset(self,dataset_name, sched=None):
        self.dataset_dic = {'custom':CustomDataSet,
                                    'oai':OaiDataSet}

        dataset = self.dataset_dic[dataset_name](dataset_name,sched)
        return dataset


class OaiDataSet(PatientStructureDataSet):
    def __init__(self,dataset_name, sched):
        PatientStructureDataSet.__init__(self,dataset_name,sched)





if __name__ == "__main__":
    data_path = "../demo/lpba_examples/data"
    label_path = "../demo/lpba_examples/label"
    divided_ratio = (0.4, 0.4, 0.2)
    file_type_list = ['*.nii.gz']
    name = 'custom'
    output_path = '../demo/demo_training_reg_net/lpba'
    lpba = RegDatasetPool().create_dataset(name)
    lpba.file_type_list = file_type_list
    lpba.set_data_path(data_path)
    lpba.set_output_path(output_path)
    lpba.set_divided_ratio(divided_ratio)
    lpba.set_label_path(label_path)
    lpba.prepare_data()



    # path = '/playpen/data/quicksilver_data/testdata/LPBA40/brain_affine_icbm'
    # data_path = "/playpen-raid/zyshen/data/lpba_seg_resize/resized_img"
    # label_path = '/playpen-raid/zyshen/data/lpba_seg_resize/label_filtered'
    # divided_ratio = (0.625, 0.125, 0.25)
    #
    # file_type_list = ['*.nii.gz']
    # name = 'custom'
    # num_c_list = [5, 10, 15, 20, 25]
    # for num_c in num_c_list:
    #     output_path = '/playpen-raid/zyshen/data/lpba_reg/train_with_test_aug_{}'.format(num_c)
    #     #sever_switch = ('/playpen-raid', '/pine/scr/z/y')
    #     sever_switch = None
    #
    #     lpba = RegDatasetPool().create_dataset(name)
    #     lpba.aug_test_for_seg_task = True
    #
    #     lpba.file_type_list = file_type_list
    #     lpba.sever_switch = sever_switch
    #     lpba.max_used_train_samples = num_c
    #     lpba.set_data_path(data_path)
    #     lpba.set_output_path(output_path)
    #     lpba.set_divided_ratio(divided_ratio)
    #     lpba.set_label_path(label_path)
    #     lpba.prepare_data()

    # num_c_list = [10,20,30,40]
    # for num_c in num_c_list:
    #     data_path = "/playpen-raid/olut/Nifti_resampled_rescaled_2Left_Affine2atlas"
    #     label_path = "/playpen-raid/olut/Nifti_resampled_rescaled_2Left_Affine2atlas"
    #     output_path = '/playpen-raid/zyshen/data/oai_reg/train_with_test_aug_{}'.format(num_c)
    #     divided_ratio = (0.8, 0.1, 0.1)
    #     name = 'custom'
    #     file_type_list =['*image.nii.gz']
    #     label_switch = ('image', 'masks')
    #     #sever_switch = ('/playpen-raid/olut', '/pine/scr/z/y/zyshen')
    #     sever_switch = None
    #
    #     oai = RegDatasetPool().create_dataset(name)
    #     oai.aug_test_for_seg_task=True
    #
    #
    #     oai.label_switch = label_switch
    #     oai.sever_switch = sever_switch
    #     oai.max_used_train_samples=num_c
    #     oai.file_type_list = file_type_list
    #     oai.set_data_path(data_path)
    #     oai.set_output_path(output_path)
    #     oai.set_divided_ratio(divided_ratio)
    #     oai.set_label_path(label_path)
    #     oai.prepare_data()


    # data_path = "/playpen-raid1/Data/Lung_Registration_clamp_normal"
    # source_image_path_list = glob(os.path.join(data_path, "**", "*EXP*img*"))
    # target_image_path_list = [path.replace("_EXP_", "_INSP_") for path in source_image_path_list]
    # coupled_pair_path_list = list(zip(source_image_path_list,target_image_path_list))
    # divided_ratio = (0.8, 0.05, 0.15)
    # name = 'custom'
    # output_path = '/playpen-raid1/zyshen/data/reg_new_lung'
    # #sever_switch = ('/playpen-raid', '/pine/scr/z/y')
    # label_switch = ('_img', '_label')
    #
    # lung = RegDatasetPool().create_dataset(name)
    # lung.reg_coupled_pair=True
    # lung.coupled_pair_list = coupled_pair_path_list
    # lung.label_switch = label_switch
    # lung.set_data_path(data_path)
    # lung.set_output_path(output_path)
    # lung.set_divided_ratio(divided_ratio)
    # lung.prepare_data()
    #
    #



