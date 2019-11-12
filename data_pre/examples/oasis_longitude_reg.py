
import sys
import os
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../easyreg'))

import numpy as np
from functools import reduce
from easyreg.reg_data_utils import make_dir
import random
import SimpleITK as sitk
from multiprocessing import Pool



"""
######################################### Section 1. Raw Data Organization  ###############################################

data root:  /playpen/zhenlinx/Data/OAI_segmentation/

The images were saved as nifti format at ./Nifti_6sets_rescaled
The list file are images_6sets_left.txt and images_6sets_right.txt. The images file name are ordered by patient IDs but not ordered by time for each patient.
For a file name like 9000099_20050712_SAG_3D_DESS_LEFT_10424405_image.nii.gz
        9000099 is the patient ID, 20050712 is the scan date,
        SAG_3D_DESS is the image modality, 
        LEFT means left knee, 
        and 10424405 is the image id.

Segmentations for images_6sets_right  predicted by UNetx2 were saved at  
/playpen/zhenlinx/Data/OAI_segmentation/segmentations/images_6sets_right/Cascaded_2_AC_residual-1-s1_end2end_multi-out_UNet_bias_Nifti_rescaled_train1_patch_128_128_32_batch_2_sample_0.01-0.02_cross_entropy_lr_0.0005_scheduler_multiStep_02262018_013038



#######################################  Section 2. Processed Data Organization  #############################################

data root:  /playpen/zyshen/summer/oai_registration/data


The patient id will be saved in patient_id.txt

(to do):The modality list will be saved in modality.txt  ( where each line will be organized as following  MRI  mod1  #newline  CT  mod2 ...........)

the patient slices list will be save in folder  "patient_slice"

./patient_slice/  :                     each patient_id is a separate folder
./patient_slice/idxxxxxxx/:             each modality is a separate folder
./patient_slice/idxxxxxxx/mod1/:        each specificity is a separate folder ie. left,  right
./patient_slice/idxxxxxxx/mod1/spec1/   paths of slice labels will be recorded in "img_label.txt", each line has a slice path and corresponded label path

########################################   Section 3. Code Organization  ####################################################




abnormal example list:
!! image size not matched , img:9901199_20090422_SAG_3D_DESS_RIGHT_12800503_image.nii.gz sz:(160, 384, 352) 
!! image size not matched , img:9052335_20090126_SAG_3D_DESS_RIGHT_12766414_image.nii.gz sz:(176, 384, 384) 
!! image size not matched , img:9163391_20110808_SAG_3D_DESS_LEFT_16613250603_image.nii.gz sz:(159, 384, 384) 
!! image size not matched , img:9712762_20090420_SAG_3D_DESS_RIGHT_12583306_image.nii.gz sz:(160, 384, 352) 
!! image size not matched , img:9388265_20040405_SAG_3D_DESS_LEFT_10016906_image.nii.gz sz:(176, 384, 384) 
!! image size not matched , img:9388265_20040405_SAG_3D_DESS_LEFT_10016903_image.nii.gz sz:(176, 384, 384) 
!! image size not matched , img:9938453_20071130_SAG_3D_DESS_RIGHT_12140103_image.nii.gz sz:(159, 384, 384) 
!! image size not matched , img:9452305_20070228_SAG_3D_DESS_RIGHT_11633112_image.nii.gz sz:(109, 384, 384) 
!! image size not matched , img:9219500_20080326_SAG_3D_DESS_RIGHT_12266509_image.nii.gz sz:(8, 384, 384) 
!! image size not matched , img:9011949_20060118_SAG_3D_DESS_LEFT_10667703_image.nii.gz sz:(156, 384, 384) 
!! image size not matched , img:9885303_20051212_SAG_3D_DESS_LEFT_10624403_image.nii.gz sz:(155, 384, 384) 
!! image size not matched , img:9833782_20090519_SAG_3D_DESS_RIGHT_12802313_image.nii.gz sz:(176, 384, 384) 
!! image size not matched , img:9462278_20050524_SAG_3D_DESS_RIGHT_10546912_image.nii.gz sz:(156, 384, 384) 
!! image size not matched , img:9126260_20060921_SAG_3D_DESS_RIGHT_11309309_image.nii.gz sz:(66, 384, 384) 
!! image size not matched , img:9487462_20081003_SAG_3D_DESS_RIGHT_11495603_image.nii.gz sz:(176, 384, 384) 
!! image size not matched , img:9847480_20081007_SAG_3D_DESS_RIGHT_11508512_image.nii.gz sz:(159, 384, 384) 
!! image size not matched , img:9020714_20101207_SAG_3D_DESS_RIGHT_16613171935_image.nii.gz sz:(118, 384, 384) 







class DataPrepare:
                class Dataprepare are specificed to oai_dataset, which will transfer Raw Data Organization into Processed Data Organization
                
                object variable included:
                    raw_data_path, output_data_path, 
                    
                function_call_outside:
                    prepare_data()
                    
                function_call_inside:
                    __factor_file(file_name)
                    __factor_file_list()
                    __build_and_write_in()




class Patient:  
                class Patient are initialized from each patient_id folder, so it need  the path of patient_id folder as input
                
                object varaible included: 
                    basic_information:
                        patient_id, modality (tuple), specificity(tuple), patient_slices_path_dic ([modality][specificity]: slice_list)  (dict),
                        patient_slices_num_dic (dict)
                        
                    annotation_information:
                        has_label, label_is_complete, patient_slices_label_path_dic (dict), 
                
                function called outside:
                    check_if_taken(self, modality=None, specificity=None, len_time_range=None, has_label=None)
                    get_slice_list(modality,specificity)
                    get_label_path_list(modality,specificity)
                    get_slice_num(modality,specificity)
                    
                
                function called inside:
                    __init__()
                    
                    
                    
                
                
                                

class Patients: 
                class Patients are initialized from patient_slice folder, so it need the path of patient_slice folder as input
                1.0this class has a list of Patient class, and can set some condtions in order to filter the patients

                object varaible included:
                    patients_id_list (list), patients( list of class Patient)
                    
                    
                function called outside:
                    get_that_patient(self,patient_id)
                    get_filtered_patients_list(self,modality=None, specificity=None, has_label=None, num_of_patients= -1, len_time_range=None, use_random=False):
                    
                    to do:
                    get_patients_statistic_distribution(is_modality=False, is_specificity= False, has_label=False)
                    
                function call inside:
                     __read_patients_id_list_from_txt(self)
                      __init_basic_info
                      __init_full_info
                      
                      





class OAILongitudeRegistration:
                first, we need to filter some patients  and implement longitude registration,
                
                for each patient, we would registrate images from different time phases to time phase 0
                
                then, we need use the moving map to map label, then do result analysis
                
                functions called outside:
                set_patients()
                set_model()
                do_registration()
                do_result_analysis
                
                
                function called inside:
                __initial_model()
                __inital_source_and_target()
                __do_registration()


                                   

"""


class Patients(object):
    def __init__(self,full_init=False, root_path= ''):
        self.full_init = full_init
        self.root_path = root_path if len(root_path) else "/playpen/zyshen/summer/oasis_registration/reg_0219/data"
        self.patients_id_txt_name = 'patient_id.txt'
        self.patients_info_folder = 'patient_slice'
        self.patients_id_list= []
        self.patients = []
        if not full_init:
            self.__init_basic_info()
        else:
            self.__init_full_info()




    def __init_basic_info(self):
        self.__read_patients_id_list_from_txt()
        self.patients_num = len(self.patients_id_list)


    def __init_full_info(self):
        self.__read_patients_id_list_from_txt()
        self.patients_num = len(self.patients_id_list)
        for patient_id in self.patients_id_list:
            patient_info_path = os.path.join(self.root_path, self.patients_info_folder, patient_id)
            self.patients.append(Patient(patient_info_path))



    def get_all_patients(self):
        if self.full_init:
            return self.patients
        else:
            self.__init_full_info()
            return self.patients

    def get_that_patient(self,patient_id):
        assert patient_id in self.patients_id_list
        patient_info_path = os.path.join(self.root_path, self.patients_info_folder, patient_id)
        patient = Patient(patient_info_path)
        return patient



    def get_filtered_patients_list(self,modality=None, specificity=None, has_complete_label=None, num_of_patients= -1, len_time_range=None, use_random=False):
        index = list(range(self.patients_num))
        num_of_patients = num_of_patients if num_of_patients>0 else self.patients_num
        filtered_patients_list =[]
        if use_random:
            random.shuffle(index)
        count = 0
        for i in index:
            if not self.full_init:
                patient_id = self.patients_id_list[i]
                patient_info_path = os.path.join(self.root_path, self.patients_info_folder, patient_id)
                patient = Patient(patient_info_path)
            else:
                patient = self.patients[i]
            modality = patient.modality[0] if modality is None else modality
            specificity_tmp = patient.specificity[0] if specificity is None else specificity
            if_taken = patient.check_if_taken(modality=modality,specificity=specificity_tmp,has_complete_label=has_complete_label,len_time_range=len_time_range)
            if if_taken:
                filtered_patients_list.append(patient)
                count+=1
                if count>= num_of_patients:
                    break
        if len(filtered_patients_list)< num_of_patients:
            print("not enough patients meet the filter requirement. We want {} but got {} patients".format(num_of_patients, len(filtered_patients_list)))
        return filtered_patients_list





    def __read_patients_id_list_from_txt(self):
        """
        get the patient id from the txt i.e patient_id.txt
        :param file_name:
        :return: type list, list of patient id
        """

        txt_path = os.path.join(self.root_path, self.patients_id_txt_name)
        with open(txt_path, 'r') as f:
            content = f.read().splitlines()
            if len(content) > 0:
                infos = [line.split('\t') for line in content]
            self.patients_id_list = [info[0] for info in infos]
            self.patients_has_label_list = [info[1]=='annotation_complete' for info in infos]




class Patient():
    def __init__(self, path):
        # patient_id, modality(set), specificity(set), patient_slices_path_dic([modality][specificity]: slice_list)
        self.patient_root_path = path
        self.patient_id = -1
        self.modality = None
        self.specificity = None
        self.patient_slices_path_dic = {}
        self.patient_slices_num_dic = {}
        self.has_label = False
        self.label_is_complete = True
        self.patient_slices_label_path_dic = {}
        self.patient_has_label_dic= {}
        self.txt_file_name = 'img_label.txt'
        self.__init_patient_info()




    def __init_patient_info(self):
        self.patient_id = os.path.split(self.patient_root_path)[1]
        modality_list = os.listdir(self.patient_root_path)
        specificity_list = os.listdir(os.path.join(self.patient_root_path, modality_list[0]))
        self.modality = tuple(modality_list)
        self.specificity = tuple(specificity_list)
        for mod in self.modality:
            for spec in self.specificity:
                if mod not in self.patient_slices_path_dic:
                    self.patient_slices_path_dic[mod]={}
                    self.patient_slices_label_path_dic[mod]={}
                    self.patient_has_label_dic[mod]= {}
                    self.patient_slices_num_dic[mod]={}
                self.patient_slices_path_dic[mod][spec], self.patient_slices_label_path_dic[mod][spec] \
                    = self.__init_path_info(mod, spec)
                self.patient_slices_num_dic[mod][spec] = len(self.patient_slices_path_dic[mod][spec])
                has_complete_spec_label = True
                for label_path in self.patient_slices_label_path_dic[mod][spec]:
                    if label_path !='None':
                        self.has_label = True
                    else:
                        self.label_is_complete= False
                        has_complete_spec_label= False
                self.patient_has_label_dic[mod][spec] = has_complete_spec_label



    def __init_path_info(self,modality, specificity):
        txt_path = os.path.join(self.patient_root_path,modality, specificity,self.txt_file_name)
        paths = []
        with open(txt_path, 'r') as f:
            content = f.read().splitlines()
            if len(content) > 0:
                paths = [line.split('\t') for line in content]
            slices_path_list = [path[0] for path in paths]
            slices_label_path_list = [path[1] for path in paths]
        return slices_path_list,slices_label_path_list

    def check_if_taken(self, modality=None, specificity=None, len_time_range=None, has_complete_label=None):
        modality_met =True if modality is None else modality in self.modality
        specificity_met = True if specificity is None else  specificity in self.specificity
        has_label_met = True if has_complete_label is None else self.label_is_complete == has_complete_label
        if modality not in self.modality or specificity not in self.specificity:
            return False

        len_time_met =True
        if len_time_range is not None:
            cur_len_time = self.patient_slices_num_dic[modality][specificity]
            len_time_met = len_time_range[0]<= cur_len_time and len_time_range[1]>= cur_len_time
        if_taken = modality_met and specificity_met and len_time_met and has_label_met
        return if_taken

    def get_slice_path_list(self, modality=None, specificity=None):
        if modality==None and specificity==None:
            return self.patient_slices_path_dic[self.modality[0]][self.specificity[0]]
        elif modality in self.modality and specificity in self.specificity:
            return self.patient_slices_path_dic[modality][specificity]
        else:
            print("patient{} doesn't has slice in format {} and {}".format(self.patient_id, modality, specificity))
            return []


    def get_label_path_list(self,modality=None, specificity=None):
        if modality==None and specificity==None:
            return self.patient_slices_label_path_dic[self.modality[0]][self.specificity[0]]
        elif modality in self.modality and specificity in self.specificity:
            return self.patient_slices_label_path_dic[modality][specificity]
        else:
            print ("patient{} doesn't has label in format {} and {}".format(self.patient_id, modality, specificity))
            return []

    def get_slice_num(self,modality=None, specificity=None):
        if modality==None and specificity==None:
            return self.patient_slices_num_dic[self.modality[0]][self.specificity[0]]
        elif modality in self.modality and specificity in self.specificity:
            return self.patient_slices_num_dic[modality][specificity]
        else:
            print("patient{} doesn't has slice in format {} and {}".format(self.patient_id, modality, specificity))
            return 0

    def get_path_for_mod_and_spec(self,mod,spec):
        if self.get_slice_num(mod,spec)>0:
            path = os.path.join(self.patient_root_path,mod,spec)
            return path
        else:
            return None



def __debug_check_img_sz(file_path_list):
    fp_to_del = []

    for fp in file_path_list:
        img = sitk.ReadImage(fp)
        img_shape = sitk.GetArrayFromImage(img).shape
        if not img_shape == (256-32,256-32,256-32):
            print("!! image size not matched , img:{} sz:{} \n".format(os.path.split(fp)[1], img_shape))
            fp_to_del.append(fp)
    return fp_to_del

f= __debug_check_img_sz


abnormal_example_list = []










class OASISDataPrepare():
    """
    the dataset is organized in the following style:  patient_id/modality/specificity/
    each folder a txt file named "img_label.txt"
    each line includes a path of volume and path for corresponding label(None if no label),

    e.g of img_label.txt
    /playpen/xhs400/OASIS_3/processed_images_crop16/OAS30001_MR_d3132_brain.nii.gz   /playpen/xhs400/OASIS_3/processed_images_crop16/OAS30001_MR_d3132_label.nii.gz


    """
    def __init__(self):
        using_unlabeled_data = True

        self.raw_data_path_list = ["/playpen/xhs400/OASIS_3/processed_images_centered_224_224_224"]
        self.raw_label_path_list = ["/playpen/xhs400/OASIS_3/processed_images_centered_224_224_224"]
        self.output_root_path = "/playpen/zyshen/summer/oasis_registration/reg_0313/data"
        self.output_data_path = "/playpen/zyshen/summer/oasis_registration/reg_0313/data/patient_slice"

        self.raw_file_path_list = []
        self.raw_file_label_path_list= []
        self.patient_info_dic= {}
        self.image_file_end = '*brain.nii.gz'

        self.label_file_end = '*label.nii.gz'
        self.debug = False


    def prepare_data(self):
        self.get_file_list()
        self.__factor_file_list()
        self.__build_and_write_in()



    def __filter_file(self, path_list, file_end):
        f_filter =[]
        import fnmatch
        for path in path_list:
            for root, dirnames, filenames in os.walk(path):
                for filename in fnmatch.filter(filenames, file_end):
                    f_filter.append(os.path.join(root, filename))
        return f_filter





    def get_file_list(self):

        self.raw_file_path_list = self.__filter_file(self.raw_data_path_list,self.image_file_end)
        self.raw_file_label_path_list = self.__filter_file(self.raw_label_path_list,self.label_file_end)
        self.remove_abnormal_data()

    def remove_abnormal_data(self):
        if self.debug:
            number_of_workers=20
            fp_to_del = []
            fp_to_del_tmp = []
            file_patitions = np.array_split(self.raw_file_path_list, number_of_workers)
            with Pool(processes=number_of_workers) as pool:
                fp_to_del_tmp=pool.map(f, file_patitions)
            for fp_list in fp_to_del_tmp:
                for fp in fp_list:
                    fp_to_del.append(fp)
            print("total {} paths need to be removed".format(len(fp_to_del)))
            for fp in fp_to_del:
                self.raw_file_path_list.remove(fp)
        else:
            for fp in self.raw_file_path_list:
                fn = os.path.split(fp)[1]
                if fn in abnormal_example_list:
                    self.raw_file_path_list.remove(fp)
                    print("!! {} is removed from the image list".format(fn))







    def __factor_file(self, f_path):
        """
        For a file name like 9000099_20050712_SAG_3D_DESS_LEFT_10424405_image.nii.gz
        9000099 is the patient ID, 20050712 is the scan date,
        SAG_3D_DESS is the image modality,
        LEFT means left knee,
        and 10424405 is the image id.
        :return:
        """
        file_name = os.path.split(f_path)[-1]
        factor_list = file_name.split('_')
        patient_id = factor_list[0]
        scan_date = int(factor_list[2].replace('d','1'))
        modality = factor_list[1]
        specificity = "non"
        f = lambda x,y : x+'_'+y
        file_name = reduce(f,factor_list[:3])
        return {'file_path': f_path,'slice_name': file_name,'patient_id':patient_id, 'scan_date':scan_date, 'modality':modality, 'specificity':specificity,'label_path':'None'}



    def __factor_file_list(self):

        for f_path in self.raw_file_path_list:
            fd = self.__factor_file(f_path)
            if fd['patient_id'] not in self.patient_info_dic:
                self.patient_info_dic[fd['patient_id']] = {}
            if fd['modality'] not in self.patient_info_dic[fd['patient_id']]:
                self.patient_info_dic[fd['patient_id']][fd['modality']] = {}
            if fd['specificity'] not in self.patient_info_dic[fd['patient_id']][fd['modality']]:
                self.patient_info_dic[fd['patient_id']][fd['modality']][fd['specificity']] = {}
            cur_dict = self.patient_info_dic[fd['patient_id']][fd['modality']][fd['specificity']][fd['slice_name']]={}
            cur_dict['file_path'] =fd['file_path']
            cur_dict['slice_name'] =fd['slice_name']
            cur_dict['scan_date'] =fd['scan_date']
            cur_dict['label_path'] = fd['label_path']



        for f_path in self.raw_file_label_path_list:
            fd = self.__factor_file(f_path)
            try:
                self.patient_info_dic[fd['patient_id']][fd['modality']][fd['specificity']][fd['slice_name']]['label_path'] = f_path
            except:
                pass




    def __build_and_write_in(self):
        make_dir(self.output_root_path)
        with open(os.path.join(self.output_root_path,'patient_id.txt'),'w') as fr:
            for pat_id in self.patient_info_dic:
                has_complete_label = True
                for mod in self.patient_info_dic[pat_id]:
                    for spec in self.patient_info_dic[pat_id][mod]:
                        folder_path = os.path.join(self.output_data_path,pat_id,mod,spec)
                        make_dir(folder_path)
                        slices_info_dict = self.patient_info_dic[pat_id][mod][spec]
                        sorted_slice_name_list = self.__sort_by_scan_date(slices_info_dict)
                        with open(os.path.join(folder_path,'img_label.txt'), 'w') as f:
                            for name in sorted_slice_name_list:
                                f.write(slices_info_dict[name]['file_path'])
                                f.write("\t")
                                f.write(slices_info_dict[name]['label_path'])
                                f.write("\n")
                                has_complete_label = has_complete_label if slices_info_dict[name]['label_path'] !='None' else False
                label_complete_str = 'annotation_complete' if has_complete_label else 'annotation_not_complete'
                fr.write(pat_id +'\t' + label_complete_str +'\n')





    def __sort_by_scan_date(self, info_dict):
        slices_name_list=[]
        slices_date_list= []
        for slice in info_dict:
            slices_name_list.append(info_dict[slice]['slice_name'])
            slices_date_list.append(info_dict[slice]['scan_date'])
        slices_name_np = np.array(slices_name_list)
        slices_date_np = np.array(slices_date_list)
        sorted_index = np.argsort(slices_date_np)
        slices_name_np = slices_name_np[sorted_index]
        return list(slices_name_np)


# #
# test = OASISDataPrepare()
# test.debug=False
# test.prepare_data()
# patients = Patients(full_init=True)
# filtered_patients = patients.get_filtered_patients_list(specificity='RIGHT',num_of_patients=3, len_time_range=[2,7], use_random=False)