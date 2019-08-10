import os
import pickle
import SimpleITK as sitk
from mermaid.utils import compute_warped_image
from mermaid.data_wrapper import AdaptVal, MyTensor
import numpy as np
from glob import glob

class Patient(object):
    def __init__(self,name):
        self.name = name
        self.nod_list = None
        self.nod_radius_list = None
        self.num_nod = None
    def set_nod_list(self, nod_list):
        self.nod_list= nod_list
        self.num_nod = len(nod_list)

    def set_nod_radius_list(self,nod_radius_list):
        self.nod_radius_list = nod_radius_list


class Patients(object):
    def __init__(self, mesh_pickle_path):
        self.mesh_pickle_path = mesh_pickle_path
        self.meshes = None
        self.patient_dict = {}
    def load_mesh(self):
        with open(mesh_pth, 'rb') as f:
            self.meshes = pickle.load(f)
        print("Num of {} meshes are loaded".format(len(self.meshes)))
    def init_patient(self):
        for patient_name,info in self.meshes:
            patient = Patient(patient_name)
            nod_radius_list = [nod[0] for nod in info]
            nod_list = [nod[1] for nod in info]
            patient.set_nod_list(nod_list)
            patient.set_nod_radius_list(nod_radius_list)
            self.patient_dict[patient_name] = patient

    def get_patient(self, patient_name):
        return self.patient_dict[patient_name]

    def set_warped_mesh_original(self,patient_name,warped_mesh_original):
        self.meshes[patient_name]['warped_mesh_original'] = warped_mesh_original

    def save_meshes(self):
        pass


class PatientPair(object):
    def __init__(self, source_name, target_name, inv_trans_path):
        self.source_name = source_name
        self.target_name = target_name
        self.pair_name = '{}_{}'.format(source_name,target_name)
        self.inv_trans_path = inv_trans_path

    def get_source_nod(self,source_patient):
        nod = source_patient.nod_list
        return nod



    def get_transform_map(self):
        map = sitk.ReadImage(self.inv_trans_path)
        map = sitk.GetArrayFromImage(map)
        img_sz = np.array(map.shape)
        map = MyTensor(map)
        return map, img_sz

    def normalize_mesh(self,nod, spacing):
        norm_nod = nod * spacing
        return norm_nod

    def warp_mesh(self, patient):
        nod = self.get_source_nod(patient)
        map, img_sz = self.get_transform_map()
        spacing = 1. / (img_sz - 1)
        norm_nod = normalize_mesh(nod, spacing)
        norm_nod = norm_nod.view([1, 1, -1, 1, 1])
        warped_mesh = compute_warped_image(norm_nod, map, spacing, spline_order=1, zero_boundary=False,
                                           use_01_input=True)
        warped_mesh_original = get_mesh_in_original_space(warped_mesh, spacing)
        return warped_mesh_original



def get_source_target_name_list(inv_folder, tag):
    f_path = os.path.join(inv_folder, '**', tag)
    inv_transform_list = glob(f_path,recursive=True)
    source_name_list = []
    target_name_list = []
    for inv_map_path in inv_transform_list:
        s, t = decompose_pair_name(inv_map_path)
        source_name_list.append(s)
        target_name_list.append(t)

    return source_name_list, target_name_list, inv_transform_list

def run_transform(inv_folder,tag,mesh_pickle_path):
    patients = Patients(mesh_pickle_path)
    file_list = get_source_target_name_list(inv_folder,tag)
    for source_name, target_name, inv_transform_path in file_list:
        patient_pair = PatientPair(source_name, target_name, inv_transform_path)
        source_patient = patients.get_patient(source_name)
        warped_mesh_original = patient_pair.warp_mesh(source_patient)
        patients.set_warped_mesh_original(source_name, warped_mesh_original)







def decompose_pair_name(inv_map_path):
    terms = inv_map_path.split('_')
    source_name = '{}_{}'.format(terms[0],terms[1])
    target_name = '{}_{}'.format(terms[2],terms[3])
    return source_name, target_name


def load_mesh(mesh_pth):
    mesh = pickle.load(mesh_pth)
    mesh = MyTensor(mesh)
    return mesh

def get_transform_map(map_pth):
    map = sitk.ReadImage(map_pth)
    map =  sitk.GetArrayFromImage(map)
    map = MyTensor(map)
    return map

def normalize_mesh(mesh, spacing):
    norm_mesh = mesh *spacing
    return norm_mesh

def get_mesh_in_original_space(mesh, spacing):
    mesh = mesh/spacing
    return mesh

def warp_mesh(mesh, map, img_sz):
    spacing = 1./(img_sz-1)
    norm_mesh = normalize_mesh(mesh,spacing)
    norm_mesh = norm_mesh.view([1,1,-1,1,1])
    warped_mesh = compute_warped_image(norm_mesh,map,spacing,spline_order=1, zero_boundary=False, use_01_input=True)
    warped_mesh_original = get_mesh_in_original_space(warped_mesh, spacing)
    return warped_mesh_original


def run_warp_mesh():
    pass


mesh_pth = '/data/zhengyang/mlsr/ima2_data008/ziming/ct_reg/padded_rescale_positive_nodule_dic.pickle'
