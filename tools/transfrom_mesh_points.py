import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''
import pickle
import nibabel as nib
from mermaid.utils import compute_warped_image_multiNC
from mermaid.data_wrapper import AdaptVal, MyTensor
import numpy as np
from glob import glob
from multiprocessing import *
import progressbar as pb
num_of_workers=12

class Patient(object):
    def __init__(self,name):
        self.name = name
        self.mesh_list = None
        self.num_mesh = None
    def set_mesh_list(self, mesh_list):
        self.mesh_list= mesh_list
        self.num_mesh = len(mesh_list)


    def get_mesh_list(self):
        return self.mesh_list


class Patients(object):
    def __init__(self, mesh_pickle_path):
        self.mesh_pickle_path = mesh_pickle_path
        self.meshes = None
        self.patient_dict = {}
        self.load_mesh()
        self.init_patient()

    def load_mesh(self):
        with open(self.mesh_pickle_path, 'rb') as f:
            self.meshes = pickle.load(f)
        print("Num of {} meshes are loaded".format(len(self.meshes)))
    def init_patient(self):
         for patient_name,info in self.meshes.items():
            patient = Patient(patient_name)
            mesh_list = [mesh[1] for mesh in info]
            patient.set_mesh_list(mesh_list)
            self.patient_dict[patient_name] = patient

    def get_patient(self, patient_name):
        return self.patient_dict[patient_name]

    def save_meshes(self):
        pass


class PatientPair(object):
    def __init__(self, source_name, target_name, inv_trans_path):
        self.source_name = source_name
        self.target_name = target_name
        self.pair_name = '{}_{}'.format(source_name,target_name)
        self.inv_trans_path = inv_trans_path

    def get_pair_name(self):
        return self.pair_name



    def get_transform_map(self):

        # map = sitk.ReadImage(self.inv_trans_path)
        # map = sitk.GetArrayFromImage(map)
        inv_map = nib.load(self.inv_trans_path)
        inv_map = inv_map.get_fdata()
        img_sz = np.array(inv_map.shape[1:])
        map = MyTensor(inv_map[None])
        return map, img_sz

    def normalize_mesh(self,mesh_list, spacing):
        norm_mesh = [mesh * spacing for mesh in mesh_list]
        return norm_mesh

    def get_mesh_in_original_space(self,mesh, spacing):
        mesh = mesh / spacing
        return mesh

    def warp_mesh(self, source_patient):
        mesh_list = source_patient.get_mesh_list()
        inv_map, img_sz = self.get_transform_map()
        spacing = 1. / (img_sz - 1)
        norm_mesh_list = self.normalize_mesh(mesh_list, spacing)
        norm_mesh_np = np.array(norm_mesh_list) # N*3
        norm_mesh_np = np.transpose(norm_mesh_np) # 3*N
        norm_mesh = MyTensor(norm_mesh_np).view([1, 3, -1, 1, 1])*2-1
        warped_mesh = compute_warped_image_multiNC(inv_map, norm_mesh, spacing, spline_order=1, zero_boundary=False,
                                           use_01_input=False)

        warped_mesh_np = warped_mesh.cpu().numpy()[0,:,:,0,0]
        warped_mesh_np = np.transpose(warped_mesh_np)# N*3
        warped_mesh_orig_np = self.get_mesh_in_original_space(warped_mesh_np, spacing)
        warped_mesh_original_list = [warped_mesh_orig_np[i] for i in range(len(mesh_list))]
        return warped_mesh_original_list



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


def __run_transform(file_list,pair_mesh,patients):
    pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(file_list)).start()
    count = 0
    for source_name, target_name, inv_transform_path in file_list:
        patient_pair = PatientPair(source_name, target_name, inv_transform_path)
        pair_name = patient_pair.get_pair_name()
        source_patient = patients.get_patient(source_name)
        warped_mesh_original = patient_pair.warp_mesh(source_patient)
        pair_mesh[pair_name] = warped_mesh_original
        count += 1
        pbar.update(count)
    pbar.finish()


def run_transform(inv_folder,tag,mesh_pickle_path, pair_warped_mesh_pth):
    patients = Patients(mesh_pickle_path)
    file_list = get_source_target_name_list(inv_folder,tag)
    manager = Manager()
    pair_mesh = manager.dict()
    file_list_comb = [item for item in zip(*file_list)]
    split_list = np.array_split(file_list_comb, num_of_workers)
    procs = []
    for i in range(num_of_workers):
        p = Process(target=__run_transform, args=(split_list[i], pair_mesh,patients))
        p.start()
        print("pid:{} start:".format(p.pid))
        procs.append(p)
    for p in procs:
        p.join()
    pair_mesh = dict(pair_mesh)

    with open(pair_warped_mesh_pth,'wb') as f:
        pickle.dump(pair_mesh, f)





def decompose_pair_name(inv_map_path):
    terms = os.path.split(inv_map_path)[1].split('_')
    source_name = '{}_{}'.format(terms[0],terms[1])
    target_name = '{}_{}'.format(terms[2],terms[3])
    return source_name, target_name



def run_warp_mesh():
    mesh_pth = '/data/zhengyang/mlsr/ima2_data008/ziming/ct_reg/padded_rescale_positive_nodule_dic.pickle'
    inv_folder = '/data/zhengyang/mlsr/ima2_data008/zyshen/ct_reg/output/reg/res/records/original_sz'
    pair_warped_mesh_pth = '/data/zhengyang/mlsr/ima2_data008/zyshen/ct_reg/output/reg/padded_rescale_positive_nodule_dic_warped.pickle'
    tag = '*inv_phi.nii.gz'
    run_transform(inv_folder,tag,mesh_pth, pair_warped_mesh_pth)


run_warp_mesh()