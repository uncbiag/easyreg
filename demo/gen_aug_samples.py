"""
the first demo is to input a txt file
each line includes the moving image, target image
the output should be the warped image, the momentum
( we would remove the first demo into easyreg package)


the second demo is to input a txt file
each line include the moving image, label,  momentum1, momentum2,...
the output should be the the warped image ( by random momentum, by random t) and the corresponding warped label
"""
""""
demo on RDMM on 2d synthetic image registration
"""
import matplotlib as matplt
matplt.use('Agg')
import sys,os
os.environ["CUDA_VISIBLE_DEVICES"] = ''
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../easyreg'))
import random
import torch
from mermaid.model_evaluation import evaluate_model
import mermaid.module_parameters as pars
from mermaid.utils import resample_image, compute_warped_image_multiNC, \
    get_resampled_image,get_res_size_from_size, get_res_spacing_from_spacing,identity_map_multiN
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from tools.image_rescale import save_image_with_given_reference
from easyreg.aug_utils import read_img_label_into_list
from easyreg.utils import gen_affine_map, get_inverse_affine_param
from glob import glob
import copy



def get_pair_list(txt_pth):
    moving_momentum_path_list = read_img_label_into_list(txt_pth)
    return moving_momentum_path_list

def get_init_weight_list(folder_path):
    weight_path = os.path.join(folder_path,'pair_weight_path_list.txt')
    init_weight_path = read_img_label_into_list(weight_path)
    return init_weight_path


def get_setting(path,output_path,setting_name = "mermaid"):
    params = pars.ParameterDict()
    params.load_JSON(path)
    os.makedirs(output_path,exist_ok=True)
    output_path = os.path.join(output_path,'{}_setting.json'.format(setting_name))
    params.write_JSON(output_path,save_int=False)
    return params




def save_deformation(phi,output_path,fname_list):
    phi_np = phi.detach().cpu().numpy()
    for i in range(phi_np.shape[0]):
        phi = nib.Nifti1Image(phi_np[i], np.eye(4))
        nib.save(phi, os.path.join(output_path,fname_list[i]+'.nii.gz'))




def get_file_name(file_path,last_ocur=True):
    if not last_ocur:
        name= os.path.split(file_path)[1].split('.')[0]
    else:
        name = os.path.split(file_path)[1].rsplit('.',1)[0]
    name = name.replace('.nii','')
    name = name.replace('.','d')
    return name

class DataAug(object):
    def __init__(self,aug_setting_path):
        self.aug_setting_path = aug_setting_path
        self.aug_setting = get_setting(aug_setting_path,"aug")
        self.max_aug_num = self.aug_setting['data_aug'][('max_aug_num',1000,"the max num of rand aug, only set when in rand_aug mode")]




class FluidAug(DataAug):
    def __init__(self,aug_setting_path,mermaid_setting_path):
        DataAug.__init__(self,aug_setting_path)
        self.mermaid_setting_path = mermaid_setting_path
        self.mermaid_setting = get_setting(mermaid_setting_path,"mermaid")
        self.init_setting()


    def init_setting(self):
        aug_setting = self.aug_setting
        self.K = aug_setting['data_aug']["fluid_aug"][('K',1,"the dimension of the geodeisc subspace")]
        self.task_type = aug_setting['data_aug']["fluid_aug"][('task_type',"rand_aug","rand_aug/data_interp,  rand_aug: random data augmentation; data_interp: data interpolation")]
        self.compute_inverse = aug_setting['data_aug']["fluid_aug"][('compute_inverse',True,"compute the inverse map")]
        self.rand_w_t = True if self.task_type=="rand_aug" else False
        self.t_aug_list= aug_setting['data_aug']["fluid_aug"]['data_interp'][('t_aug_list',[1.0],"the time points for inter-/extra-polation")]
        self.weight_list = self.aug_setting['data_aug']["fluid_aug"]['data_interp'][('weight_list',[[1.0]],"the weight for each target image, set in data_interp mode")]
        self.t_range = aug_setting['data_aug']["fluid_aug"]['rand_aug'][('t_range',[-1,2],"the range of t inter-/extra-polation, the registration completes in unit time [0,1]")]
        self.rand_momentum_scale = self.aug_setting['data_aug']["fluid_aug"]['rand_aug'][('rand_momentum_scale',8,"the size of random momentum is 1/rand_momentum_scale of the original image sz")]
        self.magnitude = self.aug_setting['data_aug']["fluid_aug"]['rand_aug'][('magnitude',1.5,"the magnitude of the random momentum")]
        self.affine_back_to_original_postion = self.aug_setting['data_aug']["fluid_aug"]['aug_with_nonaffined_data'][('affine_back_to_original_postion',False,"transform the new image to the original postion")]



    def generate_aug_data(self,*args):
        pass


    def generate_single_res(self,moving, l_moving, momentum, init_weight, initial_map, initial_inverse_map, fname, t_aug, output_path, moving_path):
        params = self.mermaid_setting
        params['model']['registration_model']['forward_model']['tTo'] = t_aug

        # here we assume the momentum is computed at low_resol_factor=0.5
        if momentum is not None:
            input_img_sz = [1, 1] + [int(sz * 2) for sz in momentum.shape[2:]]
        else:
            input_img_sz = list(moving.shape)
            momentum_sz_low = [1, 3] + [int(dim /self.rand_momentum_scale) for dim in input_img_sz[2:]]
            momentum_sz = [1, 3] + [int(dim / 2) for dim in input_img_sz[2:]]
            momentum = (np.random.rand(*momentum_sz_low) * 2 - 1) * self.magnitude
            mom_spacing = 1./(np.array(momentum_sz_low[2:])-1)
            momentum = torch.Tensor(momentum)
            momentum, _ = resample_image(momentum,mom_spacing,momentum_sz,spline_order=1,zero_boundary=True)

        org_spacing = 1.0 / (np.array(moving.shape[2:]) - 1)
        input_spacing = 1.0 / (np.array(input_img_sz[2:]) - 1)
        size_diff = not input_img_sz == list(moving.shape)
        if size_diff:
            input_img, _ = resample_image(moving, org_spacing, input_img_sz)
        else:
            input_img = moving
        low_initial_map = None
        low_init_inverse_map = None
        if initial_map is not None:
            low_initial_map, _ = resample_image(initial_map, input_spacing, [1, 3] + list(momentum.shape[2:]))
        if initial_inverse_map is not None:
            low_init_inverse_map, _ = resample_image(initial_inverse_map, input_spacing, [1, 3] + list(momentum.shape[2:]))
        individual_parameters = dict(m=momentum, local_weights=init_weight)
        sz = np.array(input_img.shape)
        extra_info = None
        visual_param = None
        res = evaluate_model(input_img, input_img, sz, input_spacing,
                             use_map=True,
                             compute_inverse_map=self.compute_inverse,
                             map_low_res_factor=0.5,
                             compute_similarity_measure_at_low_res=False,
                             spline_order=1,
                             individual_parameters=individual_parameters,
                             shared_parameters=None, params=params, extra_info=extra_info, visualize=False,
                             visual_param=visual_param, given_weight=False,
                             init_map=initial_map, lowres_init_map=low_initial_map,
                             init_inverse_map=initial_inverse_map,lowres_init_inverse_map=low_init_inverse_map)
        phi = res[1]
        phi_new = phi
        if size_diff:
            phi_new, _ = resample_image(phi, input_spacing, [1, 3] + list(moving.shape[2:]))
        if initial_inverse_map is not None and self.affine_back_to_original_postion:
            phi_new = compute_warped_image_multiNC(phi_new, initial_inverse_map, org_spacing, spline_order=1)
        warped = compute_warped_image_multiNC(moving, phi_new, org_spacing, spline_order=1, zero_boundary=True)
        save_image_with_given_reference(warped, [moving_path], output_path, [fname + '_image'])
        if l_moving is not None:
            l_warped = compute_warped_image_multiNC(l_moving, phi_new, org_spacing, spline_order=0, zero_boundary=True)
            save_image_with_given_reference(l_warped, [moving_path], output_path, [fname + '_label'])
        save_deformation(phi, output_path, [fname + '_phi_map'])
        if self.compute_inverse:
            phi_inv = res[2]
            if self.affine_back_to_original_postion:
                print("Cannot compute the inverse map when affine back to the source image position")
                return
            if size_diff:
                inv_phi_new, _ = resample_image(phi_inv, input_spacing, [1, 3] + list(moving.shape[2:]))
            save_deformation(phi_inv, output_path, [fname + '_inv_map'])

class FluidRand(FluidAug):
    def __init__(self,aug_setting_path,mermaid_setting_path):
        FluidAug.__init__(self,aug_setting_path,mermaid_setting_path)

    def get_input(self,moving_path_list, init_weight_path_list=None):
        """ each line include the path of moving, the path of label (None if not exist), path of momentum1, momentum2...."""

        fr_sitk = lambda x: torch.Tensor(sitk.GetArrayFromImage(sitk.ReadImage(x)))
        moving = fr_sitk(moving_path_list[0])[None][None]
        l_moving = None
        if moving_path_list[1] is not None:
            l_moving = fr_sitk(moving_path_list[1])[None][None]
        moving_name =get_file_name(moving_path_list[0])
        return moving, l_moving, moving_name

    def generate_aug_data(self,moving_path_list, init_weight_path_list, output_path):
        max_aug_num  = self.max_aug_num
        t_range = self.t_range
        t_span = t_range[1]-t_range[0]

        num_pair = len(moving_path_list)
        for i in range(num_pair):
            moving_path = moving_path_list[i][0]
            moving, l_moving, moving_name = self.get_input(moving_path_list[i], None)
            num_aug = round(max_aug_num / num_pair)
            for _ in range(num_aug):
                t_aug = random.random() * t_span +t_range[0]
                momentum = None
                fname = moving_name + '_{:.4f}_t_{:.2f}'.format(random.random(), t_aug)
                self.generate_single_res(moving, l_moving, momentum, None, None,None, fname, t_aug, output_path, moving_path)


class FluidAffined(FluidAug):
    def __init__(self,aug_setting_path,mermaid_setting_path):
        FluidAug.__init__(self,aug_setting_path,mermaid_setting_path)



    def get_input(self,moving_momentum_path_list, init_weight_path_list):
        """ each line include the path of moving, the path of label (None if not exist), path of momentum1, momentum2...."""

        fr_sitk = lambda x: torch.Tensor(sitk.GetArrayFromImage(sitk.ReadImage(x)))
        moving = fr_sitk(moving_momentum_path_list[0])[None][None]
        l_moving = None
        if moving_momentum_path_list[1] is not None:
            l_moving = fr_sitk(moving_momentum_path_list[1])[None][None]
        momentum_list = [np.transpose(fr_sitk(path))[None] for path in moving_momentum_path_list[2:]]

        if init_weight_path_list is not None:
            init_weight_list = [[fr_sitk(path) for path in init_weight_path_list]]
        else:
            init_weight_list = None
        fname_list = [get_file_name(path) for path in moving_momentum_path_list[2:]]
        fname_list = [fname.replace("_0000_Momentum", '') for fname in fname_list]
        return moving, l_moving, momentum_list, init_weight_list, fname_list

    def generate_aug_data(self,moving_momentum_path_list, init_weight_path_list, output_path):
        max_aug_num  = self.max_aug_num
        rand_w_t = self.rand_w_t
        t_range = self.t_range
        K = self.K
        t_span = t_range[1]-t_range[0]

        num_pair = len(moving_momentum_path_list)

        for i in range(num_pair):
            moving_path = moving_momentum_path_list[i][0]
            moving, l_moving, momentum_list, init_weight_list, fname_list = self.get_input(
                moving_momentum_path_list[i], init_weight_path_list[i] if init_weight_path_list else None)
            num_aug = round(max_aug_num / num_pair) if rand_w_t else 1
            for _ in range(num_aug):
                num_momentum = len(momentum_list)
                if rand_w_t:
                    t_aug_list = [random.random() * t_span +t_range[0]]
                    weight = np.array([random.random() for _ in range(K)])
                    weight_list = [weight / np.sum(weight)]
                    selected_index = random.sample(list(range(num_momentum)), K)
                else:
                    t_aug_list = self.t_aug_list
                    weight_list = self.weight_list
                    K = num_momentum
                    selected_index = list(range(num_momentum))
                    for weight in weight_list:
                        assert len(weight)==num_momentum,"In data-interp mode, the weight should be the same size of the momentum set"
                for t_aug in t_aug_list:
                    for weight in weight_list:
                        momentum = torch.zeros_like(momentum_list[0])
                        fname = ""
                        suffix =""
                        for k in range(K):
                            momentum += weight[k] * momentum_list[selected_index[k]]
                            fname += fname_list[selected_index[k]] + '_'
                            suffix += '{:.4f}_'.format(weight[k])

                        fname = fname + suffix +'t_{:.2f}'.format(t_aug)
                        fname = fname.replace('.', 'd')
                        init_weight = None
                        if init_weight_list is not None:
                            init_weight = random.sample(init_weight_list, 1)

                        self.generate_single_res(moving, l_moving, momentum, init_weight, None,None, fname, t_aug, output_path, moving_path)



class FluidNonAffined(FluidAug):
    def __init__(self,aug_setting_path,mermaid_setting_path):
        FluidAug.__init__(self,aug_setting_path,mermaid_setting_path)

    def read_affine_param_and_output_map(self,affine_param_path,img_sz):
        affine_param = np.load(affine_param_path)
        affine_param = torch.Tensor(affine_param)[None]
        affine_map = gen_affine_map(affine_param,img_sz)
        inverse_affine_param = get_inverse_affine_param(affine_param)
        inverse_affine_map = gen_affine_map(inverse_affine_param,img_sz)
        affine_map = (affine_map+1.)/2
        inverse_affine_map = (inverse_affine_map+1.)/2
        return affine_map, inverse_affine_map


    def get_input(self,moving_momentum_path_list, init_weight_path_list):
        """
        each line includes  path of moving, path of moving label(None if not exists), path of mom_1,...mom_m, affine_1....affine_m
        """

        fr_sitk = lambda x: torch.Tensor(sitk.GetArrayFromImage(sitk.ReadImage(x)))

        moving = fr_sitk(moving_momentum_path_list[0])[None][None]
        l_moving = None
        if moving_momentum_path_list[1] is not None:
            l_moving = fr_sitk(moving_momentum_path_list[1])[None][None]
        num_m = int((len(moving_momentum_path_list)-2)/2)
        momentum_list =[fr_sitk(path).permute(3,2,1,0)[None] for path in moving_momentum_path_list[2:num_m+2]]
        #affine_list =[fr_sitk(path).permute(3,2,1,0)[None] for path in moving_momentum_path_list[num_m+2:]]
        affine_forward_inverse_list =[self.read_affine_param_and_output_map(path,moving.shape[2:]) for path in moving_momentum_path_list[num_m+2:]]
        affine_list = [forward_inverse[0] for forward_inverse in affine_forward_inverse_list]
        inverse_affine_list = [forward_inverse[1] for forward_inverse in affine_forward_inverse_list]

        if init_weight_path_list is not None:
            init_weight_list=[[fr_sitk(path) for path in init_weight_path_list]]
        else:
            init_weight_list=None
        fname_list =[get_file_name(path) for path in moving_momentum_path_list[2:num_m+2]]
        fname_list = [fname.replace("_0000_Momentum",'') for fname in fname_list]
        return moving, l_moving, momentum_list, init_weight_list, affine_list,inverse_affine_list, fname_list

    def generate_aug_data(self,moving_momentum_path_list, init_weight_path_list, output_path):

        max_aug_num = self.max_aug_num
        rand_w_t = self.rand_w_t
        t_range = self.t_range
        K = 1 # for non-affined case, the K is set to 1
        t_span = t_range[1] - t_range[0]

        num_pair = len(moving_momentum_path_list)
        for i in range(num_pair):
            moving_path = moving_momentum_path_list[i][0]
            moving, l_moving, momentum_list, init_weight_list, affine_list,inverse_affine_list, fname_list = self.get_input(
                moving_momentum_path_list[i], init_weight_path_list[i] if init_weight_path_list else None)
            num_aug = max(round(max_aug_num / num_pair),1) if rand_w_t else 1
            for _ in range(num_aug):
                num_momentum = len(momentum_list)
                if rand_w_t:
                    t_aug_list = [random.random() * t_span + t_range[0]]
                    selected_index = random.sample(list(range(num_momentum)), K)
                else:
                    assert num_momentum ==1,"for non-affined image and for data_interp mode, the size of the momentum set should be 1"
                    t_aug_list = self.t_aug_list
                    selected_index = [0]

                for t_aug in t_aug_list:
                    momentum  = momentum_list[selected_index[0]]
                    affine = affine_list[selected_index[0]]
                    inverse_affine = inverse_affine_list[selected_index[0]]
                    fname = fname_list[selected_index[0]] + '_t_{:.2f}'.format(t_aug)

                    fname = fname.replace('.', 'd')
                    init_weight = None
                    if init_weight_list is not None:
                        init_weight = random.sample(init_weight_list, 1)

                    self.generate_single_res(moving, l_moving, momentum, init_weight, affine,inverse_affine, fname, t_aug, output_path, moving_path)




class FluidAtlas(FluidAug):
    def __init__(self,aug_setting_path,mermaid_setting_path):
        FluidAug.__init__(self,aug_setting_path,mermaid_setting_path)
        self.to_atlas_folder = self.aug_setting['data_aug']["fluid_aug"]["aug_with_atlas"][
            ('to_atlas_folder', None, "the folder containing the image to atlas transformation")]
        self.atlas_to_folder = self.aug_setting['data_aug']["fluid_aug"]["aug_with_atlas"][
            ('atlas_to_folder', None, "the folder containing the atlas to image momentum")]

    def get_input(self,moving_momentum_path_list, init_weight_path_list):
        """
        each line include the path of moving, the path of label(None if not exists)
        :return:
        """

        fr_sitk = lambda x: torch.Tensor(sitk.GetArrayFromImage(sitk.ReadImage(x)))
        moving = fr_sitk(moving_momentum_path_list[0])[None][None]
        l_moving = None
        if moving_momentum_path_list[1] is not None:
            l_moving = fr_sitk(moving_momentum_path_list[1])[None][None]
        moving_name = get_file_name(moving_momentum_path_list[0])
        return moving, l_moving,moving_name

    def generate_aug_data(self,path_list, init_weight_path_list, output_path):
        """
        here we use the low-interface of mermaid to get efficient low-res- propagration (avod saving phi and inverse phi as well as the precision loss from unnecessary upsampling and downsampling
        ) which provide high precision in maps
        """

        def create_mermaid_model(mermaid_json_pth, img_sz, compute_inverse):
            import mermaid.model_factory as py_mf
            spacing = 1. / (np.array(img_sz[2:]) - 1)
            params = pars.ParameterDict()
            params.load_JSON(mermaid_json_pth)  # ''../easyreg/cur_settings_svf.json')
            model_name = params['model']['registration_model']['type']
            params.print_settings_off()
            mermaid_low_res_factor = 0.5
            lowResSize = get_res_size_from_size(img_sz, mermaid_low_res_factor)
            lowResSpacing = get_res_spacing_from_spacing(spacing, img_sz, lowResSize)
            ##
            mf = py_mf.ModelFactory(img_sz, spacing, lowResSize, lowResSpacing)
            model, criterion = mf.create_registration_model(model_name, params['model'],
                                                            compute_inverse_map=True)
            lowres_id = identity_map_multiN(lowResSize, lowResSpacing)
            lowResIdentityMap = torch.from_numpy(lowres_id)

            _id = identity_map_multiN(img_sz, spacing)
            identityMap = torch.from_numpy(_id)
            mermaid_unit_st = model
            mermaid_unit_st.associate_parameters_with_module()
            return mermaid_unit_st, criterion, lowResIdentityMap, lowResSize, lowResSpacing, identityMap, spacing

        def _set_mermaid_param(mermaid_unit, m):
            mermaid_unit.m.data = m

        def _do_mermaid_reg(mermaid_unit, low_phi, m, low_s=None, low_inv_phi=None):
            with torch.no_grad():
                _set_mermaid_param(mermaid_unit, m)
                low_phi = mermaid_unit(low_phi, low_s, phi_inv=low_inv_phi)
            return low_phi

        def get_momentum_name(momentum_path):
            fname = get_file_name(momentum_path)
            fname = fname.replace("_0000_Momentum", '')
            return fname

        max_aug_num = self.max_aug_num
        rand_w_t = self.rand_w_t
        t_range = self.t_range
        t_span = t_range[1]-t_range[0]
        K = self.K

        num_pair = len(path_list)
        assert init_weight_path_list is None, "init weight has not supported yet"
        # load all momentums for atlas to images
        read_image = lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x))
        atlas_to_momentum_path_list = list(filter(lambda x: "Momentum" in x and get_file_name(x).find("atlas") == 0,
                                                  glob(os.path.join(self.atlas_to_folder, "*nii.gz"))))
        to_atlas_momentum_path_list = list(filter(lambda x: "Momentum" in x and get_file_name(x).find("atlas") != 0,
                                                  glob(os.path.join(self.to_atlas_folder, "*nii.gz"))))
        atlas_to_momentum_list = [torch.Tensor(read_image(atlas_momentum_pth).transpose()[None]) for atlas_momentum_pth
                                  in atlas_to_momentum_path_list]
        to_atlas_momentum_list = [torch.Tensor(read_image(atlas_momentum_pth).transpose()[None]) for atlas_momentum_pth
                                  in to_atlas_momentum_path_list]
        moving_example = read_image(path_list[0][0])
        img_sz = list(moving_example.shape)
        mermaid_unit_st, criterion, lowResIdentityMap, lowResSize, lowResSpacing, identityMap, spacing = create_mermaid_model(
            mermaid_setting_path, [1, 1] + img_sz, self.compute_inverse)

        for i in range(num_pair):
            moving, l_moving,moving_name = self.get_input(path_list[i], None)
            fname = moving_name
            # get the transformation to atlas, which should simply load the transformation map
            low_moving = get_resampled_image(moving, None, lowResSize, 1, zero_boundary=True,
                                             identity_map=lowResIdentityMap)
            init_map = lowResIdentityMap.clone()
            init_inverse_map = lowResIdentityMap.clone()
            index = list(filter(lambda x: fname in x, to_atlas_momentum_path_list))[0]
            index = to_atlas_momentum_path_list.index(index)
            # here we only interested in forward the map, so the moving image doesn't affect
            mermaid_unit_st.integrator.cparams['tTo'] = 1.0
            low_phi_to_atlas, low_inverse_phi_to_atlas = _do_mermaid_reg(mermaid_unit_st, init_map,
                                                                         to_atlas_momentum_list[index], low_moving,
                                                                         low_inv_phi=init_inverse_map)
            num_aug = max(round(max_aug_num / num_pair),1) if rand_w_t else 1

            for _ in range(num_aug):
                num_momentum = len(atlas_to_momentum_list)
                if rand_w_t:
                    t_aug_list = [random.random() * t_span + t_range[0]]
                    weight = np.array([random.random() for _ in range(K)])
                    weight_list = [weight / np.sum(weight)]
                    selected_index = random.sample(list(range(num_momentum)), K)
                else:
                   raise ValueError("In atlas augmentation mode, the data interpolation is disabled")
                for t_aug in t_aug_list:
                    if t_aug ==0:
                        continue
                    for weight in weight_list:
                        momentum = torch.zeros_like(atlas_to_momentum_list[0])
                        fname = ""
                        suffix = ""
                        for k in range(K):
                            momentum += weight[k] * atlas_to_momentum_list[selected_index[k]]
                            fname += get_momentum_name(atlas_to_momentum_path_list[selected_index[k]]) + '_'
                            suffix += '{:.4f}_'.format(weight[k])

                        fname = fname + suffix + 't_{:.2f}'.format(t_aug)
                        fname = fname.replace('.', 'd')

                        mermaid_unit_st.integrator.cparams['tTo'] = t_aug
                        low_phi_atlas_to, low_inverse_phi_atlas_to = _do_mermaid_reg(mermaid_unit_st, low_phi_to_atlas.clone(),
                                                                                     momentum, low_moving,
                                                                                     low_inv_phi=low_inverse_phi_to_atlas.clone())
                        foward_map = get_resampled_image(low_phi_atlas_to, lowResSpacing, [1, 3] + img_sz, 1,
                                                         zero_boundary=False,
                                                         identity_map=identityMap)
                        inverse_map = get_resampled_image(low_inverse_phi_atlas_to, lowResSpacing, [1, 3] + img_sz, 1,
                                                          zero_boundary=False,
                                                          identity_map=identityMap)
                        warped = compute_warped_image_multiNC(moving, foward_map, spacing, spline_order=1, zero_boundary=True)
                        if l_moving is not None:
                            l_warped = compute_warped_image_multiNC(l_moving, foward_map, spacing, spline_order=0,
                                                                    zero_boundary=True)
                            save_image_with_given_reference(l_warped, [path_list[i][0]], output_path, [fname + '_label'])
                        save_image_with_given_reference(warped, [path_list[i][0]], output_path, [fname + '_image'])
                        if self.compute_inverse:
                            # save_deformation(foward_map, output_path, [fname + '_phi'])
                            save_deformation(inverse_map, output_path, [fname + '_inv_phi'])






class RandomBSplineTransform(object):
    """
    Apply random BSpline Transformation to a 3D image
    check https://itk.org/Doxygen/html/classitk_1_1BSplineTransform.html for details of BSpline Transform
    """

    def __init__(self,mesh_size=(3,3,3), bspline_order=2, deform_scale=1.0, ratio=0.5, interpolator=sitk.sitkLinear,
                 random_mode = 'Normal'):
        self.mesh_size = mesh_size
        self.bspline_order = bspline_order
        self.deform_scale = deform_scale
        self.ratio = ratio  # control the probability of conduct transform
        self.interpolator = interpolator
        self.random_mode = random_mode

    def resample(self,image, transform, interpolator=sitk.sitkBSpline, default_value=0.0):
        """Resample a transformed image"""
        reference_image = image
        return sitk.Resample(image, reference_image, transform,
                             interpolator, default_value)

    def __call__(self, sample):
        random_state = np.random.RandomState()

        if np.random.rand(1)[0] < self.ratio:
            img_tm, seg_tm = sample['image'], sample['label']
            img = sitk.GetImageFromArray(sitk.GetArrayFromImage(img_tm).copy())
            img.CopyInformation(img_tm)
            seg = sitk.GetImageFromArray(sitk.GetArrayFromImage(seg_tm).copy())
            seg.CopyInformation(seg_tm)

            # initialize a bspline transform
            bspline = sitk.BSplineTransformInitializer(img, self.mesh_size, self.bspline_order)

            # generate random displacement for control points, the deformation is scaled by deform_scale
            if self.random_mode == 'Normal':
                control_point_displacements = random_state.normal(0, self.deform_scale/2, len(bspline.GetParameters()))
            elif self.random_mode == 'Uniform':
                control_point_displacements = random_state.random(len(bspline.GetParameters())) * self.deform_scale

            #control_point_displacements[0:int(len(control_point_displacements) / 3)] = 0  # remove z displacement
            bspline.SetParameters(control_point_displacements)

            # deform and resample image
            img_trans = self.resample(img, bspline, interpolator=self.interpolator, default_value=0.01)
            seg_trans = self.resample(seg, bspline, interpolator=sitk.sitkNearestNeighbor, default_value=0)
            new_sample = {}

            new_sample['image'] = img_trans
            new_sample['label'] = seg_trans
        else:
            new_sample = sample

        return new_sample


class BsplineAug(DataAug):
    def __init__(self, aug_setting_path):
        DataAug.__init__(self, aug_setting_path)
        self.mesh_size_list = self.aug_setting['data_aug']["bspline_aug"][("mesh_size_list",[[10,10,10]],"mesh size ")]
        self.deform_scale_list = self.aug_setting['data_aug']["bspline_aug"][("deform_scale_list",[3],"deform scale")]
        self.aug_ratio = self.aug_setting['data_aug']["bspline_aug"][("aug_ratio",0.95,
        "chance to deform the image, i.e., 0.5 refers to ratio of the deformed images and the non-deformed (original) image")]
        assert len(self.mesh_size_list) == len(self.deform_scale_list)

    def get_input(self,moving_path_list):
        moving = [sitk.ReadImage(pth[0]) for pth in moving_path_list]
        l_moving = [sitk.ReadImage(pth[1]) for pth in moving_path_list]
        fname_list = [get_file_name(pth[0]) for pth in moving_path_list]
        return moving, l_moving, fname_list

    def generate_aug_data(self, moving_path_list, output_path):
        num_pair = len(moving_path_list)
        num_aug = int(self.max_aug_num / num_pair)
        moving_list, l_moving_list, fname_list = self.get_input(moving_path_list)
        bspline_func_list = [RandomBSplineTransform(mesh_size=self.mesh_size_list[i], bspline_order=2, deform_scale=self.deform_scale_list[i], ratio=self.aug_ratio)
                             for i in range(len(self.mesh_size_list))]

        for i in range(num_pair):
            sample = {'image': moving_list[i], 'label': l_moving_list[i]}
            for _ in range(num_aug):
                bspline_func = random.sample(bspline_func_list, 1)
                aug_sample = bspline_func[0](sample)
                fname = fname_list[i] + '_{:.4f}'.format(random.random())
                fname = fname.replace('.', 'd')
                sitk.WriteImage(aug_sample['image'], os.path.join(output_path, fname + '_image.nii.gz'))
                sitk.WriteImage(aug_sample['label'], os.path.join(output_path, fname + '_label.nii.gz'))


def generate_aug_data(moving_momentum_path_list,init_weight_path_list,output_path, mermaid_setting_path,fluid_mode,aug_setting_path,fluid_aug=True):
    if fluid_aug:
        if fluid_mode=='aug_with_affined_data':
            fluid_aug = FluidAffined(aug_setting_path,mermaid_setting_path)
        elif fluid_mode=='aug_with_nonaffined_data':
            fluid_aug = FluidNonAffined(aug_setting_path,mermaid_setting_path)
        elif fluid_mode== "aug_with_atlas":
            fluid_aug = FluidAtlas(aug_setting_path,mermaid_setting_path)
        elif fluid_mode=='rand_aug':
            fluid_aug = FluidRand(aug_setting_path,mermaid_setting_path)
        else:
            raise ValueError("not supported mode, should be aug_with_affined_data/aug_with_nonaffined_data/aug_with_atlas/rand_aug")
        fluid_aug.generate_aug_data(moving_momentum_path_list, init_weight_path_list, output_path)

    else:
        moving_path_list = moving_momentum_path_list
        bspline_aug = BsplineAug(aug_setting_path)
        bspline_aug.generate_aug_data(moving_path_list, output_path)











if __name__ == '__main__':
    """
    Two data augmentation methods are supported
    1) lddmm shooting
    2) bspline random transformation
    
    
    since we relax the affine constrain here, the augmentation only works under 1d geodesic subspace
    
    For the input txt file, each line include a source image, N target image

    
    """
    import argparse
    # --txt_path=/playpen-raid/zyshen/data/lpba_reg/train_with_5/lpba_ncc_reg1/momentum_lresol.txt  --aug_setting_path=/playpen-raid/zyshen/reg_clean/demo/demo_settings/data_aug/opt_lddmm_lpba/data_aug_setting.json --mermaid_setting_path=/playpen-raid/zyshen/reg_clean/demo/demo_settings/data_aug/opt_lddmm_lpba/mermaid_nonp_settings.json --output_path=/playpen-raid/zyshen/debug/debug_aug
    # --txt_path=/playpen-raid1/zyshen/data/reg_oai_aug/momentum_lresol.txt --aug_setting_path=/playpen-raid/zyshen/reg_clean/demo/demo_settings/data_aug/opt_lddmm_lpba/data_aug_setting.json --mermaid_setting_path=/playpen-raid/zyshen/reg_clean/demo/demo_settings/data_aug/opt_lddmm_lpba/mermaid_nonp_settings.json --output_path=/playpen-raid/zyshen/debug/debug_aug
    # --txt_path=/playpen-raid/zyshen/data/lpba_seg_resize/baseline/25case/train/file_path_list.txt --aug_setting_path=/playpen-raid/zyshen/reg_clean/demo/demo_settings/data_aug/opt_lddmm_lpba/data_aug_setting.json --mermaid_setting_path=/playpen-raid/zyshen/reg_clean/demo/demo_settings/data_aug/opt_lddmm_lpba/mermaid_nonp_settings.json --output_path=/playpen-raid1/zyshen/debug/debug_aug3
    parser = argparse.ArgumentParser(description='Registration demo for data augmentation')
    parser.add_argument("-txt",'--txt_path', required=False, default=None,
                        help='the file path of input txt, exclusive with random_m')
    parser.add_argument("-w",'--rdmm_preweight_txt_path', required=False, default=None,
                        help='the file path of rdmm preweight txt, only needed when use predefined rdmm model,(need to further test)')
    parser.add_argument("-m",'--fluid_mode',required=False,default=None,
                        help='aug_with_affined_data/aug_with_nonaffined_data/aug_with_atlas/rand_aug')
    parser.add_argument('--bspline', required=False, action='store_true',
                        help='data augmentation with bspline, exclusive random_m, rdmm_preweight_txt_path,compute_inverse')
    parser.add_argument("-o",'--output_path', required=False, default='./rdmm_synth_data_generation/data_task',
                        help='the path of task folder')
    parser.add_argument("-as",'--aug_setting_path', required=False, default=None,
                        help='path of data augmentation setting json')
    parser.add_argument("-ms",'--mermaid_setting_path', required=False, default=None,
                        help='path of mermaid setting json')
    args = parser.parse_args()
    txt_path = args.txt_path
    rdmm_preweight_txt_path = args.rdmm_preweight_txt_path
    use_init_weight = rdmm_preweight_txt_path is not None
    mermaid_setting_path = args.mermaid_setting_path
    aug_setting_path = args.aug_setting_path
    fluid_mode = args.fluid_mode
    use_bspline = args.bspline
    output_path = args.output_path
    assert os.path.isfile(txt_path),"{} not exists".format(txt_path)
    assert os.path.isfile(aug_setting_path),"{} not exists".format(aug_setting_path)
    assert os.path.isfile(mermaid_setting_path),"{} not exists".format(mermaid_setting_path)
    if fluid_mode is None and not use_bspline:
        print("the fluid mode is not provided, now read from {}".format(aug_setting_path))
        params = pars.ParameterDict()
        params.load_JSON(aug_setting_path)
        fluid_mode = params["data_aug"]["fluid_aug"]["fluid_mode"]
    os.makedirs(output_path,exist_ok=True)
    # if the use_random_m is false or use_bspline, then the each only include moving and its label(optional)
    moving_momentum_path_list = get_pair_list(txt_path)
    init_weight_path_list = None
    if use_init_weight:
        init_weight_path_list = get_init_weight_list(rdmm_preweight_txt_path)

    generate_aug_data(moving_momentum_path_list,init_weight_path_list,output_path, mermaid_setting_path,fluid_mode, aug_setting_path,fluid_aug= not use_bspline)


