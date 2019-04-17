import torch
import numpy as np
import sys,os
import SimpleITK as sitk


sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../model_pool'))
sys.path.insert(0,os.path.abspath('../mermaid'))
import mermaid.pyreg.example_generation as eg
import mermaid.pyreg.module_parameters as pars
import mermaid.pyreg.multiscale_optimizer as MO
import mermaid.pyreg.smoother_factory as SF
from model_pool.nifty_reg_utils import performRegistration
import ants
from model_pool.ants_reg_utils import performAntsRegistration
#from model_pool.demo_for_demons import multiscale_demons
params = pars.ParameterDict()
params.load_JSON('../mermaid/test/json/svf_momentum_base_config.json')

params['model']['deformation']['use_map'] = True
params['model']['registration_model']['type'] = 'svf_vector_momentum_map'
record_path = '/playpen/zyshen/debugs/compare_disp'
if not os.path.exists(record_path):
    os.mkdir(record_path)
example_img_len = 64
dim = 3
szEx = np.tile(example_img_len, dim)  # size of the desired images: (sz)^dim
I0, I1,spacing = eg.CreateSquares(dim,add_noise_to_bg=False).create_image_pair(szEx, params)  # create a default image size with two sample squares
sz = np.array(I0.shape)
registration_type= 'bspline'
I0 = np.squeeze(I0)
I1 = np.squeeze(I1)
# create the source and target image as pyTorch variables
moving_img_path = os.path.join(record_path,'source.nii.gz')
target_img_path = os.path.join(record_path,'target.nii.gz')
source_np = sitk.WriteImage(sitk.GetImageFromArray(I0),moving_img_path)
target_np = sitk.WriteImage(sitk.GetImageFromArray(I1),target_img_path)
#_,_, phi=performRegistration(moving_img_path,target_img_path,registration_type,record_path=record_path,ml_path=None,affine_on=False)


#################    ants
syn_res = performAntsRegistration(moving_img_path, target_img_path, registration_type='syn', record_path=record_path, ml_path=None,tl_path= None, fname = 'ants_reg',return_syn=True)
ants.image_write(syn_res['warpedmovout'],os.path.join(record_path,'ants_warped.nii.gz'))


# ###################  demons ########################
# demons_filter =  sitk.FastSymmetricForcesDemonsRegistrationFilter()
# demons_filter.SetNumberOfIterations(20)
# # Regularization (update field - viscous, total field - elastic).
# demons_filter.SetSmoothDisplacementField(True)
# demons_filter.SetStandardDeviations(1.)
# tx, disp_np = multiscale_demons(registration_algorithm=demons_filter,
#                        fixed_image_pth = target_img_path,
#                        moving_image_pth = moving_img_path,
#                        shrink_factors =[4,2],
#                        smoothing_sigmas = [4,2],
#                        initial_transform=None,
#                                 record_path=record_path)
