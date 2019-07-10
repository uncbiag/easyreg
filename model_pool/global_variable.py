is_oai = True
is_oasis = not is_oai
use_mermaid_iter = False   #todo attention this should be changed
use_odeint = True
update_sm_by_advect = True
update_sm_with_interpolation = True
use_preweights_advect = True
# todo attention the data loader should be recovered if using oai
# todo attention the soft/abs should be checked in registration_network
use_velocity_mask = False   # todo attention this should be true
clamp_local_weight = True
local_pre_weight_max =1.5  #todo attention this should be 1.5
use_omt_const = False

bysingle_int = False  # update by single phase interoplation


is_lung = False  ###  # todo this should be False
compute_inverse_map = False
output_orginal_img_sz= False
original_img_sz = [160,384,384]

reg_factor_in_regnet = 1
reg_factor_in_mermaid = 10  # todo attention this should be 10 in learning task
lambda_factor_in_vmr= 50.
lambda_mean_factor_in_vmr =50
sigma_factor_in_vmr = 0.02
use_affine_in_vmr = False
save_jacobi_map = False
save_extra_fig =False
param_in_ants = 64
param_in_demons =None# (2,1) #(8,4)
#nifty_bin = '/playpen/zyshen/package/niftyreg-git/niftyreg_install/bin'
nifty_bin = '/pine/scr/z/y/zyshen/proj/niftyreg-git/niftyreg_install/bin'

nifty_reg_cmd =' -pad 0 -jl 0.01  '#' -sx -10 --lncc 40 -pad 0 '#' -pad 0 '#' -sx -5 --lncc 20 -pad 0  '
    #' -sx -10 --lncc 40 -pad 0 -jl 0.01  ' #' -sx -10 --lncc 40 -pad 0 ' #
               #' -sx -10 --lncc 40 -pad 0 ' #'  -sx -5 -pad 0 ' #' -pad 0 ' ' -sx -20 --lncc 40 -pad 0 '  ' -sx -10 --lncc 40 -pad 0 '
print("global varibale  file is loaded %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5")
