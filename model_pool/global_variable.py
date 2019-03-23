is_oai = True
is_oasis = not is_oai
use_mermaid_iter = True
use_odeint = True
update_sm_by_advect = True
update_sm_with_interpolation = True
use_preweights_advect = True
bysingle_int = False  # update by single phase interoplation
reg_factor_in_regnet = 1
reg_factor_in_mermaid = 20
lambda_factor_in_vmr= 50.
lambda_mean_factor_in_vmr =50
sigma_factor_in_vmr = 0.02
use_affine_in_vmr = False
save_jacobi_map = False
param_in_ants = 64
param_in_demons =None# (2,1) #(8,4)
#nifty_bin = '/playpen/zyshen/package/niftyreg-git/niftyreg_install/bin'
nifty_bin = '/pine/scr/z/y/zyshen/proj/niftyreg-git/niftyreg_install/bin'

nifty_reg_cmd =' -pad 0 -jl 0.01  '#' -sx -10 --lncc 40 -pad 0 '#' -pad 0 '#' -sx -5 --lncc 20 -pad 0  '
    #' -sx -10 --lncc 40 -pad 0 -jl 0.01  ' #' -sx -10 --lncc 40 -pad 0 ' #
               #' -sx -10 --lncc 40 -pad 0 ' #'  -sx -5 -pad 0 ' #' -pad 0 ' ' -sx -20 --lncc 40 -pad 0 '  ' -sx -10 --lncc 40 -pad 0 '
print("global varibale  file is loaded %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5")
