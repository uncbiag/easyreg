use_mermaid_iter = False
use_sym = False
gl_sym_factor = 500.
use_mermaid_multi_step =True
physical_mode_off = False # this should be  False
mermaid_step = 2 # this should be  off
intra_training = True 
reg_factor = 10
use_extra_inter_intra_judge =True
using_analyic_af_inverse=False
use_resid_momentum = True
use_lconv_momentum  =not use_resid_momentum
conv_size=5
conv_size2=5
use_llddmm = False
debug_term = 70
param_in_ants = 64
use_provided_affine_txt=True
provided_affine_path='/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_demons_jacobi/records'
param_in_demons =None# (2,1) #(8,4)
mermaid_net_setting_file  ='../mermaid/demos/cur_settings_lbfgs.json'  #'../mermaid/demos/cur_settings_lbfgs.json'
nifty_reg_cmd = ' -sx -10 --lncc 40 -pad 0 -jl 0.01  ' #' -vel -pad0' #' -sx -15 --lncc 40 -pad 0 ' #'  -sx -5 -pad 0 ' #' -pad 0 ' ' -sx -20 --lncc 40 -pad 0 '  ' -sx -10 --lncc 40 -pad 0 '
print("global varibale  file is loaded %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5")

print('use_sym :{}, use_resid_momentunm {}, debug_term {}, use_llddmm {},nifty_reg_cmd {} '.format(use_sym,use_resid_momentum,debug_term,use_llddmm,nifty_reg_cmd))