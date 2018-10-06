use_mermaid_iter = False
use_sym = True
gl_sym_factor = 200.
use_mermaid_multi_step =True
intra_training = False
reg_factor = 10

using_analyic_af_inverse=False
use_resid_momentum = True
use_llddmm = False
debug_term = 70
param_in_ants = 64
param_in_demons =None# (2,1) #(8,4)
nifty_reg_cmd =' -pad 0 ' #' -pad 0 ' ' -sx -20 --lncc 40 -pad 0 '  ' -sx -10 --lncc 40 -pad 0 '
print("global varibale  file is loaded %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5")
print('use_sym :{}, use_resid_momentunm {}, debug_term {}, use_llddmm {},nifty_reg_cmd {} '.format(use_sym,use_resid_momentum,debug_term,use_llddmm,nifty_reg_cmd))