
from .base_toolkit import ToolkitBase
from .metrics import get_multi_metric
from model_pool.utils import *
from model_pool.nifty_reg_utils import *

class NiftyRegIter(ToolkitBase):
    def name(self):
        return 'nifty_reg iter'

    def initialize(self,opt):
        ToolkitBase.initialize(self, opt)
        if self.network_name =='affine':
            self.affine_on = True
            self.warp_on = False
        elif self.network_name =='bspline':
            self.affine_on = False
            self.warp_on = True
        self.nifty_reg_param = opt['tsk_set']['reg']['nifty_reg']





    def affine_optimization(self):
        output, loutput, phi,_ = performRegistration(self.nifty_reg_param, self.resized_moving_path,self.resized_target_path,self.network_name,self.record_path,self.resized_l_moving_path,fname = self.fname_list[0])

        self.output = output
        self.warped_label_map = loutput

        self.disp = None
        # self.phi = phi
        return self.output, None, None


    def bspline_optimization(self):
        output, loutput, phi,jacobian = performRegistration(self.nifty_reg_param,self.resized_moving_path,self.resized_target_path,self.network_name,self.record_path,self.resized_l_moving_path,fname = self.fname_list[0])


        self.disp = None
        self.output = output
        self.warped_label_map = loutput
        self.jacobian = jacobian
        self.phi = phi
        return self.output,None, None


    def forward(self,input=None):
        if self.affine_on and not self.warp_on:
            return self.affine_optimization()
        elif self.warp_on:
            return self.bspline_optimization()




    def compute_jacobi_map(self,jacobian):
        jacobi_abs = - np.sum(jacobian[jacobian < 0.])  #
        jacobi_num = np.sum(jacobian < 0.)
        print("the jacobi_value of fold points for current batch is {}".format(jacobi_abs))
        print("the number of fold points for current batch is {}".format(jacobi_num))
        # np.sum(np.abs(dfx[dfx<0])) + np.sum(np.abs(dfy[dfy<0])) + np.sum(np.abs(dfz[dfz<0]))
        jacobi_abs_mean = jacobi_abs  # / np.prod(map.shape)
        return jacobi_abs_mean,jacobi_num










