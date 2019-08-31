

from .base_toolkit import ToolkitBase
from model_pool.ants_reg_utils import *

class AntsRegIter(ToolkitBase):
    def name(self):
        return 'ants_reg iter'

    def initialize(self,opt):
        ToolkitBase.initialize(self, opt)
        if self.network_name =='affine':
            self.affine_on = True
            self.warp_on = False
        elif self.network_name =='syn':
            self.affine_on = False
            self.warp_on = True


    def affine_optimization(self):
        output, loutput, phi,_ = performAntsRegistration(self.resized_moving_path,self.resized_target_path,self.network_name,self.record_path,self.resized_l_moving_path,self.resized_l_target_path,self.fname_list[0])

        self.output = output
        self.warped_label_map = loutput
        self.phi = None

        return self.output, None, None


    def syn_optimization(self):
        output, loutput, disp,jacobian = performAntsRegistration(self.resized_moving_path,self.resized_target_path,self.network_name,self.record_path,self.resized_l_moving_path,self.resized_l_target_path,self.fname_list[0])


        #self.disp = None
        self.output = output
        self.warped_label_map = loutput
        self.jacobian= jacobian

        self.phi = None
        return self.output,None, None


    def forward(self,input=None):
        if self.affine_on and not self.warp_on:
            return self.affine_optimization()
        elif self.warp_on:
            """ the syn include affine"""
            return self.syn_optimization()



    def compute_jacobi_map(self,jacobian):
        """
        In ants, negative jacobi are set to zero
        the jacobi_abs_sum is not used here
        :param jacobian:
        :return:
        """
        jacobi_abs = -0.0 # - np.sum(jacobian[jacobian < 0.])  #
        jacobi_num = np.sum(jacobian <=0.)
        print("the jacobi_value of fold points for current batch is {}".format(jacobi_abs))
        print("the number of fold points for current batch is {}".format(jacobi_num))
        # np.sum(np.abs(dfx[dfx<0])) + np.sum(np.abs(dfy[dfy<0])) + np.sum(np.abs(dfz[dfz<0]))
        jacobi_abs_sum = jacobi_abs  # / np.prod(map.shape)
        return jacobi_abs_sum, jacobi_num

