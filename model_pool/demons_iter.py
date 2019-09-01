
from .base_toolkit import ToolkitBase
from .losses import Loss
from .metrics import get_multi_metric

from model_pool.utils import *

from model_pool.demons_utils import *
import mermaid.utils as py_utils

import mermaid.simple_interface as SI
import mermaid.fileio as FIO
class DemonsRegIter(ToolkitBase):
    def name(self):
        return 'demons_reg iter'

    def initialize(self,opt):
        ToolkitBase.initialize(self, opt)
        if self.network_name =='affine':
            self.affine_on = True
            self.warp_on = False
            raise ValueError("affine is not separately used in demons")
        elif self.network_name =='demons':
            """ In this case, the nifty affine would be first called"""
            self.affine_on = False
            self.warp_on = True
        self.demons_param = opt['tsk_set']['reg']['demons']





    def affine_optimization(self):

        output, loutput, phi = performDemonsRegistration(self.demons_param, self.resized_moving_path,self.resized_target_path,self.network_name,self.record_path,self.resized_l_moving_path,self.resized_l_target_path,self.fname_list[0])

        self.output = output
        self.warped_label_map = loutput

        self.disp = None
        self.phi = None
        return self.output, None, None


    def demons_optimization(self):
        output, loutput, phi,jacobian = performDemonsRegistration(self.demons_param, self.resized_moving_path,self.resized_target_path,self.network_name,self.record_path,self.resized_l_moving_path,self.resized_l_target_path,self.fname_list[0])


        self.disp = None
        self.output = output
        self.warped_label_map = loutput
        self.jacobian =jacobian

        self.phi = phi
        return self.output,None, None


    def forward(self,input=None):
        if self.affine_on and not self.warp_on:
            return self.affine_optimization()
        elif self.warp_on:
            return self.demons_optimization()




    def compute_jacobi_map(self,jacobian):
        jacobi_abs = - np.sum(jacobian[jacobian < 0.])  #
        jacobi_num = np.sum(jacobian < 0.)
        print("the jacobi_value of fold points for current batch is {}".format(jacobi_abs))
        print("the number of fold points for current batch is {}".format(jacobi_num))
        # np.sum(np.abs(dfx[dfx<0])) + np.sum(np.abs(dfy[dfy<0])) + np.sum(np.abs(dfz[dfz<0]))
        jacobi_abs_mean = jacobi_abs  # / np.prod(map.shape)
        return jacobi_abs_mean, jacobi_num








