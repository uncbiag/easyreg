
from .base_toolkit import ToolkitBase
from .nifty_reg_utils import *

class NiftyRegIter(ToolkitBase):
    """
    an interface class to call niftyreg toolkit
    """
    def name(self):
        return 'nifty_reg iter'

    def initialize(self,opt):
        ToolkitBase.initialize(self, opt)
        if self.method_name =='affine':
            self.affine_on = True
            self.warp_on = False
        elif self.method_name =='bspline':
            self.affine_on = False
            self.warp_on = True
        self.nifty_reg_param = opt['tsk_set']['reg']['nifty_reg']





    def affine_optimization(self):
        """
        call affine optimization registration from niftyreg

        :return: warped image, transformation map (disabled), affine parameter(disabled)
        """
        output, loutput, phi,_ = performRegistration(self.nifty_reg_param, self.resized_moving_path,self.resized_target_path,self.method_name,self.record_path,self.resized_l_moving_path,fname = self.fname_list[0])

        self.output = output
        self.warped_label_map = loutput

        self.afimg_or_afparam = None
        # self.phi = phi
        return self.output, None, None


    def bspline_optimization(self):
        """
        call bspline optimization registration from niftyreg (include nifty affine)

        :return: warped image, transformation map (disabled), affine image(disabled)
        """
        output, loutput, phi,jacobian = performRegistration(self.nifty_reg_param,self.resized_moving_path,self.resized_target_path,self.method_name,self.record_path,self.resized_l_moving_path,fname = self.fname_list[0])


        self.afimg_or_afparam = None
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
        """
        compute the  negative determinant jaocbi of the transformation map

        :param jacobian: the determinant jacobi compute by the niftyreg toolkit
        :return: the sum of absolute value of  negative determinant jacobi, the num of negative determinant jacobi voxels
        """
        jacobi_abs = - np.sum(jacobian[jacobian < 0.])  #
        jacobi_num = np.sum(jacobian < 0.)
        print("the jacobi_value of fold points for current batch is {}".format(jacobi_abs))
        print("the number of fold points for current batch is {}".format(jacobi_num))
        # np.sum(np.abs(dfx[dfx<0])) + np.sum(np.abs(dfy[dfy<0])) + np.sum(np.abs(dfz[dfz<0]))
        return jacobi_abs,jacobi_num










