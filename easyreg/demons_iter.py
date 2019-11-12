
from .base_toolkit import ToolkitBase
from .demons_utils import *
class DemonsRegIter(ToolkitBase):
    """
    a symmetric forces demons  algorithm
    """
    def name(self):
        return 'demons_reg iter'

    def initialize(self,opt):
        """
        initialize the demons registration
        mehtod support: "demons"
        * the "demons" include niftyreg affine as preproccessing

        :param opt: task opt settings
        :return: None
        """
        ToolkitBase.initialize(self, opt)
        if self.method_name =='affine':
            self.affine_on = True
            self.warp_on = False
            raise ValueError("affine is not separately used in demons")
        elif self.method_name =='demons':
            """ In this case, the nifty affine would be first called"""
            self.affine_on = False
            self.warp_on = True
        self.demons_param = opt['tsk_set']['reg']['demons']


    def demons_optimization(self):
        """
        run the demons optimization
        the results, including warped image, warped label, transformation map, etc. take the demons format and saved in record path

        :return: warped image, warped label(None), transformation map(None)
        """
        output, loutput, phi,jacobian = performDemonsRegistration(self.demons_param, self.resized_moving_path,self.resized_target_path,self.method_name,self.record_path,self.resized_l_moving_path,self.resized_l_target_path,self.fname_list[0])
        self.afimg_or_afparam = None
        self.output = output
        self.warped_label_map = loutput
        self.jacobian =jacobian

        self.phi = phi
        return self.output,None, None


    def forward(self,input=None):
        """
        forward the model

        :param input:
        :return:
        """
        if self.warp_on:
            return self.demons_optimization()




    def compute_jacobi_map(self,jacobian):
        """
        compute the jacobi statistics

        :param jacobian: jacob determinant  map of the transformation map
        :return:the abs sum of the negative determinate, the num of negative determinate voxel
        """
        jacobi_abs = - np.sum(jacobian[jacobian < 0.])  #
        jacobi_num = np.sum(jacobian < 0.)
        print("the jacobi_value of fold points for current batch is {}".format(jacobi_abs))
        print("the number of fold points for current batch is {}".format(jacobi_num))
        # np.sum(np.abs(dfx[dfx<0])) + np.sum(np.abs(dfy[dfy<0])) + np.sum(np.abs(dfz[dfz<0]))
        #jacobi_abs_mean = jacobi_abs  # / np.prod(map.shape)
        return jacobi_abs, jacobi_num








