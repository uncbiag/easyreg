from .base_toolkit import ToolkitBase
from .ants_utils import *

class AntsRegIter(ToolkitBase):
    """
    The AntsRegIter provides an interface to [AntsPy](https://github.com/ANTsX/ANTsPy),
    the version we work on is 0.1.4, though the newest version is 0.2.0
    AntsPy is not fully functioned, a support on ants package is plan to replace the AntsPy.
    """
    def name(self):
        return 'ants_reg iter'

    def initialize(self,opt):
        """
        initialize the ants registration
        mehtod support: "affine", "syn"
        * the "syn" include affine as preproccessing

        :param opt: task opt settings
        :return: None
        """
        ToolkitBase.initialize(self, opt)
        if self.method_name =='affine':
            self.affine_on = True
            self.warp_on = False
        elif self.method_name =='syn':
            self.affine_on = False
            self.warp_on = True
        self.ants_param = opt['tsk_set']['reg']['ants']



    def affine_optimization(self):
        """
        run the affine optimization
        the results, including warped image, warped label, transformation map, etc. take the ants format and saved in record path

        :return: warped image, warped label(None), transformation map(None)
        """
        output, loutput, phi,_ = performAntsRegistration(self.ants_param, self.resized_moving_path,self.resized_target_path,self.method_name,self.record_path,self.resized_l_moving_path,self.resized_l_target_path,self.fname_list[0])

        self.output = output
        self.warped_label_map = loutput
        self.phi = None

        return self.output, None, None


    def syn_optimization(self):
        """
        run the syn optimization
        the results, including warped image, warped label, transformation map, etc. take the ants format and saved in record path

        :return: warped image, warped label(None), transformation map(None)
        """
        output, loutput, disp,jacobian = performAntsRegistration(self.ants_param, self.resized_moving_path,self.resized_target_path,self.method_name,self.record_path,self.resized_l_moving_path,self.resized_l_target_path,self.fname_list[0])

        #self.afimg_or_afparam = None
        self.output = output
        self.warped_label_map = loutput
        self.jacobian= jacobian

        self.phi = None
        return self.output,None, None


    def forward(self,input=None):
        """
        forward the model

        :param input:
        :return:
        """
        if self.affine_on and not self.warp_on:
            return self.affine_optimization()
        elif self.warp_on:
            """ the syn include affine"""
            return self.syn_optimization()



    def compute_jacobi_map(self,jacobian):
        """
        In ants, negative jacobi are set to zero,
        we compute the num of zero jacobi instead
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

