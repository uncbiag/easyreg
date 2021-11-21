
from .utils import *
import SimpleITK as sitk




class RegModelBase():
    """
    the base class for image registration
    """
    def name(self):
        return 'RegModelBase'

    def initialize(self, opt):
        """
        :param opt: ParameterDict, task settings
        :return: None
        """
        self.opt = opt
        self.gpu_ids = opt['tsk_set'][('gpu_ids',0,'the gpu id used for network methods')]
        self.isTrain = opt['tsk_set'][('train',True,'True, take the train mode')]
        self.save_dir = opt['tsk_set']['path']['check_point_path']
        self.record_path = opt['tsk_set']['path']['record_path']
        self.spacing = None
        self.use_physical_coord = self.opt['tsk_set'][('use_physical_coord',False,"Keep physical spacing")]
        self.continue_train = opt['tsk_set'][('continue_train',False,"for network training method, continue training the model loaded from model_path")]
        self.criticUpdates = opt['tsk_set'][('criticUpdates',1,"for network training method, the num determines gradient update every # iter")]
        self.n_in_channel = opt['tsk_set'][('n_in_channel',1,"for network training method, the color channel typically set to 1")]
        self.input_img_sz = self.opt['dataset'][('img_after_resize',None,"image size after resampling")]
        self.original_im_sz = None
        self.original_spacing = None
        #self.input_resize_factor = opt['dataset']['input_resize_factor'] # todo remove this
        self.optimizer= None
        self.lr_scheduler = None
        self.exp_lr_scheduler= None
        self.iter_count = 0
        self.dim = 3#len(self.input_img_sz)
        self.network =None
        self.val_res_dic = {}
        self.fname_list = None
        self.pair_path = None
        self.moving = None
        self.target = None
        self.output = None
        self.warped_label_map = None
        self.l_moving = None
        self.l_target = None
        self.jacobi_val = None
        self.phi = None
        self.inverse_phi = None
        self.afimg_or_afparam = None
        self.jacobian=None
        self.multi_gpu_on =False # todo for now the distributed computing is not supported






    def set_input(self, input):
        """
        set the input of the method
        :param input:
        :return:
        """
        self.input = input

    def forward(self,input=None):
        pass

    def test(self):
        pass

    def set_train(self):
        """
        set the model in train mode (only for learning methods)
        :return:
        """
        self.network.train(True)
        self.is_train =True
    def set_val(self):
        """
        set the model in validation mode (only for learning methods)
        :return:
        """
        self.network.train(False)
        self.is_train = False

    def set_debug(self):
        """
        set the model in debug (subset of training set) mode (only for learning methods)
        :return:
        """
        self.network.train(False)
        self.is_train = False

    def set_test(self):
        """
        set the model in test mode ( only for learning methods)
        :return:
        """
        self.network.train(False)
        self.is_train = False


    def set_multi_gpu_on(self):
        """
        multi gpu support (disabled)
        :return:
        """
        self.multi_gpu_on = True



    def optimize_parameters(self):
        """
        optimize model parameters
        :return:
        """
        pass




    def get_debug_info(self):
        """ get debug info"""
        return None

    # get image paths
    def get_image_names(self):
        """get image name list"""
        return self.fname_list


    def set_cur_epoch(self,epoch):
        """
        set epoch
        :param epoch:
        :return:
        """
        self.cur_epoch = epoch




    def cal_loss(self,output= None):
       pass


    def get_current_errors(self):
        """
        get the current loss
        :return:
        """
        return self.loss.data[0]



    def compute_jacobi_map(self,map):
        pass




    def cal_val_errors(self):
        """ compute the loss on validatoin set"""
        self.cal_test_errors()

    def cal_test_errors(self):
        """ compute the loss on test set"""
        self.get_evaluation()

    def get_evaluation(self):
        """evaluate the performance of the current model"""
        pass



    def update_loss(self, epoch, end_of_epoch):
        pass

    def get_val_res(self, detail=False):
        """
        if the label map is given, evaluate the overlap sturcture
        :param detail:
        if detail, then output average dice score of each non-bg structure; and different scores of each structure
        if not, then output average dice score of each non-bg structure; and dice score of each structure
        :return:
        """
        if len(self.val_res_dic):
            if not detail:

                return np.mean(self.val_res_dic['batch_avg_res']['dice'][0, 1:]), self.val_res_dic['batch_avg_res'][
                    'dice']
            else:
                return np.mean(self.val_res_dic['batch_avg_res']['dice'][0, 1:]), self.val_res_dic['multi_metric_res']
        else:
            return -1, np.array([-1, -1])


    def get_test_res(self, detail=False):
        """
        if the label map is given, evaluate the overlap strucrue
        :param detail:
        if detail, then output average dice score of each non-bg structure; and different scores of each structure
        if not, then output average dice score of each non-bg structure; and dice score of each structure
        :return:
        """
        return self.get_val_res(detail=detail)

    def get_jacobi_val(self):
        """
        :return: the sum of absolute value of  negative determinant jacobi, the num of negative determinant jacobi voxels
        """
        return None


    def save_fig(self,phase):
       pass



    def do_some_clean(self):
        self.loss = None
        self.gt = None
        self.input = None
        self.output = None



    def save_fig_3D(self,phase=None):
        """
        save 3d output, i.e. moving, target and warped images,
        the propose of this function is for visualize the reg performance
        for toolkit based method, they will default save the 3d images, so no need to call this function
        for mermaid related method, this function is for result analysis, for original sz output, see "save_image_into_original_sz_with_given_reference",
        the physical information like  origin, orientation is not saved, todo, include this information
        :param phase: train|val|test|debug
        :return:

        """
        moving = self.moving
        target = self.target
        warped = self.output
        l_moving = self.l_moving
        l_target = self.l_target
        l_warped = self.warped_label_map
        to_numpy = lambda x: x if isinstance(x, np.ndarray) else x.detach().cpu().numpy()
        if type(self.moving) is not None:
            moving = to_numpy(self.moving)
            target = to_numpy(self.target)
            warped = to_numpy(self.output)
        if self.warped_label_map is not None:
            l_moving = to_numpy(self.l_moving)
            l_target = to_numpy(self.l_target)
            l_warped = to_numpy(self.warped_label_map)


        saving_folder_path = os.path.join(self.record_path, '3D')
        make_dir(saving_folder_path)
        for i in range(moving.shape[0]):
            appendix = self.fname_list[i] + "_"+phase+ "_iter_" + str(self.iter_count)
            saving_file_path = saving_folder_path + '/' + appendix + "_moving.nii.gz"
            output = sitk.GetImageFromArray(moving[i, 0, ...])
            output.SetSpacing(np.flipud(self.spacing))
            sitk.WriteImage(output, saving_file_path)
            saving_file_path = saving_folder_path + '/' + appendix + "_target.nii.gz"
            output = sitk.GetImageFromArray(target[i, 0, ...])
            output.SetSpacing(np.flipud(self.spacing))
            sitk.WriteImage(output, saving_file_path)
            saving_file_path = saving_folder_path + '/' + appendix + "_warped.nii.gz"
            output = sitk.GetImageFromArray(warped[i, 0, ...])
            output.SetSpacing(np.flipud(self.spacing))
            sitk.WriteImage(output, saving_file_path)
            if l_warped is not None:
                saving_file_path = saving_folder_path + '/' + appendix + "_moving_l.nii.gz"
                output = sitk.GetImageFromArray(l_moving[i, 0, ...])
                output.SetSpacing(np.flipud(self.spacing))
                sitk.WriteImage(output, saving_file_path)
                saving_file_path = saving_folder_path + '/' + appendix + "_target_l.nii.gz"
                output = sitk.GetImageFromArray(l_target[i, 0, ...])
                output.SetSpacing(np.flipud(self.spacing))
                sitk.WriteImage(output, saving_file_path)
                saving_file_path = saving_folder_path + '/' + appendix + "_warped_l.nii.gz"
                output = sitk.GetImageFromArray(l_warped[i, 0, ...])
                output.SetSpacing(np.flipud(self.spacing))
                sitk.WriteImage(output, saving_file_path)


    def save_deformation(self):
        pass







