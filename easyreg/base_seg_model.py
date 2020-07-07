
from .utils import *
import SimpleITK as sitk
from tools.visual_tools import save_3D_img_from_numpy



class SegModelBase():
    """
    the base class for image segmentation
    """
    def name(self):
        return 'SegModelBase'

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
        self.input = None
        self.output = None
        self.gt = None
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




    def save_fig(self,phase):
       pass



    def do_some_clean(self):
        self.loss = None
        self.gt = None
        self.input = None
        self.output = None



    def save_fig_3D(self,phase=None):
        """
        save 3d output,
        the propose of this function is for visualize the seg performance
        for toolkit based method, they will default save the 3d images, so no need to call this function
        the physical information like  origin, orientation is not saved, todo, include this information
        :param phase: train|val|test|debug
        :return:
        """
        if type(self.output)==torch.Tensor:
            output = self.output.detach().cpu().numpy()
        else:
            output = self.output
        if type(self.gt)==torch.Tensor:
            gt = self.gt.detach().cpu().numpy()
        else:
            gt = self.gt

        output = output.astype(np.int32)
        if gt is not None:
            gt = gt.astype(np.int32)

        spacing = self.spacing.cpu().numpy()


        saving_folder_path = os.path.join(self.record_path, '3D')
        make_dir(saving_folder_path)
        num_output = output.shape[0]
        for i in range(num_output):
            appendix = self.fname_list[i] + "_"+phase+ "_iter_" + str(self.iter_count)
            saving_file_path = saving_folder_path + '/' + appendix + "_output.nii.gz"
            output = sitk.GetImageFromArray(output[i, 0, ...])
            output.SetSpacing(np.flipud(spacing[i]))
            sitk.WriteImage(output, saving_file_path)
            if gt is not None:
                saving_file_path = saving_folder_path + '/' + appendix + "_gt.nii.gz"
                output = sitk.GetImageFromArray(gt[i, 0, ...])
                output.SetSpacing(np.flipud(spacing[i]))
                sitk.WriteImage(output, saving_file_path)





