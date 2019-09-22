
from model_pool.utils import *
import torch.optim.lr_scheduler as lr_scheduler
import SimpleITK as sitk




class BaseModel():
    """
    the base class for image registration
    """
    def name(self):
        return 'BaseModel'

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
        self.input_img_sz = self.opt['dataset'][('img_after_resize',None,"image size after resample")]
        self.original_im_sz = None
        self.original_spacing = None
        #self.input_resize_factor = opt['dataset']['input_resize_factor'] # todo remove this
        self.evaluate_label_list = opt['tsk_set']['evaluate_label_list',[-100],'evaluate_label_list']
        self.optimizer= None
        self.lr_scheduler = None
        self.exp_lr_scheduler= None
        self.iter_count = 0
        self.dim = 3#len(self.input_img_sz)
        self.network =None
        self.val_res_dic = {}
        self.fname_list = None
        self.moving = None
        self.target = None
        self.output = None
        self.warped_label_map = None
        self.l_moving = None
        self.l_target = None
        self.jacobi_val = None
        self.phi = None
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

    def init_optim(self, opt,network, warmming_up = False):
        """
        set optimizers and scheduler
        :param opt: settings on optimizer
        :param network: model with learnable parameters
        :param warmming_up: if set as warmming up
        :return: optimizer, custom scheduler, plateau scheduler
        """
        optimize_name = opt['optim_type']
        if not warmming_up:
            lr = opt['lr']
            print(" no warming up the learning rate is {}".format(lr))
        else:
            lr = 5e-4
            print(" warming up on the learning rate is {}".format(lr))
        beta = opt['adam']['beta']
        lr_sched_opt = opt['lr_scheduler']
        self.lr_sched_type = lr_sched_opt['type']
        if optimize_name == 'adam':
            re_optimizer = torch.optim.Adam(network.parameters(), lr=lr, betas=(beta, 0.999))
        else:
            re_optimizer = torch.optim.SGD(network.parameters(), lr=lr)
        re_optimizer.zero_grad()
        re_lr_scheduler = None
        re_exp_lr_scheduler = None
        if self.lr_sched_type == 'custom':
            step_size = lr_sched_opt['custom']['step_size']
            gamma = lr_sched_opt['custom']['gamma']
            re_lr_scheduler = torch.optim.lr_scheduler.StepLR(re_optimizer, step_size=step_size, gamma=gamma)
        elif self.lr_sched_type == 'plateau':
            patience = lr_sched_opt['plateau']['patience']
            factor = lr_sched_opt['plateau']['factor']
            threshold = lr_sched_opt['plateau']['threshold']
            min_lr = lr_sched_opt['plateau']['min_lr']
            re_exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(re_optimizer, mode='min', patience=patience,
                                                                   factor=factor, verbose=True,
                                                                   threshold=threshold, min_lr=min_lr)
        return re_optimizer,re_lr_scheduler,re_exp_lr_scheduler


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
        if type(self.moving)==torch.Tensor:
            moving = self.moving.detach().cpu().numpy()
            target = self.target.detach().cpu().numpy()
            warped = self.output.detach().cpu().numpy()
        else:
            moving = self.moving
            target = self.target
            warped = self.output


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

    def save_deformation(self):
        pass







