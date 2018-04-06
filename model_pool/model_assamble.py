import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from glob import glob
from .base_model import get_from_model_pool
from pipLine.utils import get_test_model


class Assemble_Net_Test(nn.Module):
    """
    assemble model class, this is only for the test phase
    default_mode: dic(sched, model_name, model_folder_path, settings)given the mode_type, folder path, read models in that
    assign_mode: dic(sched, model_name_list,model_path_list,setting_dic_list) given the path of the model

    """
    def __init__(self, pars):
        super(Assemble_Net_Test, self).__init__()
        self.sched  = pars['sched']
        self.pars = pars
        self.model_list = []
        if self.sched == 'default':
            self.init_default_net()
        elif self.sched == 'assign':
            self.init_assign_net()
        else:
            raise ValueError("not implemented yet")


    def init_default_net(self):
        model_folder_path = self.pars['model_folder_path']
        #f_path = os.path.join(model_folder_path, '**', '*')
        #model_path_list = glob(f_path, recursive=True)
        model_name = self.pars['model_name']
        in_channel = self.pars['setting']['in_channel']
        n_class = self.pars['setting']['n_class']
        gpu_switcher = self.pars['setting']['gpu_switcher']
        epoch_list = self.pars['setting']['epoch_list']
        print("the epoch list is :{}".format(epoch_list))
        model_path_list = [os.path.join(model_folder_path,'epoch_'+str(epoch)+'_') for epoch in epoch_list]
        model_list = []
        for path in model_path_list:
            model = get_from_model_pool(model_name,in_channel, n_class)
            get_test_model(path,model,None,old_gpu=gpu_switcher[0],cur_gpu=gpu_switcher[1])
            model_list.append(model)
        self.model_list = nn.ModuleList(model_list).cuda()
        self.n_class = n_class
        print("default assemble model is successfully initialized")

    def init_assign_net(self):
        model_name_list = self.pars['model_name_list']
        model_path_list = self.pars['model_path_list']
        setting_dic_list = self.pars['setting_dic_list']
        model_list = []
        for i, path in enumerate(model_path_list):
            in_channel = setting_dic_list[i]['in_channel']
            n_class = setting_dic_list[i]['n_class']
            gpu_switcher = setting_dic_list[i]['gpu_switcher']
            model = get_from_model_pool(model_name_list[i],in_channel,n_class)
            get_test_model(model_path_list[i],None,old_gpu=gpu_switcher[0],cur_gpu=gpu_switcher[1])
            model_list.append(model)
        self.model_list = nn.ModuleList(model_list).cuda()
        self.n_class = n_class
        print("assigned assemble model is successfully initialized")

    def cal_voting_map(self, input):
        """
        :param input:  batch x period x X x  Y x Z
        :return:
        """
        count_map =torch.cuda.FloatTensor(input.shape[0], self.n_class,input.shape[2],input.shape[3],input.shape[4]).fill_(0)
        count_map = Variable(count_map)
        #count_map = torch.zeros([list(input.shape)[0]]+[self.n_classes] + list(input.shape)[2:]).cuda()

        for i in range(self.n_class):
            count_map[:,i,...] = torch.sum(input == i, dim=1)
        return count_map

    def forward(self, input):
        output_list = []
        for model in self.model_list:
            output_list += [torch.max(model(input),1)[1]]
        output =torch.stack(output_list,dim=1)
        output = self.cal_voting_map(output)
        return output


