import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from glob import glob

def encoder(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
            bias=True, batchnorm=False):
    if batchnorm:
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
    else:
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU())
    return layer


def decoder( in_channels, out_channels, kernel_size, stride=1, padding=0,
            output_padding=0, bias=True, batchnorm=False):
    if batchnorm:
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
    else:
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
    return layer





class UNet_Fea(nn.Module):
    def __init__(self, in_channel, bias=True, BN=True):
        super(UNet_Fea, self).__init__()
        self.e0cb = encoder(in_channel, 32, bias=True, batchnorm=True)
        self.e0ce = encoder(32, 64, bias=True, batchnorm=True)
        self.e0_sq = nn.Sequential(self.e0cb, self.e0ce)
        self.pool0 = nn.MaxPool3d(2)
        self.e1cb = encoder(64, 64, bias=True, batchnorm=True)
        self.e1ce = encoder(64, 128, bias=True, batchnorm=True)
        self.e1_sq = nn.Sequential(self.pool0, self.e1cb, self.e1ce)
        self.pool1 = nn.MaxPool3d(2)
        self.e2cb = encoder(128, 128, bias=True, batchnorm=True)
        self.e2ce = encoder(128, 256, bias=True, batchnorm=True)
        self.e2_sq = nn.Sequential(self.pool1, self.e2cb, self.e2ce)

    def forward(self, x):
        e0_sq_res = self.e0_sq(x)
        e1_sq_res = self.e1_sq(e0_sq_res)
        e2_sq_res = self.e2_sq(e1_sq_res)
        return e0_sq_res,e1_sq_res,e2_sq_res


class UNet_Dis(nn.Module):
    def __init__(self, n_classes, bias=True, BN=True):
        super(UNet_Dis, self).__init__()


        self.pool2 = nn.MaxPool3d(2)
        self.e3cb = encoder(256, 256, bias=True, batchnorm=True)
        self.e3c1 = encoder(256, 512, bias=True, batchnorm=True)
        self.e3ce = decoder(512, 512, kernel_size=2, stride=2, bias=True, batchnorm=True)
        self.e3_sq = nn.Sequential(self.pool2, self.e3cb, self.e3c1, self.e3ce)

        self.d2cb = decoder(256 + 512, 256, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)
        self.d2c1 = decoder(256, 256, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)
        self.d2ce = decoder(256, 256, kernel_size=2, stride=2, bias=True, batchnorm=True)
        self.d2_sq = nn.Sequential(self.d2cb, self.d2c1, self.d2ce)

        self.d1cb = decoder(128 + 256, 128, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)
        self.d1c1 = decoder(128, 128, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)
        self.d1ce = decoder(128, 128, kernel_size=2, stride=2, bias=True, batchnorm=True)
        self.d1_sq = nn.Sequential(self.d1cb, self.d1c1, self.d1ce)

        self.d0cb = decoder(64 + 128, 128, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)
        self.d0c1 = decoder(128, 128, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)
        self.d0ce = decoder(128, n_classes, kernel_size=1, stride=1, bias=True, batchnorm=True)
        self.d0_sq = nn.Sequential(self.d0cb, self.d0c1, self.d0ce)


    def forward(self, input):
        e0_sq_res, e1_sq_res, e2_sq_res = input
        e3_sq_res = self.e3_sq(e2_sq_res)
        d2_sq_res = self.d2_sq(torch.cat((e3_sq_res, e2_sq_res), dim=1))
        d1_sq_res = self.d1_sq(torch.cat((d2_sq_res, e1_sq_res), dim=1))
        d0_sq_res = self.d0_sq(torch.cat((d1_sq_res, e0_sq_res), dim=1))
        return d0_sq_res


class UNet_asm_full(nn.Module):
    #  there is a bug here before 3.9
    def __init__(self,in_channel, n_classes, bias=True, BN=True):
        super(UNet_asm_full, self).__init__()
        self.net_fea = UNet_Fea(in_channel,bias, BN)
        self.net_dis = UNet_Dis(n_classes,bias,BN)

    def forward(self, input):
        output = self.net_fea(input)
        output = self.net_dis(output)
        return output






class Vonet_test(nn.Module):
    def __init__(self, in_channel, n_classes, path, epoch_list, gpu_switcher,bias=False, BN=False):
        super(Vonet_test, self).__init__()

        self.in_channel = in_channel
        self.n_classes = n_classes
        self.bias_on = bias
        self.BN_on = BN
        self.net_fea = UNet_Fea(in_channel,bias, BN)
        self.epoch_str_list = [str(epoch) for epoch in epoch_list]
        self.gpu_switcher = gpu_switcher
        self.net_dis_list = nn.ModuleList([UNet_Dis(n_classes, bias,BN) for i in self.epoch_str_list])
        self.init_net(path)
        self.selected_epoch = -1


    def init_net(self, path):
        self.load_fea_state(path)
        self.load_dis_state(path)
        print("vonet test successfully initialized")

    def load_fea_state(self,path):
        fpath =  os.path.join(os.path.join(path,'asm_models'),'fea')
        fpath_alter = os.path.join(path, 'model_best.pth.tar')

        if os.path.exists(fpath):
            model_state = torch.load(fpath, map_location={
                'cuda:' + str(self.gpu_switcher[0]): 'cuda:' + str(self.gpu_switcher[1])})
            self.net_fea.load_state_dict(model_state)
        elif os.path.exists(fpath_alter):
            checkpoint = torch.load(fpath_alter,map_location={'cuda:' + str(self.gpu_switcher[0]): 'cuda:' + str(self.gpu_switcher[1])})
            net_tmp = UNet_asm(self.in_channel,self.n_classes,self.bias_on, self.BN_on)
            net_tmp.load_state_dict(checkpoint['state_dict'])
            self.net_fea.load_state_dict(net_tmp.net_fea.state_dict())
        else:
            print("no fea model is found")
            exit(2)


    def load_dis_state(self, path):
        asm_path = os.path.join(path, 'asm_models')
        f_path = os.path.join(asm_path, '**', 'epoch' + '*')
        f_filter = glob(f_path, recursive=True)
        if not len(f_filter):
            print("Error, no asmable file finded in folder {}".format(f_path))
            exit(3)
        fname_epoch_dic = {os.path.split(file)[1].split('_')[1]:file for file in f_filter}
        for fname_epoch in self.epoch_str_list:
            if fname_epoch in fname_epoch_dic:
                idx = self.epoch_str_list.index(fname_epoch)
                model_state = torch.load(fname_epoch_dic[fname_epoch], map_location={'cuda:' + str(self.gpu_switcher[0]): 'cuda:' + str(self.gpu_switcher[1])})
                self.net_dis_list[idx].load_state_dict(model_state)
            else:
                print("Error, during asmble learning, epoch {} is not found".format(fname_epoch))
                exit(4)

    def cal_voting_map(self, input):
        """

        :param input:  batch x period x X x  Y x Z
        :return:
        """
        count_map =torch.cuda.FloatTensor(input.shape[0], self.n_classes,input.shape[2],input.shape[3],input.shape[4]).fill_(0)
        count_map = Variable(count_map)
        #count_map = torch.zeros([list(input.shape)[0]]+[self.n_classes] + list(input.shape)[2:]).cuda()

        for i in range(self.n_classes):
            count_map[:,i,...] = torch.sum(input == i, dim=1)
        return count_map

    def forward(self, input):
        fea_output = self.net_fea(input)
        output_list = []
        for dis_net in self.net_dis_list:
            output_list += [torch.max(dis_net(fea_output),1)[1]]
        output =torch.stack(output_list,dim=1)
        output = self.cal_voting_map(output)
        return output




# class UNet_Fea(nn.Module):
#     def __init__(self, in_channel, bias=True, BN=True):
#         super(UNet_Fea, self).__init__()
#
#         self.in_channel = in_channel
#         self.ec0 = encoder(self.in_channel, 16, bias=bias, batchnorm=BN)
#         self.ec1 = encoder(16, 32, bias=bias, batchnorm=BN)
#         self.ec2 = encoder(32, 32, bias=bias, batchnorm=BN)
#         self.ec3 = encoder(32, 64, bias=bias, batchnorm=BN)
#         self.ec4 = encoder(64, 64, bias=bias, batchnorm=BN)
#         self.ec5 = encoder(64, 128, bias=bias, batchnorm=BN)
#
#         self.pool0 = nn.MaxPool3d(2)
#         self.pool1 = nn.MaxPool3d(2)
#         # self.pool2 = nn.MaxPool3d(2)
#
#         self.dc6 = decoder(128, 128, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
#         self.dc5 = decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
#         self.dc4 = decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
#
#
#
#
#     def forward(self, x):
#         e0 = self.ec0(x)
#         syn0 = self.ec1(e0)
#         e1 = self.pool0(syn0)
#         e2 = self.ec2(e1)
#         syn1 = self.ec3(e2)
#         del e0, e1, e2
#
#         e3 = self.pool1(syn1)
#         e4 = self.ec4(e3)
#         e5 = self.ec5(e4)
#         del e3, e4
#
#         d6 = torch.cat((self.dc6(e5), syn1), dim=1)
#         del e5, syn1
#
#         d5 = self.dc5(d6)
#         d4 = self.dc4(d5)
#         del d6, d5
#         return d4,syn0
#
#
#
#
# class UNet_Dis(nn.Module):
#     def __init__(self, n_classes, bias=True, BN=True):
#         super(UNet_Dis, self).__init__()
#         self.dc3 = decoder(64, 64, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
#         self.dc2 = decoder(32 + 64, 32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
#         self.dc1 = decoder(32, 32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
#         self.dc0 = nn.Conv3d(32, n_classes, kernel_size=1, stride=1, padding=0, bias=bias)
#
#
#     def forward(self, input):
#         d4, syn0 =input
#         d3 = torch.cat((self.dc3(d4), syn0), dim=1)
#         del d4, syn0
#         d2 = self.dc2(d3)
#         d1 = self.dc1(d2)
#         del d3, d2
#         d0 = self.dc0(d1)
#         return d0
