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





class UNet_Fea_light1(nn.Module):
    def __init__(self, in_channel, bias=True, BN=True):
        super(UNet_Fea_light1, self).__init__()

        self.in_channel = in_channel
        self.ec0 = encoder(self.in_channel, 16, bias=bias, batchnorm=BN)
        self.ec1 = encoder(16, 32, bias=bias, batchnorm=BN)
        self.ec2 = encoder(32, 32, bias=bias, batchnorm=BN)
        self.ec3 = encoder(32, 64, bias=bias, batchnorm=BN)
        self.ec4 = encoder(64, 64, bias=bias, batchnorm=BN)
        self.ec5 = encoder(64, 128, bias=bias, batchnorm=BN)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        # self.pool2 = nn.MaxPool3d(2)

        self.dc6 = decoder(128, 128, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
        self.dc5 = decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc4 = decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)




    def forward(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        e5 = self.ec5(e4)
        del e3, e4

        d6 = torch.cat((self.dc6(e5), syn1), dim=1)
        del e5, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        del d6, d5
        return d4,syn0


class UNet_Dis_light1(nn.Module):
    def __init__(self, n_classes, bias=True, BN=True):
        super(UNet_Dis_light1, self).__init__()
        self.dc3 = decoder(64, 64, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
        self.dc2 = decoder(32 + 64, 32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc1 = decoder(32, 32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc0 = nn.Conv3d(32, n_classes, kernel_size=1, stride=1, padding=0, bias=bias)


    def forward(self, input):
        d4, syn0 =input
        d3 = torch.cat((self.dc3(d4), syn0), dim=1)
        del d4, syn0
        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2
        d0 = self.dc0(d1)
        return d0


class UNet_asm(nn.Module):
    #  there is a bug here before 3.9
    def __init__(self,in_channel, n_classes, bias=True, BN=True):
        super(UNet_asm, self).__init__()
        self.net_fea = UNet_Fea_light1(in_channel,bias, BN)
        self.net_dis = UNet_Dis_light1(n_classes,bias,BN)

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






class UNet_light2(nn.Module):
    def __init__(self, in_channel, n_classes, bias=False, BN=False):
        super(UNet_light2, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.ec0 = self.encoder(self.in_channel, 8, bias=bias, batchnorm=BN)
        self.ec1 = self.encoder(8, 16, bias=bias, batchnorm=BN)
        self.ec2 = self.encoder(16, 16, bias=bias, batchnorm=BN)
        self.ec3 = self.encoder(16, 32, bias=bias, batchnorm=BN)
        self.ec4 = self.encoder(32, 32, bias=bias, batchnorm=BN)
        self.ec5 = self.encoder(32, 64, bias=bias, batchnorm=BN)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)

        self.dc6 = self.decoder(64, 64, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
        self.dc5 = self.decoder(32 + 64, 32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc4 = self.decoder(32, 32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc3 = self.decoder(32, 32, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
        self.dc2 = self.decoder(16 + 32, 16, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc1 = self.decoder(16, 16, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        # self.dc0 = self.decoder(64, n_classes, kernel_size=1, stride=1, bias=False)
        self.dc0 = nn.Conv3d(16, n_classes, kernel_size=1, stride=1, padding=0, bias=bias)

        # self.weights_init()


    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                if not m.weight is None:
                    nn.init.xavier_normal(m.weight.data)
                if not m.bias is None:
                    m.bias.data.zero_()

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
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


    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
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

    def forward(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        e5 = self.ec5(e4)
        del e3, e4

        d6 = torch.cat((self.dc6(e5), syn1), dim=1)
        del e5, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0), dim=1)
        del d4, syn0

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2

        d0 = self.dc0(d1)
        return d0


class UNet_light3(nn.Module):
    def __init__(self, in_channel, n_classes, bias=False, BN=False):
        super(UNet_light3, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.ec0 = self.encoder(self.in_channel, 8, bias=bias, batchnorm=BN)
        self.ec1 = self.encoder(8, 16, bias=bias, batchnorm=BN)
        self.ec2 = self.encoder(16, 16, bias=bias, batchnorm=BN)
        self.ec3 = self.encoder(16, 32, bias=bias, batchnorm=BN)
        self.ec4 = self.encoder(32, 32, bias=bias, batchnorm=BN)
        self.ec5 = self.encoder(32, 32, bias=bias, batchnorm=BN)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)

        self.dc6 = self.decoder(32, 32, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
        self.dc5 = self.decoder(32 + 32, 32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc4 = self.decoder(32, 32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc3 = self.decoder(32, 16, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
        self.dc2 = self.decoder(16 + 16, 16, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc1 = self.decoder(16, 8, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        # self.dc0 = self.decoder(64, n_classes, kernel_size=1, stride=1, bias=False)
        self.dc0 = nn.Conv3d(8, n_classes, kernel_size=1, stride=1, padding=0, bias=bias)

        # self.weights_init()


    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                if not m.weight is None:
                    nn.init.xavier_normal(m.weight.data)
                if not m.bias is None:
                    m.bias.data.zero_()

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
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


    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
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

    def forward(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        e5 = self.ec5(e4)
        del e3, e4

        d6 = torch.cat((self.dc6(e5), syn1), dim=1)
        del e5, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0), dim=1)
        del d4, syn0

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2

        d0 = self.dc0(d1)
        return d0



class CascadedModel(nn.Module):
    """
    A cascaded model from a give model list
    Only train the last model and all other model are pre-trained.
    """
    def __init__(self, model_list, end2end=False, auto_context=True, residual=True, residual_scale=1.0):
        super(CascadedModel, self).__init__()
        self.models = nn.ModuleList(model_list)
        self.num_models = len(model_list)
        self.end2end = end2end
        self.auto_context = auto_context
        self.residual = residual
        self.residual_scale = residual_scale

        # freeze all models except the last one
        if not self.end2end:
            for ind in range(self.num_models-1):
                for param in self.models[ind].parameters():
                    param.requires_grad = False
                self.models[ind].eval()

        self.softmax = nn.Softmax(dim=1)

    def weights_init(self):
        if self.end2end:
            for m in self.modules():
                classname = m.__class__.__name__
                if classname.find('Conv') != -1:
                    if not m.weight is None:
                        nn.init.xavier_normal(m.weight.data)
                    if not m.bias is None:
                        m.bias.data.zero_()
        else:
            for m in self.models[-1].modules():
                classname = m.__class__.__name__
                if classname.find('Conv') != -1:
                    if not m.weight is None:
                        nn.init.xavier_normal(m.weight.data)
                    if not m.bias is None:
                        m.bias.data.zero_()


    def cascaded_eval(self):
        self.training = False
        if self.end2end:
            self.eval()
        else:
            self.models[-1].eval()

    def cascaded_train(self):
        self.training = True
        if self.end2end:
            self.train()
        else:
            self.models[-1].train()

    def forward(self, input, train=True, multi_output=True):
        """
        Forward though the cascased models.
        If using residual, each sub-model's output is added to the output of previous models
        If using auto-context, each sub-model's input is the concatenation of the raw input
        and the output of the previous sub-model
        :param input: input for the first model
        :param train: if training mode
        :return:the output of the last model
        """
        if multi_output:
            temp_output=[None]*self.num_models

            temp_output[0] = self.models[0](input)
            for i in range(1, self.num_models):
                if self.auto_context:
                    temp_input = Variable(torch.cat([self.softmax(temp_output[i-1]).data, input.data], dim=1),
                                          volatile=True if i<self.num_models-1 and not self.end2end else not train)
                else:
                    temp_input = Variable(input.data.cuda(), volatile=True if i<self.num_models-1 and not self.end2end else not train)

                if i == self.num_models-1 and not self.end2end and train:
                    temp_output[i - 1] = temp_output[i-1].detach()
                    temp_output[i-1].volatile=False

                if self.residual:
                    temp_output[i] = self.models[i](temp_input)*self.residual_scale + temp_output[i-1]
                else:
                    temp_output[i] = self.models[i](temp_input)
            return temp_output

        else:
            temp_output = self.models[0](input)
            for i in range(1, self.num_models):
                if self.auto_context:
                    temp_input = Variable(torch.cat([self.softmax(temp_output).data, input.data], dim=1),
                                          volatile=True if i < self.num_models - 1 and not self.end2end else not train)
                else:
                    temp_input = Variable(input.data.cuda(),
                                          volatile=True if i < self.num_models - 1 and not self.end2end else not train)

                if i == self.num_models-1 and not self.end2end and train:
                    temp_output = temp_output.detach()
                    temp_output.volatile=False

                if self.residual:
                    temp_output = self.models[i](temp_input) * self.residual_scale + temp_output
                else:
                    temp_output = self.models[i](temp_input)
            return temp_output

    def cascaded_parameters(self):
        if self.end2end:
            return self.parameters()
        else:
            return self.models[-1].parameters()