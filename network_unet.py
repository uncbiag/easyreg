
import torch
import torch.nn as nn
import torch.nn.functional as F


dim = 3
Conv = nn.Conv2d if dim==2 else nn.Conv3d
MaxPool = nn.MaxPool2d if dim ==2 else nn.MaxPool3d
ConvTranspose = nn.ConvTranspose2d if dim==2 else nn.ConvTranspose3d
BatchNorm = nn.BatchNorm2d if dim==2 else nn.BatchNorm3d
conv = F.conv2d if dim==2 else F.conv3d


class conv_bn_rel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, active_unit='relu', same_padding=False,
                 bn=False, reverse=False,group = 1,dilation = 1):
        super(conv_bn_rel, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        if not reverse:
            self.conv = Conv(in_channels, out_channels, kernel_size, stride, padding=padding, groups=group,dilation=dilation)
        else:
            self.conv = ConvTranspose(in_channels, out_channels, kernel_size, stride, padding=padding,groups=group,dilation=dilation)

        self.bn = BatchNorm(out_channels, eps=0.0001, momentum=0, affine=True) if bn else None
        if active_unit == 'relu':
            self.active_unit = nn.ReLU(inplace=True)
        elif active_unit == 'elu':
            self.active_unit = nn.ELU(inplace=True)
        elif active_unit =='leaky_relu':
            self.active_unit = nn.LeakyReLU(inplace=True)
        else:
            self.active_unit = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.active_unit is not None:
            x = self.active_unit(x)
        return x



class FcRel(nn.Module):
    # fc+ relu(option)
    def __init__(self, in_features, out_features, active_unit='relu'):
        super(FcRel, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        if active_unit == 'relu':
            self.active_unit = nn.ReLU(inplace=True)
        elif active_unit == 'elu':
            self.active_unit = nn.ELU(inplace=True)
        else:
            self.active_unit = None

    def forward(self, x):
        x = self.fc(x)
        if self.active_unit is not None:
            x = self.active_unit(x)
        return x



###########################################


class Unet(nn.Module):
    """
    unet include 4 down path (1/16)  and 4 up path (16)
    """
    def __init__(self, n_in_channel, n_out_channel):
        # each dimension of the input should be 16x
        super(Unet,self).__init__()
        self.down_path_1 = conv_bn_rel(n_in_channel, 16, 3, stride=1, active_unit='relu', same_padding=True, bn=False)
        self.down_path_2_1 = conv_bn_rel(16, 32, 3, stride=2, active_unit='relu', same_padding=True, bn=False)
        self.down_path_2_2 = conv_bn_rel(32, 32, 3, stride=1, active_unit='relu', same_padding=True, bn=False)
        self.down_path_4_1 = conv_bn_rel(32, 32, 3, stride=2, active_unit='relu', same_padding=True, bn=False)
        self.down_path_4_2 = conv_bn_rel(32, 32, 3, stride=1, active_unit='relu', same_padding=True, bn=False)
        self.down_path_8_1 = conv_bn_rel(32, 64, 3, stride=2, active_unit='relu', same_padding=True, bn=False)
        self.down_path_8_2 = conv_bn_rel(64, 64, 3, stride=1, active_unit='relu', same_padding=True, bn=False)
        self.down_path_16 = conv_bn_rel(64, 64, 3, stride=2, active_unit='relu', same_padding=True, bn=False)


        # output_size = strides * (input_size-1) + kernel_size - 2*padding
        self.up_path_8_1 = conv_bn_rel(64, 64, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=False,reverse=True)
        self.up_path_8_2= conv_bn_rel(128, 64, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=False)
        self.up_path_4_1 = conv_bn_rel(64, 64, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=False,reverse=True)
        self.up_path_4_2 = conv_bn_rel(96, 32, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=False)
        self.up_path_2_1 = conv_bn_rel(32, 32, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=False,reverse=True)
        self.up_path_2_2 = conv_bn_rel(64, 8, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=False)
        self.up_path_1_1 = conv_bn_rel(8, 8, 2, stride=2, active_unit='None', same_padding=False, bn=False, reverse=True)
        self.up_path_1_2 = conv_bn_rel(24, n_out_channel, 3, stride=1, active_unit='None', same_padding=True, bn=False)

    def forward(self, x):
        d1 = self.down_path_1(x)
        d2_1 = self.down_path_2_1(d1)
        d2_2 = self.down_path_2_2(d2_1)
        d4_1 = self.down_path_4_1(d2_2)
        d4_2 = self.down_path_4_2(d4_1)
        d8_1 = self.down_path_8_1(d4_2)
        d8_2 = self.down_path_8_2(d8_1)
        d16 = self.down_path_16(d8_2)


        u8_1 = self.up_path_8_1(d16)
        u8_2 = self.up_path_8_2(torch.cat((d8_2,u8_1),1))
        u4_1 = self.up_path_4_1(u8_2)
        u4_2 = self.up_path_4_2(torch.cat((d4_2,u4_1),1))
        u2_1 = self.up_path_2_1(u4_2)
        u2_2 = self.up_path_2_2(torch.cat((d2_2, u2_1), 1))
        u1_1 = self.up_path_1_1(u2_2)
        output = self.up_path_1_2(torch.cat((d1, u1_1), 1))

        return output






