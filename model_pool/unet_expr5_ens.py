import torch
import torch.nn as nn
from torch.autograd import Function
class UNet3D5BE(nn.Module):
    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet3D5BE, self).__init__()
        self.ec0 = self.encoder(self.in_channel, 32, bias=True, batchnorm=True)
        self.ec1 = self.encoder(32, 64, bias=True, batchnorm=True)
        self.ec2 = self.encoder(64, 64, bias=True, batchnorm=True)
        self.ec3 = self.encoder(64, 128, bias=True, batchnorm=True)
        self.ec4 = self.encoder(128, 128, bias=True, batchnorm=True)
        self.ec5 = self.encoder(128, 256, bias=True, batchnorm=True)
        self.ec6 = self.encoder(256, 256, bias=True, batchnorm=True)
        self.ec7 = self.encoder(256, 256, bias=True, batchnorm=True)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.dc9 = self.decoder(256, 256, kernel_size=2, stride=2, bias=True, batchnorm=True)
        self.dc8 = self.decoder(256 + 256, 256, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)
        self.dc7 = self.decoder(256, 256, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)
        self.cd_low = self.cdecoder_7(256)
        self.dc6 = self.decoder(256, 256, kernel_size=2, stride=2, bias=True, batchnorm=True)
        self.dc5 = self.decoder(128 + 256, 128, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)
        self.dc4 = self.decoder(128, 128, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)
        self.cd_mid = self.cdecoder_4(128)
        self.dc3 = self.decoder(128, 128, kernel_size=2, stride=2, bias=True, batchnorm=True)
        self.dc2 = self.decoder(64 + 128, 128, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)
        self.dc1 = self.decoder(128, 128, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)
        self.dc0 = self.decoder(128, n_classes, kernel_size=1, stride=1, bias=True, batchnorm=True)
        # self.weights_init()
        # self.dc0 = self.decoder(64, n_classes, kernel_size=1, stride=1, bias=False)
        self.dc0_re = self.decoder_0(128, n_classes, kernel_size=1, stride=1, bias=False)
        self.confid_conv = self.combine_conv()
        self.adapt_conv = nn.Conv3d(n_classes*3, n_classes, 1, stride=1, groups=n_classes)



    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU(inplace=True))
        return layer


    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, bias=True,batchnorm=False):
        if  batchnorm:
            layer = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                   padding=padding, output_padding=output_padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                   padding=padding, output_padding=output_padding, bias=bias),
                nn.ReLU(inplace=True))
        return layer

    def cdecoder_7(self,in_channels):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, 128, 2, stride=2,bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ConvTranspose3d(128, self.n_classes, 2, stride=2, bias=True),
            nn.ReLU(inplace=True))

        return layer

    def cdecoder_4(self,in_channels):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, self.n_classes,  2, stride=2,bias=True))
        return layer

    def decoder_0(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias))
        return layer

    def combine_conv(self):
        layer = nn.Sequential(
            nn.ConvTranspose3d(self.n_classes*3, self.n_classes, 1, stride=1,padding=0))
        return layer


    def adpative_conv(self):
        layer = nn.Sequential(
            nn.Conv3d(self.n_classes*3, 2*self.n_classes, 1, stride=1, groups=self.n_classes),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.n_classes*2 ,  self.n_classes, 1, stride=1, groups=self.n_classes),
        )
        return layer
    def adapt_weighted_layer(self,input,n_class,n_map):
        """
            this function is to learning adaptive weighted for each class,
            meaning, each class channel (m class) of the N input maps would be reoganized into
            (ch1_in1,ch1_in2,.ch1_inN,ch2_in1,ch2_in2,...ch2_inN,.......,chm_inN)
            then an group convolution would be implemented on m group
        """
        permute_idx = [i+j*n_class for i in range(n_class) for j in range(n_map)]
        permute_input = input[:,permute_idx]
        output =self.adapt_conv(permute_input)
        return output





    def forward(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        #e2 += e1
        syn1 = self.ec3(e2)
        del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        #e4 += e3
        syn2 = self.ec5(e4)
        del e3, e4

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        #e6 += e5
        e7 = self.ec7(e6)
        del e5, e6

        d9 = torch.cat((self.dc9(e7), syn2), dim=1)
        del e7, syn2

        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        #d7 += d8
        c_low = self.cd_low(d7)
        del d9, d8

        d6 = torch.cat((self.dc6(d7), syn1), dim=1)
        del d7, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        #d4 += d5
        c_mid = self.cd_mid(d4)
        del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0), dim=1)
        del d4, syn0

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        #d1 += d2
        c_high = self.dc0_re(d1)  # 1. in 64 o 57
        del d3, d2

        d0 = torch.cat( (c_low ,c_mid,c_high),1)
        #d0 = self.confid_conv(d0)
        d0 = self.adapt_weighted_layer(d0,self.n_classes, 3)
        return d0


class AdaptWeightedClass(nn.Module):

    def __init__(self,n_class,n_map,sz):
        super(AdaptWeightedClass,self).__init__()
        self.n_class =n_class
        self.n_map = n_map
        self.concat_map = torch.zeros()


