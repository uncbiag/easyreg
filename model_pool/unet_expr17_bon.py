import torch
import torch.nn as nn
from torch.autograd import Function
class UNet3DB17(nn.Module):
    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet3DB17, self).__init__()
        self.ec0 = self.encoder(self.in_channel, 128, bias=True, batchnorm=True)
        self.ec1 = self.encoder(128, 128, bias=True, batchnorm=True)
        self.pool0 = nn.MaxPool3d(2)
        self.ec2 = self.encoder(128, 128, bias=True, batchnorm=True)
        self.ec3 = self.encoder(128, 128, bias=True, batchnorm=True)
        self.pool1 = nn.MaxPool3d(2)
        self.ec4 = self.encoder(128, 128, bias=True, batchnorm=True)
        self.ec5 = self.encoder(128, 128, bias=True, batchnorm=True)


        self.cmid1 = self.cmid(128,128, bias=True)
        self.cmid2 = self.cmid(128,128, bias=True)
        self.cmid3 = self.cmid(128,128, bias=True)

        self.pool2 = nn.MaxPool3d(2)
        self.ec6 = self.encoder(128, 128, bias=True, batchnorm=True)
        self.ec7 = self.encoder(128, 128, bias=True, batchnorm=True)
        self.dc9 = self.decoder(128, 128, kernel_size=2, stride=2, bias=True, batchnorm=True)

        self.dc8_pu = self.decoder(128, 128, kernel_size=1, stride=1, padding=0, bias=True, batchnorm=True)
        self.ud2 = self.udecoder_2(128,256)   #  in dc8_pu
        self.dc6 = self.decoder(128, 128, kernel_size=2, stride=2, bias=True, batchnorm=True)
        self.dc5_pu = self.decoder(128, 128, kernel_size=1, stride=1, padding=0, bias=True, batchnorm=True)
        self.ud1 = self.udecoder_1(128,256)
        self.dc2_pu = self.decoder(128, 256, kernel_size=1, stride=1, padding=0, bias=True, batchnorm=True)
        self.ud0 = self.udecoder_0(256)
        self.count=0


        print("this is b17")




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
    def ddecoder_1(self, in_channels,out_channels, bias=True):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=2,
                               padding=1,  bias=bias),
            nn.ReLU(inplace=True))
        return layer

    def ddecoder_2(self, in_channels,out_channels, bias=True):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 5, stride=4,
                               padding=2, bias=bias),
            nn.ReLU(inplace=True))
        return layer

    def cmid(self, in_channels, out_channels, bias=True):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=1,
                      padding=1, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, 3, stride=1,
                      padding=1, bias=bias),
            nn.ReLU(inplace=True))
        return layer

    def udecoder_2(self,in_channels,out_channels, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2,bias=bias),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2,bias=bias),
            #  To Do  maybe here should be exchanged
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(in_channels),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            # To Do  comment bias
            nn.Conv3d(out_channels, self.n_classes, 1, stride=1, bias=False))
        return layer

    def udecoder_1(self,in_channels,out_channels, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels,  2, stride=2,bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, self.n_classes, 1, stride=1, bias=False))
        return layer

    def udecoder_0(self, in_channels):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, self.n_classes, 1, stride=1,bias=False))
        return layer

    def combine_conv(self):
        layer = nn.Sequential(
            nn.ConvTranspose3d(self.n_classes*3, self.n_classes*2, 1, stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(self.n_classes * 2, self.n_classes, 1, stride=1, padding=0)
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
    def gate_module(self, input1,input2,input3):
        self.adpative_th(th1=0.4, th2=0.6)
        from torch.nn.functional import softmax, threshold
        input1_s=torch.max(softmax(input1,1),1,keepdim=True)[0]
        input1_s = threshold(input1_s,self.th1,0.1,inplace=True)
        input2 = (1-input1_s)*input2+input1_s*input1
        input2_s = torch.max(softmax(input2,1),1,keepdim=True)[0]
        input2_s = threshold(input2_s,self.th2,0.1,inplace=True)
        input3= (1-input2_s)*input3+ input2_s*input2
        if self.count%400==0:
            print("the th1 is:,  the th2 is:", self.th1, self.th2)
            print("the input1 is {}, the input2 is {}".format(input1[0,0,32],input2[0,0,32]))
            print("input1_s: {}, input2_s:{}".format(input1_s[0,0,32],input2_s[0,0,32]))
        return input3

    def adpative_th(self, th1, th2):
        th = self.count/15000.
        self.th1 = min(th, th1)
        self.th2 = min(th, th2)









    def forward(self, x):

        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
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
        del e5, e6
        mid3 = self.cmid3(syn2)
        del syn2

        d8_pu = self.dc8_pu(mid3)
        #d7 += d8
        dc_low = self.ud2(d8_pu)
        del  mid3,d8_pu

        mid2 = self.cmid2(syn1)

        del syn1
        d5_pu = self.dc5_pu(mid2)

        #d4 += d5
        dc_mid = self.ud1(d5_pu)
        del mid2, d5_pu

        mid1 =self.cmid1(syn0)
        del syn0

        d2_pu = self.dc2_pu(mid1)
        #d1 += d2
        dc_high = self.ud0(d2_pu)  # 1. in 64 o 57
        del  d2_pu,mid1

        d0 = self.gate_module(dc_low ,dc_mid,dc_high)
        self.count+=1

        return d0


class AdaptWeightedClass(nn.Module):

    def __init__(self,n_class,n_map,sz):
        super(AdaptWeightedClass,self).__init__()
        self.n_class =n_class
        self.n_map = n_map
        self.concat_map = torch.zeros()


