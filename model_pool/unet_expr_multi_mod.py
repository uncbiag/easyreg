import torch
import torch.nn as nn


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


def decoder(in_channels, out_channels, kernel_size, stride=1, padding=0,
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


class Fea_part():
    def initial(self, in_channels):
        self.e0cb = encoder(in_channels, 32, bias=True, batchnorm=True)
        self.e0c1 = encoder(32, 32, bias=True, batchnorm=True)
        self.e0ce = encoder(32, 32, bias=True, batchnorm=True)
        self.e0_sq =  nn.ModuleList([nn.Sequential(self.e0cb, self.e0c1, self.e0ce)])
        return self.e0_sq


class Mod_part():
    def initial(self, n_classes):
        self.pool0 = nn.MaxPool3d(2)
        self.e1cb = encoder(32, 32, bias=True, batchnorm=True)
        self.e1ce = encoder(32, 64, bias=True, batchnorm=True)
        self.e1_sq = nn.Sequential(self.pool0, self.e1cb, self.e1ce)

        self.pool1 = nn.MaxPool3d(2)
        self.e2cb = encoder(64, 64, bias=True, batchnorm=True)
        self.e2ce = encoder(64, 64, bias=True, batchnorm=True)
        self.e2_sq = nn.Sequential(self.pool1, self.e2cb, self.e2ce)

        self.pool2 = nn.MaxPool3d(2)
        self.e3cb = encoder(64, 64, bias=True, batchnorm=True)
        self.e3c1 = encoder(64, 64, bias=True, batchnorm=True)
        self.e3ce = decoder(64, 64, kernel_size=2, stride=2, bias=True, batchnorm=True)
        self.e3_sq = nn.Sequential(self.pool2, self.e3cb, self.e3c1, self.e3ce)

        self.d2cb = decoder(64 + 64, 64, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)
        self.d2c1 = decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)
        self.d2ce = decoder(64, 64, kernel_size=2, stride=2, bias=True, batchnorm=True)
        self.d2_sq = nn.Sequential(self.d2cb, self.d2c1, self.d2ce)

        self.d1cb = decoder(64 + 64, 64, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)
        self.d1c1 = decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)
        self.d1ce = decoder(64, 64, kernel_size=2, stride=2, bias=True, batchnorm=True)
        self.d1_sq = nn.Sequential(self.d1cb, self.d1c1, self.d1ce)

        self.d0cb = decoder(64 + 32, 64, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)
        self.d0c1 = decoder(64, 32, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)
        self.d0ce = decoder(32, n_classes, kernel_size=1, stride=1, bias=True, batchnorm=True)
        self.d0_sq = nn.Sequential(self.d0cb, self.d0c1, self.d0ce)
        self.mod_sq = nn.ModuleList([self.e1_sq, self.e2_sq, self.e3_sq, self.d2_sq, self.d1_sq, self.d0_sq])
        return self.mod_sq




class UNet3DMM(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(UNet3DMM, self).__init__()

        self.in_channel = in_channel
        self.n_classes = n_classes
        self.fea_list = nn.ModuleList([Fea_part().initial(in_channels=1) for _ in range(self.in_channel)])
        self.mod_list = nn.ModuleList([Mod_part().initial(n_classes=n_classes) for _ in range(self.in_channel)])
        print("this is Unet 3d Multi Modality")


        # self.weights_init()


    def disc_layer(self, input):
        # input.shape  BxModxCxXxYxZ
        # output.shape  BxCxXxYxZ
        output = torch.max(input, 1, keepdim=False)[0]
        return output






    def forward(self, x):
        output= []
        for i in range(self.in_channel):
            e0_sq_res = self.fea_list[i][0](x[:,i:i+1])
            e1_sq_res = self.mod_list[i][0](e0_sq_res)
            e2_sq_res = self.mod_list[i][1](e1_sq_res)
            e3_sq_res = self.mod_list[i][2](e2_sq_res)
            d2_sq_res = self.mod_list[i][3](torch.cat((e3_sq_res,e2_sq_res),dim=1))
            d1_sq_res = self.mod_list[i][4](torch.cat((d2_sq_res,e1_sq_res),dim=1))
            d0_sq_res = self.mod_list[i][5](torch.cat((d1_sq_res,e0_sq_res),dim=1))
            output.append(d0_sq_res)
        #output = [self.mod_module_list[i](self.fea_mod_list[i](x[:,i:i+1])) for i in range(self.in_channel)]
        stack_output = torch.stack(output,1)
        output = self.disc_layer(stack_output)

        return output
