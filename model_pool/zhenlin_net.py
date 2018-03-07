import torch
import torch.nn as nn
from torch.autograd import Variable

# 3D UNet and its variants

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


class UNet(nn.Module):
    def __init__(self, in_channel, n_classes, bias=False, BN=False):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet, self).__init__()
        self.ec0 = self.encoder(self.in_channel, 32, bias=bias, batchnorm=BN)
        self.ec1 = self.encoder(32, 64, bias=bias, batchnorm=BN)
        self.ec2 = self.encoder(64, 64, bias=bias, batchnorm=BN)
        self.ec3 = self.encoder(64, 128, bias=bias, batchnorm=BN)
        self.ec4 = self.encoder(128, 128, bias=bias, batchnorm=BN)
        self.ec5 = self.encoder(128, 256, bias=bias, batchnorm=BN)
        self.ec6 = self.encoder(256, 256, bias=bias, batchnorm=BN)
        self.ec7 = self.encoder(256, 512, bias=bias, batchnorm=BN)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.dc9 = self.decoder(512, 512, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
        self.dc8 = self.decoder(256 + 512, 256, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc7 = self.decoder(256, 256, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc6 = self.decoder(256, 256, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
        self.dc5 = self.decoder(128 + 256, 128, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc4 = self.decoder(128, 128, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc3 = self.decoder(128, 128, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
        self.dc2 = self.decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc1 = self.decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        # self.dc0 = self.decoder(64, n_classes, kernel_size=1, stride=1, bias=False)
        self.dc0 = nn.Conv3d(64, n_classes, kernel_size=1, stride=1, padding=0, bias=bias)

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
        syn2 = self.ec5(e4)
        del e3, e4

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)
        del e5, e6

        d9 = torch.cat((self.dc9(e7), syn2), dim=1)
        del e7, syn2

        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        del d9, d8

        d6 = torch.cat((self.dc6(d7), syn1), dim=1)
        del d7, syn1

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



class AutoContextAsRNN(nn.Module):
    def __init__(self, in_channel, n_classes, FCN, residual=0, BN=False, bias=False):
        super(AutoContextAsRNN, self).__init__()
        self.n_classes = n_classes
        self.in_channel = in_channel
        self.FCN = FCN(n_classes=self.n_classes, in_channel=self.in_channel, BN=BN, bias=bias)
        self.softmax = nn.Softmax(dim=1)
        self.residual = residual

    def forward(self, input, hidden, residual_scale=1.0):
        combined_input = torch.cat((input, self.softmax(hidden)), 1)
        if not self.residual:
            hidden = self.FCN(combined_input)
        elif self.residual == 1:
            hidden = hidden + self.FCN(combined_input)*residual_scale

        output = self.softmax(hidden)
        return output, hidden

    def init_hidden(self, hidden_size):
        return Variable(torch.ones(hidden_size) * 0.5)

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                if not m.weight is None:
                    nn.init.xavier_normal(m.weight.data)
                if not m.bias is None:
                    m.bias.data.zero_()

class UNet_light1(nn.Module):
    def __init__(self, in_channel, n_classes, bias=False, BN=False):
        super(UNet_light1, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.ec0 = self.encoder(self.in_channel, 16, bias=bias, batchnorm=BN)
        self.ec1 = self.encoder(16, 32, bias=bias, batchnorm=BN)
        self.ec2 = self.encoder(32, 32, bias=bias, batchnorm=BN)
        self.ec3 = self.encoder(32, 64, bias=bias, batchnorm=BN)
        self.ec4 = self.encoder(64, 64, bias=bias, batchnorm=BN)
        self.ec5 = self.encoder(64, 128, bias=bias, batchnorm=BN)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        # self.pool2 = nn.MaxPool3d(2)

        self.dc6 = self.decoder(128, 128, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
        self.dc5 = self.decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc4 = self.decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc3 = self.decoder(64, 64, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
        self.dc2 = self.decoder(32 + 64, 32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc1 = self.decoder(32, 32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        # self.dc0 = self.decoder(64, n_classes, kernel_size=1, stride=1, bias=False)
        self.dc0 = nn.Conv3d(32, n_classes, kernel_size=1, stride=1, padding=0, bias=bias)

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


class UNet_light4(nn.Module):
    def __init__(self, in_channel, n_classes, bias=False, BN=False):
        super(UNet_light4, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.ec0 = self.encoder(self.in_channel, 8, bias=bias, batchnorm=BN)
        self.ec1 = self.encoder(8, 16, bias=bias, batchnorm=BN)
        self.ec2 = self.encoder(16, 16, bias=bias, batchnorm=BN)
        self.ec3 = self.encoder(16, 32, bias=bias, batchnorm=BN)


        self.pool0 = nn.MaxPool3d(2)
        # self.pool1 = nn.MaxPool3d(2)

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
        e3 = self.ec3(e2)
        del e0, e1, e2

        d3 = torch.cat((self.dc3(e3), syn0), dim=1)
        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2

        d0 = self.dc0(d1)
        return d0


class UNet_light4_2(nn.Module):
    def __init__(self, in_channel, n_classes, bias=False, BN=False):
        super(UNet_light4_2, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.ec0 = encoder(self.in_channel, 8, bias=bias, batchnorm=BN)
        self.ec1 = encoder(8, 16, bias=bias, batchnorm=BN)
        self.ec2 = encoder(16, 16, bias=bias, batchnorm=BN)
        self.ec3 = encoder(16, 32, bias=bias, batchnorm=BN)


        self.pool0 = nn.MaxPool3d(2)
        # self.pool1 = nn.MaxPool3d(2)

        self.dc3 = decoder(32, 32, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
        self.dc2 = decoder(32 + 16, 16, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc1 = decoder(16, 8, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
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

    def forward(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        e3 = self.ec3(e2)
        del e0, e1, e2

        d3 = torch.cat((self.dc3(e3), syn0), dim=1)
        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2

        d0 = self.dc0(d1)
        return d0



class UNet_light4x2(nn.Module):
    def __init__(self, in_channel, n_classes, bias=False, BN=False):
        super(UNet_light4x2, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.m1_ec0 = encoder(self.in_channel, 8, bias=bias, batchnorm=BN)
        self.m1_ec1 = encoder(8, 16, bias=bias, batchnorm=BN)
        self.m1_ec2 = encoder(16, 16, bias=bias, batchnorm=BN)
        self.m1_ec3 = encoder(16, 32, bias=bias, batchnorm=BN)
        self.m1_pool0 = nn.MaxPool3d(2)
        self.m1_dc3 = decoder(32, 16, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
        self.m1_dc2 = decoder(16 + 16, 16, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.m1_dc1 = decoder(16, 8, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.m1_dc0 = nn.Conv3d(8, self.n_classes, kernel_size=1, stride=1, padding=0, bias=bias)

        self.softmax = nn.Softmax(dim=1)
        self.m2_ec0 = encoder(self.in_channel+self.n_classes, 8, bias=bias, batchnorm=BN)
        self.m2_ec1 = encoder(8, 16, bias=bias, batchnorm=BN)
        self.m2_ec2 = encoder(16, 16, bias=bias, batchnorm=BN)
        self.m2_ec3 = encoder(16, 32, bias=bias, batchnorm=BN)
        self.m2_pool0 = nn.MaxPool3d(2)
        self.m2_dc3 = decoder(32, 16, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
        self.m2_dc2 = decoder(16 + 16, 16, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.m2_dc1 = decoder(16, 8, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.m2_dc0 = nn.Conv3d(8, n_classes, kernel_size=1, stride=1, padding=0, bias=bias)


    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                if not m.weight is None:
                    nn.init.xavier_normal(m.weight.data)
                if not m.bias is None:
                    m.bias.data.zero_()

    def forward(self, input):
        m1_e0 = self.m1_ec0(input)
        m1_syn0 = self.m1_ec1(m1_e0)
        m1_e1 = self.m1_pool0(m1_syn0)
        m1_e2 = self.m1_ec2(m1_e1)
        m1_e3 = self.m1_ec3(m1_e2)
        del m1_e0, m1_e1, m1_e2

        m1_d3 = torch.cat((self.m1_dc3(m1_e3), m1_syn0), dim=1)
        m1_d2 = self.m1_dc2(m1_d3)
        m1_d1 = self.m1_dc1(m1_d2)
        del m1_d3, m1_d2

        m1_d0 = self.m1_dc0(m1_d1)
        del m1_d1

        m2_input = torch.cat((self.softmax(m1_d0), input), dim=1)
        m2_e0 = self.m2_ec0(m2_input)
        m2_syn0 = self.m2_ec1(m2_e0)
        m2_e1 = self.m2_pool0(m2_syn0)
        m2_e2 = self.m2_ec2(m2_e1)
        m2_e3 = self.m2_ec3(m2_e2)
        del m2_e0, m2_e1, m2_e2

        m2_d3 = torch.cat((self.m2_dc3(m2_e3), m2_syn0), dim=1)
        m2_d2 = self.m2_dc2(m2_d3)
        m2_d1 = self.m2_dc1(m2_d2)
        del m2_d3, m2_d2

        m2_d0 = self.m2_dc0(m2_d1)+m1_d0
        del m2_d1

        return m2_d0


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