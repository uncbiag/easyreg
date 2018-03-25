import torch
import torch.nn as nn





class UNet3D_Deep(nn.Module):
    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet3D_Deep, self).__init__()
        self.e0cb = self.encoder(self.in_channel, 32, bias=True, batchnorm=True)
        self.e0c1 = self.encoder(32, 32, bias=True, batchnorm=True)
        self.e0c2 = self.encoder(32, 32, bias=True, batchnorm=True)
        self.e0c3 = self.encoder(32, 32, bias=True, batchnorm=True)
        self.e0ce = self.encoder(32, 32, bias=True, batchnorm=True)
        self.pool0 = nn.MaxPool3d(2)

        self.e1cb = self.encoder(32, 32, bias=True, batchnorm=True)
        self.e1c1 = self.encoder(32, 32, bias=True, batchnorm=True)
        self.e1c2 = self.encoder(32, 32, bias=True, batchnorm=True)
        self.e1c3 = self.encoder(32, 32, bias=True, batchnorm=True)
        self.e1c4 = self.encoder(32, 32, bias=True, batchnorm=True)
        self.e1ce = self.encoder(32, 64, bias=True, batchnorm=True)
        self.pool1 = nn.MaxPool3d(2)

        self.e2cb = self.encoder(64, 64, bias=True, batchnorm=True)
        self.e2c1 = self.encoder(64, 64, bias=True, batchnorm=True)
        self.e2c2 = self.encoder(64, 64, bias=True, batchnorm=True)
        self.e2c3 = self.encoder(64, 64, bias=True, batchnorm=True)
        self.e2c4 = self.encoder(64, 64, bias=True, batchnorm=True)
        self.e2c5 = self.encoder(64, 64, bias=True, batchnorm=True)
        self.e2ce = self.encoder(64, 64, bias=True, batchnorm=True)
        self.pool2 = nn.MaxPool3d(2)

        self.e3cb = self.encoder(64, 64, bias=True, batchnorm=True)
        self.e3c1 = self.encoder(64, 64, bias=True, batchnorm=True)
        self.e3c2 = self.encoder(64, 64, bias=True, batchnorm=True)
        self.e3c3 = self.encoder(64, 64, bias=True, batchnorm=True)
        self.e3c4 = self.encoder(64, 64, bias=True, batchnorm=True)
        self.e3c5 = self.encoder(64, 64, bias=True, batchnorm=True)
        self.e3ce = self.decoder(64, 64, kernel_size=2, stride=2, bias=True,batchnorm=True)

        self.d2cb = self.decoder(64 + 64, 32, kernel_size=3, stride=1, padding=1, bias=True,batchnorm=True)
        self.d2c1 = self.decoder(32, 32, kernel_size=3, stride=1, padding=1, bias=True,batchnorm=True)
        self.d2c2 = self.decoder(32, 32, kernel_size=3, stride=1, padding=1, bias=True,batchnorm=True)
        self.d2c3 = self.decoder(32, 32, kernel_size=3, stride=1, padding=1, bias=True,batchnorm=True)
        self.d2c4 = self.decoder(32, 32, kernel_size=3, stride=1, padding=1, bias=True,batchnorm=True)
        self.d2c5 = self.decoder(32, 32, kernel_size=3, stride=1, padding=1, bias=True,batchnorm=True)

        self.d2ce = self.decoder(32, 32, kernel_size=2, stride=2, bias=True,batchnorm=True)

        self.d1cb = self.decoder(32 + 64, 32, kernel_size=3, stride=1, padding=1, bias=True,batchnorm=True)
        self.d1c1 = self.decoder(32, 32, kernel_size=3, stride=1, padding=1, bias=True,batchnorm=True)
        self.d1c2 = self.decoder(32, 32, kernel_size=3, stride=1, padding=1, bias=True,batchnorm=True)
        self.d1c3 = self.decoder(32, 32, kernel_size=3, stride=1, padding=1, bias=True,batchnorm=True)
        self.d1c4 = self.decoder(32, 32, kernel_size=3, stride=1, padding=1, bias=True,batchnorm=True)
        self.d1ce = self.decoder(32, 32, kernel_size=2, stride=2, bias=True,batchnorm=True)

        self.d0cb = self.decoder(32 + 32, 32, kernel_size=3, stride=1, padding=1, bias=True,batchnorm=True)
        self.d0c1 = self.decoder(32, 32, kernel_size=3, stride=1, padding=1, bias=True,batchnorm=True)
        self.d0c2 = self.decoder(32, 32, kernel_size=3, stride=1, padding=1, bias=True,batchnorm=True)
        self.d0c3 = self.decoder(32, 32, kernel_size=3, stride=1, padding=1, bias=True,batchnorm=True)
        self.d0ce = self.decoder(32, n_classes, kernel_size=1, stride=1, bias=True,batchnorm=True)
        print("this is extreme deep unet")


        # self.weights_init()



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
                output_padding=0, bias=True,batchnorm=False):
        if  batchnorm:
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
        e0cb = self.e0cb(x)
        e0c1 = self.e0c1(e0cb)
        e0c2 = self.e0c2(e0c1)
        e0c3 = self.e0c3(e0c2)
        e0ce = self.e0ce(e0c3)
        e0_pool = self.pool0(e0ce)
        del e0cb

        e1cb = self.e1cb(e0_pool)
        e1c1 = self.e1c1(e1cb)
        e1c2 = self.e1c2(e1c1)
        e1c3 = self.e1c3(e1c2)
        e1c4 = self.e1c4(e1c3)
        e1ce = self.e1ce(e1c4)
        e1_pool = self.pool1(e1ce)
        del e0_pool, e1cb

        e2cb = self.e2cb(e1_pool)
        e2c1 = self.e2c1(e2cb)
        e2c2 = self.e2c2(e2c1)
        e2c3 = self.e2c3(e2c2)
        e2c4 = self.e2c4(e2c3)
        e2c5 = self.e2c5(e2c4)
        e2ce = self.e2ce(e2c5)
        e2_pool = self.pool2(e2ce)

        del e1_pool, e2cb

        e3cb = self.e3cb(e2_pool)
        e3c1 = self.e3c1(e3cb)
        e3c2 = self.e3c2(e3c1)
        e3c3 = self.e3c3(e3c2)
        e3c4 = self.e3c4(e3c3)
        e3c5 = self.e3c5(e3c4)
        e3ce = self.e3ce(e3c5)
        del e2_pool, e3cb, e3c1

        d2cc = torch.cat((e3ce, e2ce), dim=1)
        d2cb = self.d2cb(d2cc)
        d2c1 = self.d2c1(d2cb)
        d2c2 = self.d2c2(d2c1)
        d2c3 = self.d2c3(d2c2)
        d2c4 = self.d2c4(d2c3)
        d2c5 = self.d2c5(d2c4)
        d2ce = self.d2ce(d2c5)
        del e3ce,e2ce, d2cc,d2cb, d2c1

        d1cc = torch.cat((d2ce, e1ce), dim=1)
        d1cb = self.d1cb(d1cc)
        d1c1 = self.d1c1(d1cb)
        d1c2 = self.d1c2(d1c1)
        d1c3 = self.d1c3(d1c2)
        d1c4 = self.d1c4(d1c3)
        d1ce = self.d1ce(d1c4)
        del d2ce, e1ce, d1cc, d1cb,d1c1

        d0cc = torch.cat((d1ce, e0ce), dim=1)
        d0cb = self.d0cb(d0cc)
        d0c1 = self.d0c1(d0cb)
        d0c2 = self.d0c2(d0c1)
        d0c3 = self.d0c3(d0c2)
        d0ce = self.d0ce(d0c3)
        del d1ce, e0ce, d0cc, d0cb, d0c1

        return d0ce
