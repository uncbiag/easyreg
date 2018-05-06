import torch
import torch.nn as nn

class PriorNet(nn.Module):
    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(PriorNet, self).__init__()
        self.ec0 = self.encoder(self.in_channel, 32,kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)
        self.ec1 = self.encoder(32, 64,kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)
        self.ec2 = self.encoder(64, 64,kernel_size=3, stride=2, padding=1, bias=True, batchnorm=True)
        self.ec3 = self.encoder(64, 128,kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)  # 2, 128 ,18 30, 30
        self.ec4 = self.encoder(128, 128,kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)  # 2, 128 ,18 30, 30
        self.max_pool1 = nn.AdaptiveMaxPool3d((18,18,18))
        self.ec5 = self.encoder(128, 128,kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)
        self.ec6 = self.encoder(128, 256,kernel_size=3, stride=2, padding=1, bias=True, batchnorm=True)
        self.ec7 = self.encoder(256, 256,kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)
        self.ec8 = self.encoder(256, 256,kernel_size=3, stride=2, padding=1, bias=True, batchnorm=True)
        self.max_pool2 = nn.AdaptiveMaxPool3d(1)
        self.linear1 = nn.Linear(256,32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32,n_classes)


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



    def forward(self, x):
        output = self.ec0(x)
        output = self.ec1(output)
        output = self.ec2(output)
        output = self.ec3(output)
        output = self.ec4(output)
        output = self.max_pool1(output)
        output = self.ec5(output)
        output = self.ec6(output)
        output = self.ec7(output)
        output = self.ec8(output)
        output = self.max_pool2(output)
        output = output.view(output.size(0),-1)
        output = self.linear1(output)
        output = self.relu(output)
        output = self.linear2(output)

        return output


