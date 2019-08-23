


import torch
import torch.nn as nn







def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)


        return out

def cal_loss(input,moving):
    return ((input-moving)**2).sum()





def _symmetrize_filter_center_at_zero_2D(filter):
    sz = filter.shape
    h_sz0 = sz[2] // 2
    h_sz1 = sz[3] // 2
    filter_fliped = filter.clone()
    if sz[2] % 2 == 0:
        # symmetrize if it is even
        filter_fliped[:,:,:h_sz0,h_sz1:] = torch.flip(filter[:,:,h_sz0:,h_sz1:],[2])
    if sz[3] % 2 == 0:
        filter_fliped[:,:,:,:h_sz1] = torch.flip(filter_fliped[:,:,:,h_sz1:],[3])
    return filter_fliped



def symmetrize_filter_center_at_zero(filter,renormalize=False):
    """
    Symmetrizes filter. The assumption is that the filter is already in the format for input to an FFT.
    I.e., that it has been transformed so that the center of the pixel is at zero.

    :param filter: Input filter (in spatial domain). Will be symmetrized (i.e., will change its value)
    :param renormalize: (bool) if true will normalize so that the sum is one
    :return: n/a (returns via call by reference)
    """
    sz = filter.shape
    dim = len(sz[2:])
    if dim==2:
        filter_fliped=  _symmetrize_filter_center_at_zero_2D(filter)
    else:
        raise ValueError('Only implemented for dimensions 1,2, and 3 so far')

    if renormalize:
        filter_fliped = filter_fliped / filter_fliped.sum()
    return filter_fliped