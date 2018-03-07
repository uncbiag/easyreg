def forward(self, x):
    e0 = self.ec0(x)  # 1. in 1 o:32
    syn0 = self.ec1(e0)  # 1. in 32 o:64
    e1 = self.pool0(syn0)  # .5 in 64  o 64
    e2 = self.ec2(e1)  # .5 in 64 o 64
    e2 += e1
    syn1 = self.ec3(e2)  # .5 in 64 o 128
    del e0, e1, e2

    e3 = self.pool1(syn1)  # .25 in 128 o 128
    e4 = self.ec4(e3)  # .25 in 128 o 128
    e4 += e3
    syn2 = self.ec5(e4)  # .25 in 128 o 256
    del e3, e4

    e5 = self.pool2(syn2)  # .125 in 256 o 256
    e6 = self.ec6(e5)  # .125 in 256 o 256
    e6 += e5
    e7 = self.ec7(e6)  # .125 in 256 o 512
    del e5, e6

    d9_up = self.dc9(e7)  # .25 in 512 o 512
    d9 = torch.cat([d9_up, self.center_crop(syn2, d9_up.size()[2:5])], 1)
    del d9_up, e7, syn2

    d8 = self.dc8(d9)  # .25 in  256+512 o 256
    d7 = self.dc7(d8)  # .25 in 256 o 256
    d7 += d8
    c_low = self.cd_low(d7)
    del d9, d8

    dc6_up = self.dc6(d7)  # .5 in 256 o 256
    d6 = torch.cat([dc6_up, self.center_crop(syn1, dc6_up.size()[2:5])], 1)
    del dc6_up, d7, syn1

    d5 = self.dc5(d6)  # .5 in 128+256 o 128
    d4 = self.dc4(d5)  # .5 in 128 o 128
    d4 += d5
    c_mid = self.cd_mid(d4)

    del d6, d5

    dc3_up = self.dc3(d4)  # 1. in 128 o 128
    d3 = torch.cat([dc3_up, self.center_crop(syn0, dc3_up.size()[2:5])], 1)
    del dc3_up, d4, syn0

    d2 = self.dc2(d3)  # 1. in 128+64 o 64
    d1 = self.dc1(d2)  # 1. in 64  o 64
    d1 += d2
    del d3, d2
    c_high = self.dc0_re(d1)  # 1. in 64 o 57
    d0 = 20 * c_low + 5 * c_mid + c_high

    # interpolate to original output dimensions
    s0 = x.size()[2]
    s1 = x.size()[3]
    s2 = x.size()[4]
    self.interp = nn.Upsample(size=(s0, s1, s2), mode='trilinear')
    d0 = self.interp(d0)
    return d0


def center_crop(self, layer, target_sizes):
    batch_size, n_channels, dim1, dim2, dim3 = layer.size()
    dim1_c = (dim1 - target_sizes[0]) // 2
    dim2_c = (dim2 - target_sizes[1]) // 2
    dim3_c = (dim3 - target_sizes[2]) // 2
    return layer[:, :, dim1_c:dim1_c + target_sizes[0], dim2_c:dim2_c + target_sizes[1],
           dim3_c:dim3_c + target_sizes[2]]