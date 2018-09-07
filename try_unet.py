import torch
from network_unet import *

input = torch.rand(2,3,128,128,128)
network = Unet(n_in_channel=3,n_out_channel=5)

output = network(input)
print(output.shape)