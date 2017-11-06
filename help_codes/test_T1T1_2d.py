from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.serialization import load_lua
import prediction_network_2d
import util_2d
import numpy as np
import argparse
import time
import h5py
from torch.autograd import Variable

start = time.time()

print('config for testing')
print('Loading configuration and network')
config = torch.load('/playpen2/pytorch_lddmm_data/parameters/checkpoint_10.pth.tar');
patch_size = config['patch_size']
network_feature = config['network_feature']

batch_size = 32*8

step_size = 14
use_multiGPU = False;

print('creating net')
net_single = prediction_network_2d.net(network_feature).cuda();
net_single_state = net_single.state_dict();

net_single.load_state_dict(config['state_dict'])

if use_multiGPU:
    print('multi GPU net')
    device_ids=range(0, 8)
    net = torch.nn.DataParallel(net_single, device_ids=device_ids).cuda()
else:
    net = net_single

net.train()

input_batch = torch.zeros(batch_size, 2, patch_size, patch_size).cuda()
# input_batch_variable = Variable(input_batch, volatile=True)
# temp_output = net(input_batch_variable)
# temp_output = None

base_idx = 1    
image_from_dataset = util_2d.readHDF5("/playpen2/pytorch_lddmm_data/test_source.h5").float()
image_to_dataset = util_2d.readHDF5("/playpen2/pytorch_lddmm_data/test_lambda.h5").float()
dataset_size = image_from_dataset.size()
for slice_idx in range(0, dataset_size[0]):
    image_from_slice = image_from_dataset[slice_idx];
    image_to_slice = image_to_dataset[slice_idx];
    predict_result = util_2d.predict_momentum(image_from_slice, image_to_slice, input_batch, batch_size, patch_size, net);
    predict_result = predict_result.numpy();
    f = h5py.File("/playpen2/pytorch_lddmm_data/test_results/pv_" + str(base_idx) + ".h5", "w") 
    dset = f.create_dataset("dataset", data=predict_result)
    f.close()
    base_idx += 1

# base_idx = 1;
# for datapart in range(1, 11):
#     image_from_dataset = util.readHDF5("/pine/scr/x/y/xy/martin_multimodal/T1/autism_t1_single_split_from_" + str(datapart) + ".h5").float()
#     image_to_dataset = util.readHDF5("/pine/scr/x/y/xy/martin_multimodal/T2/autism_t2_single_split_to_" + str(datapart) + ".h5").float()
#     dataset_size = image_from_dataset.size()
#     for slice_idx in range(0, dataset_size[0]):
#         image_from_slice = image_from_dataset[slice_idx];
#         image_to_slice = image_to_dataset[slice_idx];
#         predict_result = util.predict_momentum(image_from_slice, image_to_slice, input_batch, batch_size, patch_size, net);
#         predict_result = predict_result.numpy();
#         f = h5py.File("/pine/scr/x/y/xy/martin_multimodal/predict_momentums/T2/"+str(base_idx)+".h5", "w") 
#         dset = f.create_dataset("dataset", data=predict_result)
#         f.close()
#         base_idx += 1;
# image_appear_set = load_lua('/pine/scr/x/y/xy/OASIS_image_data/IBSR.t7')

# for test_from in range(0, 18) :
#     for test_to in range(0, 18) :
#         image_from = image_appear_set[test_from].squeeze()
#         image_to = image_appear_set[test_to].squeeze()
#         predict_result = util.predict_momentum(image_from, image_to, input_batch, batch_size, patch_size, net);
#         predict_result = predict_result.numpy();

#         f = h5py.File("test.h5", "w") 
#         dset = f.create_dataset("dataset", data=predict_result)
#         f.close()

# #endfor
# end = time.time()
# print((end-start)/18/18);






