from data.data_utils import *

import torch
from torch.autograd import Variable

val1= Variable(torch.Tensor(10,10))

val2 = 'string'

val_dict = {'data': val1.data.numpy(), 'info':val2}

write_file('test.h5py',val_dict)

result_dict = read_file('test.h5py')
print('finished')
