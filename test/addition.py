import numpy as np
from torch.autograd import Variable
import torch
import torch.nn.functional as F

criterion = torch.nn.MSELoss()
batch_size = 3
dfdx = np.array([1., -1.], dtype='float32')
x = Variable(torch.randn(3, 2, 128, 128)).cuda()
recon_batch_variable = Variable(torch.randn(3, 2, 128, 128), requires_grad=True).cuda()
recon_batch_variable = x * recon_batch_variable
filter_x = (np.tile(dfdx, (batch_size, 1))).reshape(batch_size, 1, 1, 2)
filter_y = (np.tile(dfdx.reshape(2, 1), (batch_size, 1))).reshape(batch_size, 1, 2, 1)
filt_x = Variable(torch.from_numpy(filter_x), requires_grad=True).cuda()
filt_y = Variable(torch.from_numpy(filter_y)).cuda()
recon_batch_variable_grad_x = F.conv2d(recon_batch_variable, filt_x, groups=2)
recon_batch_variable_grad_y = F.conv2d(recon_batch_variable, filt_y, groups=2)
recon_batch_variable_grad_y.requires_grad = False
print(recon_batch_variable_grad_x.volatile)
print(recon_batch_variable_grad_x.requires_grad)
# criterion(recon_batch_variable, output_batch_variable)
recon_batch_variable_grad_x=torch.max(recon_batch_variable_grad_x).backward()
loss = 0.1 * criterion(recon_batch_variable_grad_x, recon_batch_variable_grad_y + 1)
#loss.backward(2)
