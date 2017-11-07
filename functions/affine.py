import torch
from torch.autograd import Function
import numpy as np
from torch.nn import Module
class AffineGridGenFunction(Function):
    def __init__(self, height, width):
        super(AffineGridGenFunction, self).__init__()
        self.height, self.width = height, width
        self.grid = np.zeros( [3,self.height, self.width], dtype=np.float32)
        # here the y is the first element then the x then the one
        self.grid[0,...] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.height), 0), repeats = self.width, axis = 0).T, 0)
        self.grid[1,...] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.width), 0), repeats = self.height, axis = 0), 0)
        self.grid[2,...] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        #print(self.grid)

    def forward(self, input1):
        self.input1 = input1  #1*2*3
        output = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size()) # batch_num * channel* height * width   output 1*3*64*128 grid: 3*64*128
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid

        if input1.is_cuda:
            self.batchgrid = self.batchgrid.cuda()
            output = output.cuda()

        for i in range(input1.size(0)):
                output = torch.bmm(self.batchgrid.view(-1, self.height*self.width, 3), torch.transpose(input1, 1, 2)).view(-1, self.height, self.width, 2)

        return output

    def backward(self, grad_output):

        grad_input1 = torch.zeros(self.input1.size())

        if grad_output.is_cuda:
            self.batchgrid = self.batchgrid.cuda()
            grad_input1 = grad_input1.cuda()
            #print('gradout:',grad_output.size())
        grad_input1 = torch.baddbmm(grad_input1, torch.transpose(grad_output.view(-1, self.height*self.width, 2), 1,2), self.batchgrid.view(-1, self.height*self.width, 3))

        #print(grad_input1)
        return grad_input1
