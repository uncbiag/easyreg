from .modules import Seg_resid
from .utils import *
import torch.nn as nn

class SegUnet(nn.Module):
    def __init__(self, img_sz=None, opt=None):
        super(SegUnet, self).__init__()
        self.img_sz= img_sz
        self.opt = opt
        self.is_train = opt['tsk_set'][('train',False,'if is in train mode')]
        seg_opt = opt['tsk_set'][('seg',{},"settings for seg task")]
        num_class = seg_opt['num_class',-1,"the num of class"]
        use_bn = seg_opt["use_bn", True, "use the batch normalization"]
        self.unet = Seg_resid(num_class,bn=use_bn)

    def set_loss_fn(self, loss_fn):
        """ set loss function"""
        self.loss_fn = loss_fn

    def get_loss(self, output, gt):
        loss = self.loss_fn(output,gt)
        if self.print_count % 10 == 0:
            print('current loss is {} '.format(loss))

    def check_if_update_lr(self):
        return False, None




    def forward(self, input):
        if self.is_train:
            output = self.unet(input)
        else:
            pass
        return output