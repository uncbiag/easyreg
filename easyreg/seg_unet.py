from .modules import Seg_resid
from .utils import *
import torch.nn as nn

class SegUnet(nn.Module):
    def __init__(self, img_sz=None, opt=None):
        super(SegUnet, self).__init__()
        self.img_sz= img_sz
        self.opt = opt
        seg_opt = opt[('seg',{},"settings for seg task")]
        num_class = seg_opt['num_class',-1,"the num of class"]
        use_bn = seg_opt["use_bn", True, "use the batch normalization"]
        self.unet = Seg_resid(num_class,bn=use_bn)



    def get_loss(self):
        """
        get the overall loss

        :return:
        """
        return self.overall_loss

    def forward(self, input):
        pass