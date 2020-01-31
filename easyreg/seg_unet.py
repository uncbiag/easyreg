from .modules import Seg_resid
from .utils import *
import torch.nn as nn
from data_pre.partition import partition

class SegUnet(nn.Module):
    def __init__(self,  opt=None):
        super(SegUnet, self).__init__()
        self.opt = opt
        seg_opt = opt['tsk_set'][('seg',{},"settings for seg task")]
        num_class = seg_opt['class_num',-1,"the num of class"]
        use_bn = seg_opt["use_bn", True, "use the batch normalization"]
        patch_sz = opt['dataset']['seg']['patch_size',[-1,-1,-1],"the size of input patch"]
        patch_sz_itk = list(np.flipud(np.array(patch_sz)))
        self.img_sz = None
        self.unet = Seg_resid(num_class,bn=use_bn)
        self.print_count = 0
        self.partition = partition(opt['dataset']['seg']['partition'],patch_sz_itk)

    def set_loss_fn(self, loss_fn):
        """ set loss function"""
        self.loss_fn = loss_fn

    def get_loss(self, output, gt):
        loss = self.loss_fn.get_loss(output,gt)
        return loss

    def check_if_update_lr(self):
        return False, None

    def set_img_sz(self, img_sz):
        self.img_sz = img_sz






    def forward(self, input, is_train=True):
        if is_train:
            output = self.unet(input)
        else:
            with torch.no_grad():
                output = self.get_assamble_pred(input)
        self.print_count += 1
        return output

    def get_assamble_pred(self, input, split_size=8):
        output = []
        input_split = torch.split(input, split_size)
        for input in input_split:
            res = self.forward(input)
            if isinstance(res, list):
                res = res[-1]
            output.append(res.detach().cpu())
        pred_patched = torch.cat(output, dim=0)
        pred_patched = torch.max(pred_patched.data, 1)[1]
        output_np = self.partition.assemble(pred_patched,image_size=self.img_sz)
        return output_np



