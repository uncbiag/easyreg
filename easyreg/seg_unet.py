from .modules import Seg_resid
from .utils import *
import torch.nn as nn
from data_pre.partition import partition

class SegUnet(nn.Module):
    def __init__(self,  opt=None):
        super(SegUnet, self).__init__()
        self.opt = opt
        seg_opt = opt['tsk_set'][('seg',{},"settings for seg task")]
        self.num_class = seg_opt['class_num',-1,"the num of class"]
        use_bn = seg_opt["use_bn", True, "use the batch normalization"]
        patch_sz = opt['dataset']['seg']['patch_size',[-1,-1,-1],"the size of input patch"]
        overlap_sz = opt['dataset']['seg']['partition']['overlap_size',[-1,-1,-1],"the size of input patch"]
        patch_sz_itk = list(np.flipud(np.array(patch_sz)))
        overlap_sz_itk = list(np.flipud(np.array(overlap_sz)))
        self.img_sz = None
        self.unet = Seg_resid(self.num_class,bn=use_bn)
        self.print_count = 0
        self.partition = partition(opt['dataset']['seg']['partition'],patch_sz_itk,overlap_sz_itk)

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
                output = self.get_assemble_pred(input)
        self.print_count += 1
        return output

    def get_assemble_pred(self, input, split_size=8):
        output = []
        input_split = torch.split(input, split_size)
        for input_sub in input_split:
            res = self.forward(input_sub)
            if isinstance(res, list):
                res = res[-1]
            output.append(res.detach().cpu())
        pred_patched = torch.cat(output, dim=0)
        pred_patched = torch.max(pred_patched.data, 1)[1]
        output_np = self.partition.assemble(pred_patched,image_size=self.img_sz)
        return output_np


    def set_file_path(self, file_path):
        self.file_path =file_path


    def get_assemble_pred_tmp(self, input, split_size=8):
        output = []
        input_split = torch.split(input, split_size)
        for input_sub in input_split:
            res = self.forward(input_sub)
            if isinstance(res, list):
                res = res[-1]
            output.append(res.detach().cpu())
        pred_patched = torch.cat(output, dim=0)

        return pred_patched


    def get_assemble_inv(self, input):
        import os
        from .reg_data_utils import read_txt_into_list, get_file_name
        from tools.image_rescale import save_image_with_given_reference
        import SimpleITK as sitk
        import torch
        import numpy as np
        from glob import glob
        from copy import deepcopy
        from mermaid.utils import compute_warped_image_multiNC
        patch_sz = self.opt['dataset']['seg']['patch_size', [-1, -1, -1], "the size of input patch"]
        overlap_sz = self.opt['dataset']['seg']['partition']['overlap_size', [-1, -1, -1], "the size of input patch"]
        option_p = self.opt['dataset']['seg'][('partition', {}, "settings for the partition")]
        patch_sz_itk = list(np.flipud(np.array(patch_sz)))
        overlap_sz_itk = list(np.flipud(np.array(overlap_sz)))
        corr_partition_pool = deepcopy(partition(option_p, patch_sz_itk, overlap_sz_itk))

        def compute_warped_image_label(input, warped_pth, warped_type,inv_phi_pth,inv_switcher,num_max=50,t_zero_ratio=0):
            warped_pth_list = glob(os.path.join(warped_pth, warped_type))
            num_max = min(len(warped_pth_list),num_max)
            inv_phi_pth_list = [pth.replace(warped_pth,inv_phi_pth).replace(*inv_switcher) for pth in warped_pth_list]
            f = lambda pth: sitk.GetArrayFromImage(sitk.ReadImage(pth))
            fname = get_file_name(self.file_path[0][0])
            f_warped = lambda x: get_file_name(x).find(fname+'_') == 0
            warped_sub_list = list(filter(f_warped, warped_pth_list))
            inv_phi_sub_list = list(filter(f_warped, inv_phi_pth_list))
            warped_sub_list = warped_sub_list[:num_max]
            inv_phi_sub_list = inv_phi_sub_list[:num_max]
            num_aug = len(warped_sub_list)
            warped_list = [f(pth) for pth in warped_sub_list]
            inv_phi_list = [f(pth) for pth in inv_phi_sub_list]
            warped_img = np.stack(warped_list, 0)[:,None]
            #warped_img = torch.Tensor(warped_img)*2-1.
            warped_img = self.normalize_input(warped_img,None)#self.file_path[0][0])
            warped_img = torch.Tensor(warped_img)
            inv_phi = np.stack(inv_phi_list, 0)
            inv_phi = np.transpose(inv_phi, (0, 4, 3, 2, 1))
            inv_phi = torch.Tensor(inv_phi)

            sz = np.array(warped_img.shape[2:])
            spacing = 1. / (sz - 1)
            output_np = np.zeros([1, self.num_class] + self.img_sz)
            if t_zero_ratio!=0:
                tzero_img = self.get_assemble_pred_tmp(input)
                tzero_pred = self.partition.assemble_multi_torch(tzero_img, image_size=self.img_sz)
                output_np = tzero_pred.cpu().numpy() * float(round(t_zero_ratio*num_aug))

            for i in range(num_aug):
                sample = {"image":[warped_img[i,0].numpy()]}
                sample_p =corr_partition_pool(sample)
                pred_patched = self.get_assemble_pred_tmp(torch.Tensor(sample_p["image"]).cuda())
                pred_patched = self.partition.assemble_multi_torch(pred_patched, image_size=self.img_sz)
                pred_patched = torch.nn.functional.softmax(pred_patched,1)
                pred_patched = compute_warped_image_multiNC(pred_patched.cuda(), inv_phi[i:i+1].cuda(),spacing, spline_order=1, zero_boundary=True)
                output_np += pred_patched.cpu().numpy()
            res = torch.max(torch.Tensor(output_np), 1)[1]
            return res[None]

        # phi_pth = "/playpen-raid/zyshen/data/lpba_reg/test_aug/reg/res/records"
        # inv_phi_pth = "/playpen-raid/zyshen/data/lpba_reg/test_aug/reg/res/records/original_sz"

        # warped_pth = "/playpen-raid/zyshen/data/lpba_reg/test_aug_new/reg/res/records/original_sz"
        # inv_phi_pth = "/playpen-raid/zyshen/data/lpba_reg/test_aug_new/reg/res/records/original_sz"
        # warped_type = '*_warped.nii.gz'
        # inv_switcher = ('_warped.nii.gz','_inv_phi.nii.gz')

        # warped_pth = "/playpen-raid/zyshen/data/oai_reg/test_aug_opt/reg/res/records/original_sz"
        # inv_phi_pth = "/playpen-raid/zyshen/data/oai_reg/test_aug_opt/reg/res/records/original_sz"

        warped_pth = self.opt['tsk_set']["debug"]["warped_pth"]
        inv_phi_pth = self.opt['tsk_set']["debug"]["inv_phi_pth"]
        warped_type = self.opt['tsk_set']["debug"]["warped_type"]
        inv_switcher = self.opt['tsk_set']["debug"]["inv_switcher"]
        num_max =  self.opt['tsk_set']["debug"]["num_max"]
        t_zero_ratio = self.opt['tsk_set']["debug"][("t_zero_ratio",0.0,"the weight of main image")]

        # warped_pth = "/playpen-raid1/zyshen/data/oai_reg/test_aug_1500/"
        # inv_phi_pth = "/playpen-raid1/zyshen/data/oai_reg/test_aug_1500/"
        # warped_type = '*_image.nii.gz'
        # inv_switcher = ('_image.nii.gz','_inv_map.nii.gz')
        output_np = compute_warped_image_label(input, warped_pth, warped_type,inv_phi_pth,inv_switcher,num_max=num_max,t_zero_ratio=t_zero_ratio)
        return output_np




    #
    # def get_assemble_inv(self, input):
    #     import os
    #     from .reg_data_utils import read_txt_into_list, get_file_name
    #     from tools.image_rescale import save_image_with_given_reference
    #     import SimpleITK as sitk
    #     import torch
    #     import numpy as np
    #     from glob import glob
    #     from copy import deepcopy
    #     from mermaid.utils import compute_warped_image_multiNC
    #     patch_sz = self.opt['dataset']['seg']['patch_size', [-1, -1, -1], "the size of input patch"]
    #     overlap_sz = self.opt['dataset']['seg']['partition']['overlap_size', [-1, -1, -1], "the size of input patch"]
    #     option_p = self.opt['dataset']['seg'][('partition', {}, "settings for the partition")]
    #     patch_sz_itk = list(np.flipud(np.array(patch_sz)))
    #     overlap_sz_itk = list(np.flipud(np.array(overlap_sz)))
    #     corr_partition_pool = deepcopy(partition(option_p, patch_sz_itk, overlap_sz_itk))
    #
    #     def compute_warped_image_label(input, phi_pth, phi_type,inv_phi_pth,inv_switcher,num_max=50):
    #         phi_pth_list = glob(os.path.join(phi_pth, phi_type))
    #         inv_phi_pth_list = [pth.replace(phi_pth,inv_phi_pth).replace(*inv_switcher) for pth in phi_pth_list]
    #         num_max = min(len(phi_pth_list), num_max)
    #         f = lambda pth: sitk.GetArrayFromImage(sitk.ReadImage(pth))
    #         fname = get_file_name(self.file_path[0][0])
    #         img = input.detach().cpu()
    #         f_phi = lambda x: get_file_name(x).find(fname) == 0
    #         phi_sub_list = list(filter(f_phi, phi_pth_list))
    #         inv_phi_sub_list = list(filter(f_phi, inv_phi_pth_list))
    #         phi_sub_list = phi_sub_list[:num_max]
    #         inv_phi_sub_list = inv_phi_sub_list[:num_max]
    #         num_aug = len(phi_sub_list)
    #         phi_list = [f(pth) for pth in phi_sub_list]
    #         inv_phi_list = [f(pth) for pth in inv_phi_sub_list]
    #         phi = np.stack(phi_list, 0)
    #         phi = np.transpose(phi, (0, 4, 3, 2, 1))
    #         phi = torch.Tensor(phi)
    #         inv_phi = np.stack(inv_phi_list, 0)
    #         inv_phi = np.transpose(inv_phi, (0, 4, 3, 2, 1))
    #         inv_phi = torch.Tensor(inv_phi)
    #
    #         img = self.partition.assemble(img[:,0], image_size=self.img_sz)
    #         img = torch.Tensor(img)
    #         img = img.repeat(num_aug, 1, 1, 1, 1)
    #
    #         sz = np.array(img.shape[2:])
    #         spacing = 1. / (sz - 1)
    #         warped_img = compute_warped_image_multiNC(torch.Tensor(img), phi, spacing, spline_order=1, zero_boundary=True)
    #         output_np = np.zeros([1,self.num_class]+self.img_sz)
    #         for i in range(num_aug):
    #             sample = {"image":[warped_img[i,0].numpy()]}
    #             sample_p =corr_partition_pool(sample)
    #             pred_patched = self.get_assemble_pred_tmp(torch.Tensor(sample_p["image"]).cuda())
    #             pred_patched = self.partition.assemble_multi_torch(pred_patched, image_size=self.img_sz)
    #             pred_patched = compute_warped_image_multiNC(pred_patched.cuda(), inv_phi[i:i+1].cuda(),spacing, spline_order=1, zero_boundary=True)
    #             output_np += pred_patched.cpu().numpy()
    #         res = torch.max(torch.Tensor(output_np), 1)[1]
    #         return res[None]
    #
    #     # phi_pth = "/playpen-raid/zyshen/data/lpba_reg/test_aug/reg/res/records"
    #     # inv_phi_pth = "/playpen-raid/zyshen/data/lpba_reg/test_aug/reg/res/records/original_sz"
    #     # phi_type = '*_phi.nii.gz'
    #     # inv_switcher = ('_phi.nii.gz', '_inv_phi.nii.gz')
    #
    #     phi_pth = "/playpen-raid1/zyshen/data/lpba_reg/test_aug_300_range2/"
    #     inv_phi_pth = "/playpen-raid1/zyshen/data/lpba_reg/test_aug_300_range2/"
    #     phi_type = '*_phi_map.nii.gz'
    #     inv_switcher = ('_phi_map.nii.gz', '_inv_map.nii.gz')
    #
    #
    #     output_np = compute_warped_image_label(input, phi_pth, phi_type,inv_phi_pth,inv_switcher,num_max=20)
    #     return output_np









    def normalize_input(self,img,refer_img_path):
        import SimpleITK as sitk
        if refer_img_path is not None:
            refer_img =  sitk.GetArrayFromImage(sitk.ReadImage(refer_img_path))
        else:
            refer_img = img
        min_intensity = refer_img.min()
        max_intensity = refer_img.max()
        normalized_img = (img - refer_img.min()) / (max_intensity - min_intensity)
        normalized_img = normalized_img * 2 - 1
        return normalized_img



