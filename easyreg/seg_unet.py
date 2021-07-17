from .modules import Seg_resid
from .utils import *
import torch.nn as nn
from data_pre.partition import partition

class SegUnet(nn.Module):
    def __init__(self,  opt=None):
        super(SegUnet, self).__init__()
        self.opt = opt
        seg_opt = opt['tsk_set'][('seg',{},"settings for seg task")]
        self.is_train = opt['tsk_set']["train"]
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
        self.ensemble_during_the_test = opt['tsk_set']['seg'][("ensemble_during_the_test",False,"do test phase ensemble, which needs the test phase data augmentation already done")]

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
                if not self.is_train and self.ensemble_during_the_test:
                    output  = self.get_assemble_ensemble(input)
                else:
                    output = self.get_assemble_pred(input)
        self.print_count += 1
        return output

    def get_assemble_pred(self, input, split_size=6):
        output = []
        input_split = torch.split(input, split_size)
        for input_sub in input_split:
            res = self.forward(input_sub)
            if isinstance(res, list):
                res = res[-1]
            output.append(res.detach().cpu())
        pred_patched = torch.cat(output, dim=0)
        pred_patched = torch.max(pred_patched, 1)[1]
        output_np = self.partition.assemble(pred_patched,image_size=self.img_sz)
        return output_np


    def set_file_path(self, file_path, fname):
        self.file_path =file_path
        self.fname = fname


    def get_assemble_pred_for_ensemble(self, input, split_size=6):
        output = []
        input_split = torch.split(input, split_size)
        for input_sub in input_split:
            res = self.forward(input_sub)
            if isinstance(res, list):
                res = res[-1]
            output.append(res.detach().cpu())
        pred_patched = torch.cat(output, dim=0)

        return pred_patched


    def get_assemble_ensemble(self, input):
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

        def compute_warped_image_label(input, warped_pth, warped_type,inv_phi_pth,inv_switcher,num_max=50,weight_for_orig_img=0,nomralize_type="same_as_refer"):
            warped_pth_list = glob(os.path.join(warped_pth, warped_type))
            num_max = min(len(warped_pth_list),num_max)
            inv_phi_pth_list = [pth.replace(warped_pth,inv_phi_pth).replace(*inv_switcher) for pth in warped_pth_list]
            f = lambda pth: sitk.GetArrayFromImage(sitk.ReadImage(pth))
            fname = get_file_name(self.fname[0])
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
            warped_img = self.normalize_input(warped_img,nomralize_type,None)#self.file_path[0][0])
            warped_img = torch.Tensor(warped_img)
            inv_phi = np.stack(inv_phi_list, 0)
            inv_phi = np.transpose(inv_phi, (0, 4, 3, 2, 1))
            inv_phi = torch.Tensor(inv_phi)
            img_input_sz = self.opt["dataset"]["img_after_resize"]
            differ_sz =  any(np.array(warped_img.shape[2:]) != np.array(img_input_sz))



            sz = np.array(self.img_sz)
            spacing = 1. / (sz - 1)
            output_np = np.zeros([1, self.num_class] + self.img_sz)
            if weight_for_orig_img!=0:
                tzero_img = self.get_assemble_pred_for_ensemble(input)
                tzero_pred = self.partition.assemble_multi_torch(tzero_img, image_size=self.img_sz)
                output_np = tzero_pred.cpu().numpy() * float(round(weight_for_orig_img*num_aug))

            for i in range(num_aug):
                if differ_sz:
                    warped_img_cur, _ = resample_image(warped_img[i:i+1].cuda(), [1, 1, 1], [1, 3] + self.img_sz)
                    inv_phi_cur, _ = resample_image(inv_phi[i:i+1].cuda(), [1, 1, 1], [1, 1] + self.img_sz)
                    warped_img_cur = warped_img_cur.detach().cpu()
                    inv_phi_cur = inv_phi_cur.detach().cpu()
                else:
                    warped_img_cur = warped_img[i:i+1]
                    inv_phi_cur = inv_phi[i:i+1]
                sample = {"image":[warped_img_cur[0,0].numpy()]}
                sample_p =corr_partition_pool(sample)
                pred_patched = self.get_assemble_pred_for_ensemble(torch.Tensor(sample_p["image"]).cuda())
                pred_patched = self.partition.assemble_multi_torch(pred_patched, image_size=self.img_sz)
                pred_patched = torch.nn.functional.softmax(pred_patched,1)
                pred_patched = compute_warped_image_multiNC(pred_patched.cuda(), inv_phi_cur.cuda(),spacing, spline_order=1, zero_boundary=True)
                output_np += pred_patched.cpu().numpy()
            res = torch.max(torch.Tensor(output_np), 1)[1]
            return res[None]
        seg_ensemble_opt = self.opt['tsk_set']['seg'][("seg_ensemble",{},"settings of test phase data ensemble")]
        warped_pth = seg_ensemble_opt[("warped_pth", None,"the folder path containing the warped image from the original image")]
        warped_type = seg_ensemble_opt[("warped_type","*_warped.nii.gz","the suffix of the augmented data")]
        inv_phi_pth = seg_ensemble_opt[("inv_phi_pth",None,"the folder path containing the inverse transformation")]
        inv_switcher = seg_ensemble_opt[("inv_switcher",["_warped.nii.gz","_inv_phi.nii.gz"],"the fname switcher from warped image to inverse transformation map")]
        num_max =  seg_ensemble_opt[("num_max",20,"max num of augmentation for per test image")]
        weight_for_orig_img = seg_ensemble_opt[("weight_for_orig_img",0.0,"the weight of original image")]
        nomralize_type = seg_ensemble_opt[("nomralize_type","same_as_refer","'same_as_refer'/'same_as_loader")]

        output_np = compute_warped_image_label(input, warped_pth, warped_type,inv_phi_pth,inv_switcher,num_max=num_max,weight_for_orig_img=weight_for_orig_img, nomralize_type=nomralize_type)
        return output_np




    def normalize_input(self,img,nomralize_type,refer_img_path):
        if nomralize_type == "same_as_refer":
            return self.normalize_from_reference(img,refer_img_path)
        if nomralize_type == "same_as_loader":
            return self.normalize_as_dataloader(img)
        else:
            raise ValueError(
                "the testing phase augmentation normalize type should be either 'same_as_refer'/'same_as_loader'")


    def normalize_from_reference(self,img,refer_img_path):
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

    def normalize_as_dataloader(self, img):
        """
        a numpy image, normalize into intensity [-1,1]
        (img-img.min())/(img.max() - img.min())
        :param img: image
        :param percen_clip:  Linearly normalized image intensities so that the 95-th percentile gets mapped to 0.95; 0 stays 0
        :param range_clip:  Linearly normalized image intensities from (range_clip[0], range_clip[1]) to 0,1
        :return
        """
        dataset_opt = self.opt["dataset"]
        normalize_via_percentage_clip = dataset_opt[('normalize_via_percentage_clip',-1,"normalize the image via percentage clip, the given value is in [0-1]")]
        normalize_via_range_clip = dataset_opt[
            ('normalize_via_range_clip', (-1, -1), "normalize the image via range clip")]
        if normalize_via_percentage_clip>0:
            img = img - img.min()
            normalized_img = img / np.percentile(img, 95) * 0.95
        else:
            range_clip = normalize_via_range_clip
            if range_clip[0]<range_clip[1]:
                img = np.clip(img,a_min=range_clip[0], a_max=range_clip[1])
            min_intensity = img.min()
            max_intensity = img.max()
            normalized_img = (img - img.min()) / (max_intensity - min_intensity)
        normalized_img = normalized_img * 2 - 1
        return normalized_img



