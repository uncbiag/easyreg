from __future__ import absolute_import

import torch
import torch_dct as dct
import SimpleITK as sitk
example_img_len = 224
half_len = example_img_len//2
dim = 3

f_path = '/playpen/zyshen/oasis_data/fix_for_reg_debug_3000_pair_reg_224_oasis3_reg_inter/data/test/img/OAS30006_MR_d0166_brain.nii.gz'
I0 = sitk.ReadImage(f_path)
I0 = sitk.GetArrayFromImage(I0)
sz = [1,1] + list(I0.shape)
I0 = torch.Tensor(I0).cuda()
dct_3d = dct.dct_3d
I0 = I0.view(*sz)
dct_I0 = dct_3d(I0)
dct_I0_clone = torch.zeros_like(dct_I0)

dct_I0_clone[:,:,:half_len,:half_len,:half_len] = dct_I0[:,:,:half_len,:half_len,:half_len]
I0_rec = dct.idct_3d(dct_I0_clone)
I0_rec = sitk.GetImageFromArray(I0_rec.detach().cpu().numpy()[0,0])
f_save_path ='/playpen/zyshen/debugs/dct/OAS30006_MR_d0166_brain_origin.nii.gz'
sitk.WriteImage(I0_rec,f_save_path)




# dct_2d = dct.dct_2d
# vt.plot_2d_img(I0[0,0],'I0')
# dct_I0 = dct_2d(I0)
# dct_I0_copy = dct_I0.clone()
# dct_I0=torch.roll(dct_I0, shifts=(64, 64), dims=(2, 3))
# vt.plot_2d_img(dct_I0[0,0],'dct_rolled')
# dct_I0=dod.symmetrize_filter_center_at_zero(dct_I0)
# vt.plot_2d_img(dct_I0[0,0],'dct_fliped')
# dct_I0_clone = torch.zeros_like(dct_I0)
# dct_I0_clone[:,:,:64,:64] = dct_I0[:,:,64:,64:]
#
# print("the diff between dct {}".format(torch.abs(dct_I0_copy-dct_I0_clone).sum()))
#
# # dct_I0_t = (dct_I0-dct_I0.min())/(dct_I0.max()-dct_I0.min())
# # dct_I0_t = torch.log(dct_I0_t+1)
# # print(np.histogram(dct_I0.numpy()))
# # print("the min mean and max of the dct transform result is {},{},{}".format(dct_I0.min(),dct_I0.mean(),dct_I0.max()))
# # vt.plot_2d_img(dct_I0_t[0,0],'dct_I0')
#
#
#
# I0_rec = dct.idct_2d(dct_I0_clone)
# vt.plot_2d_img(I0_rec[0,0],'I0_rec')
