from __future__ import absolute_import

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch_dct as dct
from tools import visual_tools as vt
import mermaid.example_generation as eg
from toremove import depend_on_dct as dod

example_img_len = 64
dim = 2
I0, I1,spacing = eg.CreateRealExampleImages(dim).create_image_pair()  # create a default image size with two sample squares
I0 = torch.Tensor(I0)
I1 = torch.Tensor(I1)
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



run_train = True
if run_train:

    model = dod.UNet(n_class = 1).cuda()

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)

    moving = I0.cuda()
    target = I1.cuda()
    iter = 0
    while iter<6000:
        iter +=1
        scheduler.step()
        optimizer.zero_grad()
        dct_m = dct.dct_2d(moving,'ortho')
        dct_m = torch.roll(dct_m, shifts=(64, 64), dims=(2, 3))
        dct_m = dod.symmetrize_filter_center_at_zero(dct_m)
        recons = model(dct_m)

        rec_factors = recons
        mv_factors = torch.zeros_like(moving)
        mv_factors[:,:,:64,:64] = recons[:,:,64:,64:]
        recons = dct.idct_2d(mv_factors)
        moving = target

        loss = dod.cal_loss(recons,moving)
        loss.backward()

        if iter%20 ==0:
            print("iter {} the reconstr loss is {}".format(iter,loss.item()))
        optimizer.step()
        if iter%200 ==0:
            vt.plot_2d_img(recons[0,0], 'recons_'+str(iter))



    I1 = torch.Tensor(I1).cuda()
    moving = I1
    dct_m = dct.dct_2d(moving)
    dct_m = torch.roll(dct_m, shifts=(64, 64), dims=(2, 3))
    dct_m = dod.symmetrize_filter_center_at_zero(dct_m)
    recons = model(dct_m)
    loss = dod.cal_loss(recons,moving)
    print("iter {} the reconstr loss is {}".format('test', loss.item()))
    vt.plot_2d_img(recons[0, 0], 'recons_' + 'test')





