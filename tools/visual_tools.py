
import matplotlib.pyplot as plt
import numpy as np
from model_pool import utils
import SimpleITK as sitk
import torch
import numpy as np
import mermaid.pyreg.finite_differences as fdt
import mermaid.pyreg.utils as py_utils
import os

def save_jacobi_map(map,img_sz,fname,output_path,save_neg_jacobi=True):
    img_sz = np.array(img_sz)
    map_sz = np.array(map.shape[2:])
    spacing = 1. / (np.array(img_sz) - 1)  # the disp coorindate is [-1,1]

    need_resampling = not all(list(img_sz==map_sz))
    if need_resampling:
        id = py_utils.identity_map_multiN(img_sz, spacing)
        map = py_utils.compute_warped_image_multiNC(map, id, spacing, 1,
                                           zero_boundary=False)
    map = map.detach().cpu().numpy()

    fd = fdt.FD_np(spacing)
    dfx = fd.dXc(map[:, 0, ...])
    dfy = fd.dYc(map[:, 1, ...])
    dfz = fd.dZc(map[:, 2, ...])
    jacobi_det = dfx * dfy * dfz
    # self.temp_save_Jacobi_image(jacobi_det,map)
    jacobi_neg_bool = jacobi_det < 0.
    jacobi_neg = jacobi_det[jacobi_neg_bool]
    jacobi_abs = np.abs(jacobi_det)
    jacobi_abs_scalar = - np.sum(jacobi_neg)  #
    jacobi_num_scalar = np.sum(jacobi_neg_bool)
    print("fname:{}  folds for each channel {},{},{}".format(fname,np.sum(dfx < 0.), np.sum(dfy < 0.), np.sum(dfz < 0.)))
    print("fname:{} the jacobi_value of fold points  is {}".format(fname,jacobi_abs_scalar))
    print("fname:{} the number of fold points is {}".format(fname, jacobi_num_scalar))
    for i in range(jacobi_abs.shape[0]):
        if not save_neg_jacobi:
            jacobi_img = sitk.GetImageFromArray(jacobi_abs[i])
        else:
            jacobi_img = sitk.GetImageFromArray(jacobi_neg[i])
        pth = os.path.join(output_path,fname)+'.nii.gz'
        sitk.WriteImage(jacobi_img, pth)



def save_smoother_map(adaptive_smoother_map,gaussian_stds,t,path,dim=3):

    adaptive_smoother_map = adaptive_smoother_map.detach()
    gaussian_stds = gaussian_stds.detach()
    view_sz = [1] + [len(gaussian_stds)] + [1] * dim
    gaussian_stds = gaussian_stds.view(*view_sz)
    smoother_map = adaptive_smoother_map*(gaussian_stds**2)
    smoother_map = torch.sqrt(torch.sum(smoother_map,1,keepdim=True))
    print(t)
    fname = str(t)+"sm_map"
    plot_2d_img(smoother_map[0,0,:,10,:],fname,path)





def plot_2d_img(img,name,path,show=False):

    sp=111

    font = {'size': 10}

    plt.setp(plt.gcf(), 'facecolor', 'white')
    plt.style.use('bmh')

    plt.subplot(sp).set_axis_off()
    plt.imshow(utils.t2np(img))
    plt.colorbar().ax.tick_params(labelsize=3)
    plt.title(name, font)
    if show:
        plt.show()
    else:
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path,name)+'.png', dpi=300)
        plt.clf()







def test():
    import torch
    img = torch.randn(80,80)
    fname = 'test_program'
    output_path = '/playpen/zyshen/debugs/plot_2d'
    plot_2d_img(img,fname,output_path,show=False)