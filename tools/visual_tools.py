
import matplotlib.pyplot as plt
from easyreg import utils
import SimpleITK as sitk
import torch
import numpy as np
import mermaid.finite_differences as fdt
import mermaid.utils as py_utils
import os
from scipy import misc

def read_png_into_numpy(file_path,name=None,visual=False):
    image = misc.imread(file_path,flatten=True)
    image = (image-image.min())/(image.max()-image.min())
    if visual:
        plot_2d_img(image,name if name is not None else'image')
    return image

def read_png_into_standard_form(file_path,name=None,visual=False):
    image = read_png_into_numpy(file_path,name,visual)
    sz  =[1,1]+list(image.shape)
    image = image.reshape(*sz)
    spacing = 1. / (np.array(sz[2:]) - 1)
    return image,spacing


def save_3D_img_from_numpy(input,file_path,spacing=None,orgin=None,direction=None):
    output = sitk.GetImageFromArray(input)
    if spacing is not None:
        output.SetSpacing(spacing)
    if orgin is not None:
        output.SetOrigin(orgin)
    if direction is not None:
        output.SetDirection(direction)
    os.makedirs(os.path.split(file_path)[0], exist_ok=True)
    sitk.WriteImage(output, file_path)

def save_3D_img_from_itk(output,file_path,spacing=None,orgin=None,direction=None):
    if spacing is not None:
        output.SetSpacing(spacing)
    if orgin is not None:
        output.SetOrigin(orgin)
    if direction is not None:
        output.SetDirection(direction)
    os.makedirs(os.path.split(file_path)[0], exist_ok=True)
    sitk.WriteImage(output, file_path)




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



def save_smoother_map(adaptive_smoother_map,gaussian_stds,t,path=None,weighting_type=None):
    dim = len(adaptive_smoother_map.shape)-2
    adaptive_smoother_map = adaptive_smoother_map.detach()
    if weighting_type=='w_K_w':
        adaptive_smoother_map = adaptive_smoother_map**2
    gaussian_stds = gaussian_stds.detach()
    view_sz = [1] + [len(gaussian_stds)] + [1] * dim
    gaussian_stds = gaussian_stds.view(*view_sz)
    smoother_map = adaptive_smoother_map*(gaussian_stds**2)
    smoother_map = torch.sqrt(torch.sum(smoother_map,1,keepdim=True))
    print(t)
    fname = str(t)+"sm_map"
    if dim ==2:
        plot_2d_img(smoother_map[0,0],fname,path)
    elif dim==3:
        y_half = smoother_map.shape[3]//2
        plot_2d_img(smoother_map[0,0,:,y_half,:],fname,path)


def save_momentum(momentum,t=None,path=None):
    dim = len(momentum.shape)-2
    momentum = momentum.detach()
    momentum = torch.sum(momentum**2,1,keepdim=True)
    if t is not None:
        print(t)
        fname = str(t)+"momentum"
    else:
        fname = "momentum"
    if dim ==2:
        plot_2d_img(momentum[0,0],fname,path)
    elif dim==3:
        y_half = momentum.shape[3]//2
        plot_2d_img(momentum[0,0,:,y_half,:],fname,path)

def save_velocity(velocity,t,path=None):
    dim = len(velocity.shape)-2
    velocity = velocity.detach()
    velocity = torch.sum(velocity**2,1,keepdim=True)
    print(t)
    fname = str(t)+"velocity"
    if dim ==2:
        plot_2d_img(velocity[0,0],fname,path)
    elif dim==3:
        y_half = velocity.shape[3]//2
        plot_2d_img(velocity[0,0,:,y_half,:],fname,path)


def plot_2d_img(img,name,path=None):
    """
    :param img:  X x Y x Z
    :param name: title
    :param path: saving path
    :param show:
    :return:
    """
    sp=111
    img = torch.squeeze(img)

    font = {'size': 10}

    plt.setp(plt.gcf(), 'facecolor', 'white')
    plt.style.use('bmh')

    plt.subplot(sp).set_axis_off()
    plt.imshow(utils.t2np(img))#,vmin=0.0590, vmax=0.0604) #vmin=0.0590, vmax=0.0604
    plt.colorbar().ax.tick_params(labelsize=10)
    plt.title(name, font)
    if not path:
        plt.show()
    else:
        plt.savefig(path, dpi=300)
        plt.clf()




def visualize_jacobi(phi,spacing, img=None, file_path=None, visual=True):
    """
    :param phi:  Bxdimx X xYxZ
    :param spacing: [sx,sy,sz]
    :param img: Bx1xXxYxZ
    :param file_path: saving path
    :return:
    """
    phi_sz = phi.shape
    n_batch = phi_sz[0]
    dim =phi_sz[1]
    phi_np = utils.t2np(phi)
    if img is not None:
        assert phi.shape[0] == img.shape[0]
        img_np = utils.t2np(img)
    fd = fdt.FD_np(spacing)
    dfx = fd.dXc(phi_np[:, 0, ...])
    dfy = fd.dYc(phi_np[:, 1, ...])
    dfz =1.
    if dim==3:
        dfz = fd.dZc(phi_np[:, 2, ...])
    jacobi_det = dfx * dfy * dfz
    jacobi_neg = np.ma.masked_where(jacobi_det>= 0, jacobi_det)
    #jacobi_neg = (jacobi_det<0).astype(np.float32)
    jacobi_abs = - np.sum(jacobi_det[jacobi_det < 0.])  #
    jacobi_num = np.sum(jacobi_det < 0.)
    if dim==3:
        print("print folds for each channel {},{},{}".format(np.sum(dfx < 0.), np.sum(dfy < 0.), np.sum(dfz < 0.)))
    print("the jacobi_value of fold points for current map is {}".format(jacobi_abs))
    print("the number of fold points for current map is {}".format(jacobi_num))

    if visual:
        for i in range(n_batch):
            if dim == 2:
                sp = 111
                font = {'size': 10}
                plt.setp(plt.gcf(), 'facecolor', 'white')
                plt.style.use('bmh')
                plt.subplot(sp).set_axis_off()
                plt.imshow(utils.t2np(img_np[i,0]))
                plt.imshow(jacobi_neg[i], cmap='gray', alpha=1.)
                plt.colorbar().ax.tick_params(labelsize=10)
                plt.title('img_jaocbi', font)
                if not file_path:
                    plt.show()
                else:
                    plt.savefig(file_path, dpi=300)
                    plt.clf()
            if dim ==3:
                if file_path:
                    jacobi_abs_map = np.abs(jacobi_det)
                    jacobi_img = sitk.GetImageFromArray(jacobi_abs_map[i])
                    pth = os.path.join(file_path)
                    sitk.WriteImage(jacobi_img, pth)









def test():
    import torch
    img = torch.randn(80,80)
    fname = 'test_program'
    output_path = '/playpen/zyshen/debugs/plot_2d'
    plot_2d_img(img,fname,output_path,show=False)
