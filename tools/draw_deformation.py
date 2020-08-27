import numpy as np
import sys,os
os.environ["CUDA_VISIBLE_DEVICES"] = ''
from easyreg.viewers import *
from mermaid.utils import *
from mermaid.data_utils import *
import SimpleITK as sitk
from glob import glob
import os



sz  = [160,200,200]
def get_image_list_to_draw(refer_folder,momentum_folder,img_type,source_target_folder,t_list):
    """
    we first need to get a dict, where we can get the {"pair_name": "pair":[source,target], "fluid_warped":[warped],"phi"[phi for t=1],"t":[],"linear_warped"[]}
    :param refer_folder:
    :param img_type:
    :param source_txt:
    :return:
    """
    pair_path_list = glob(os.path.join(refer_folder,"*"+img_type))
    #pair_path_list = glob(os.path.join(refer_folder,"*9069761_image_9074437_image_9069761_image_9397988_image_0d0000_1d0000_t_1d00_image.nii.gz"))
    pair_name_list = [get_file_name(path).replace(img_type.split(".")[0],"") for path in pair_path_list]
    source_name_list = [name.split("_")[0]+"_image" for name in pair_name_list]
    target_name_list = [name.split("_")[6]+"_image" for name in pair_name_list]
    momentum_list = [source_name+'_'+target_name+"_0000Momentum.nii.gz" for source_name, target_name in zip(source_name_list,target_name_list)]
    momentum_list = [os.path.join(momentum_folder,fname) for fname in momentum_list]
    source_path_list = [os.path.join(source_target_folder,source_name+'.nii.gz') for source_name in source_name_list]
    target_path_list = [os.path.join(source_target_folder,target_name+'.nii.gz') for target_name in target_name_list]
    lsource_path_list = [path.replace("image.nii.gz","masks.nii.gz") for path in source_path_list]
    ltarget_path_list = [path.replace("image.nii.gz","masks.nii.gz") for path in target_path_list]
    warped_path_list = [[os.path.join(refer_folder,pair_name+"_0d0000_1d0000_t_{}_image.nii.gz".format(str("{:.2f}".format(t)).replace(".","d")))for t in t_list]  for pair_name in pair_name_list ]
    phi_path_list =[[path.replace("_image.nii.gz","_phi_map.nii.gz") for path in paths] for paths in warped_path_list]
    inv_phi_path_list =[[path.replace("_image.nii.gz","_inv_map.nii.gz") for path in paths] for paths in warped_path_list]
    lwarped_path_list = [[warped_path.replace("image.nii.gz","label.nii.gz") for warped_path in pair_warped_path] for pair_warped_path in warped_path_list]
    phi1_path = [path.replace("_image.nii.gz","_phi_map.nii.gz") for path in pair_path_list]
    dict_to_draw = {}
    for i, pair_name in enumerate(pair_name_list):
        dict_to_draw[pair_name] = {"pair_name": pair_name, "pair_path":[source_path_list[i],target_path_list[i],lsource_path_list[i],ltarget_path_list[i]]
            ,"fluid_path":warped_path_list[i], "lfluid_path":lwarped_path_list[i],"phi_path":phi_path_list[i],"phi1":phi1_path[i],"t":t_list,"momentum_path":momentum_list[i],"inv_phi_path":inv_phi_path_list[i]}

    return dict_to_draw





def draw_images(dict_to_draw):
    for pair_name in dict_to_draw:
        try:
            draw_image(dict_to_draw[pair_name])
        except:
            pass

def draw_image(single_image_dict):
    source_path = single_image_dict['pair_path'][0]
    target_path = single_image_dict['pair_path'][1]
    lsource_path =  single_image_dict['pair_path'][2]
    ltarget_path =  single_image_dict['pair_path'][3]
    fluid_path_list = single_image_dict['fluid_path']
    lfluid_path_list = single_image_dict['lfluid_path']
    phi_path_list = single_image_dict['phi_path']
    phi1_path  =single_image_dict["phi1"]
    t_list  =single_image_dict["t"]
    fr_sitk = lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x))
    source =  fr_sitk(source_path)
    lsource =  fr_sitk(lsource_path)
    target =  fr_sitk(target_path)
    ltarget =  fr_sitk(ltarget_path)
    fluid_images = [fr_sitk(path) for path in fluid_path_list]
    lfluid_images = [fr_sitk(path) for path in lfluid_path_list]
    phis = [np.transpose(fr_sitk(path),[3,2,1,0]) for path in phi_path_list]
    phi1 = np.transpose(fr_sitk(phi1_path),[3,2,1,0])
    phi1_tensor = torch.Tensor(phi1[None])
    spacing = 1./(np.array(source.shape)-1)
    identity_map_np = identity_map_multiN([1,1]+sz,spacing)
    identity_map = torch.Tensor(identity_map_np)
    source_tensor = torch.Tensor(source)[None][None]
    lsource_tensor = torch.Tensor(lsource)[None][None]
    if list(phi1_tensor.shape[2:])!=list(source.shape[2:]):
        fres = lambda x:resample_image(x, spacing, [1, 3] + list(lsource_tensor.shape[2:]))
        phi1_tensor, _ = fres(phi1_tensor)
        phis = [fres(torch.Tensor(phi[None]))[0] for phi in phis]
        phis =[phi[0].numpy() for phi in phis]
    disp = phi1_tensor - identity_map
    linear_images = []
    llinear_images = []
    linear_phis = []
    for t in t_list:
        phi = identity_map + disp*t
        linear = compute_warped_image_multiNC(source_tensor,phi,spacing,spline_order=1,zero_boundary=True)
        llinear = compute_warped_image_multiNC(lsource_tensor,phi,spacing,spline_order=0,zero_boundary=True)
        linear_images.append(linear.numpy()[0,0])
        llinear_images.append(llinear.numpy()[0,0])
        linear_phis.append(phi.numpy()[0])

    draw_defomation(fluid_images, phis, linear_images, linear_phis,source,identity_map_np[0])


def draw_defomation(fluid_images,phis,linear_images,linear_phis,source,identity_map):

    fig, ax = plt.subplots(2, 6, figsize=(45, 16))
    # img = np.zeros_like(img)
    # plt.setp(plt.gcf(), 'facecolor', 'white')
    # plt.style.use('grayscale')
    plt.style.use("bmh")

    ivx = ImageViewer3D_Sliced_Contour(ax[0][0], linear_images[0], linear_phis[0], 0, '', showColorbar=False)
    ivy = ImageViewer3D_Sliced_Contour(ax[0][1], linear_images[1], linear_phis[1], 0, '', showColorbar=False)
    #ivz = ImageViewer3D_Sliced_Contour(ax[0][2], source, identity_map, 0, '', showColorbar=False)
    ivz = ImageViewer3D_Sliced_Contour(ax[0][2], linear_images[2], linear_phis[2], 0, '', showColorbar=False)

    ivz = ImageViewer3D_Sliced_Contour(ax[0][3], linear_images[3], linear_phis[3], 0, '', showColorbar=False)
    ivz = ImageViewer3D_Sliced_Contour(ax[0][4], linear_images[4], linear_phis[4], 0, '', showColorbar=False)
    ivz = ImageViewer3D_Sliced_Contour(ax[0][5], linear_images[5], linear_phis[5], 0, '', showColorbar=False)
    #ivz = ImageViewer3D_Sliced_Contour(ax[0][7], linear_images[6], linear_phis[6], 0, '', showColorbar=False)

    ivx = ImageViewer3D_Sliced_Contour(ax[1][0], fluid_images[0], phis[0], 0, '', showColorbar=False)
    ivy = ImageViewer3D_Sliced_Contour(ax[1][1], fluid_images[1], phis[1], 0, '', showColorbar=False)
    ivz = ImageViewer3D_Sliced_Contour(ax[1][2], fluid_images[2], phis[2], 0, '', showColorbar=False)
    #ivz = ImageViewer3D_Sliced_Contour(ax[1][3], source, identity_map, 0, '', showColorbar=False)

    ivz = ImageViewer3D_Sliced_Contour(ax[1][3], fluid_images[3], phis[3], 0, '', showColorbar=False)
    ivz = ImageViewer3D_Sliced_Contour(ax[1][4], fluid_images[4], phis[4], 0, '', showColorbar=False)
    ivz = ImageViewer3D_Sliced_Contour(ax[1][5], fluid_images[5], phis[5], 0, '', showColorbar=False)
    #ivz = ImageViewer3D_Sliced_Contour(ax[1][7], fluid_images[6], phis[6], 0, '', showColorbar=False)
    plt.axis('off')


    plt.clim(vmin=-1., vmax=1.)
    plt.show()



def view_2d_from_3d(img=None, phi=None,fpth=None,color=True):
    fig, ax = plt.subplots(1,1)
    #plt.setp(plt.gcf(), 'facecolor', 'white')
    if not color:
        plt.style.use('grayscale')
    else:
        plt.style.use("bmh")
    ax.set_axis_off()
    if img is None:
        img = np.zeros_like(phi[0])
    ImageViewer3D_Sliced(ax, img, 0, '', False)
    if phi is not None:
        ImageViewer3D_Sliced_Contour(ax, img, phi, 0, '', showColorbar=False)
    if fpth is not None:
        plt.savefig(fpth, dpi=100, bbox_inches='tight')
        plt.close('all')
    else:
        plt.show()
        plt.clf()

#
# img_type = "_0d0000_1d0000_t_1d00_image.nii.gz"
# t_list = [-3,-1,0.5,1,3,4]
# source_target_folder = "/playpen-raid/olut/Nifti_resampled_rescaled_2Left_Affine2atlas"
# #
# refer_folder = "/playpen-raid/zyshen/data/oai_reg/draw4"
# dict_to_draw = get_image_list_to_draw(refer_folder,"",img_type,source_target_folder,t_list)
# draw_images(dict_to_draw)
#


def read_img_phi(img_path_list, phi_path_list=None):
    f = lambda pth: sitk.GetArrayFromImage(sitk.ReadImage(pth))
    img_list = [f(pth) for pth in img_path_list]
    phi_list = None
    if phi_path_list is not None:
        phi_list = [f(pth) for pth in phi_path_list]
        phi_list = [np.transpose(phi, (3, 2, 1,0)) for phi in phi_list]
    return img_list, phi_list


from tools.visual_tools import *
img_type = "_0d0000_1d0000_t_1d00_image.nii.gz"
t_list = [-1,  -0.5, 0.5,1, 1.5, 2.0]
source_target_folder = "/playpen-raid/olut/Nifti_resampled_rescaled_2Left_Affine2atlas"
#/playpen-raid/zyshen/data/oai_reg/train_with_10/momentum_lresol/9397988_image_9074437_image_0000Momentum.nii.gz
momentum_folder ="/playpen-raid/zyshen/data/oai_reg/train_with_10/momentum_lresol"
momentum_ftype = "_0000Momentum.nii.gz"
refer_folder = "/playpen-raid/zyshen/data/oai_reg/draw4"
dict_to_draw = get_image_list_to_draw(refer_folder,momentum_folder,img_type,source_target_folder,t_list)
output_folder = "/playpen-raid1/zyshen/data/oai_reg/draw_output4"
"""
dict_to_draw[pair_name] = {"pair_name": pair_name, "pair_path":[source_path_list[i],target_path_list[i],lsource_path_list[i],ltarget_path_list[i]]
            ,"fluid_path":warped_path_list[i], "lfluid_path":lwarped_path_list[i],"phi_path":phi_path_list[i],"phi1":phi1_path[i],"t":t_list,"momentum_path":momentum_list[i]}
for each pair name, we have source.png, target.png, momentum.png, phi_name.png, warped_name.png, l_warped_name.png """
for pair_name, pair_detail in dict_to_draw.items():
    output_path = os.path.join(output_folder,pair_name)
    os.makedirs(output_path,exist_ok=True)
    source_path = pair_detail["pair_path"][0]
    target_path = pair_detail["pair_path"][1]
    lsource_path = pair_detail["pair_path"][2]
    momentum_path =  pair_detail["momentum_path"]
    phi_path_list = pair_detail["phi_path"]
    inv_phi_path_list = pair_detail["inv_phi_path"]
    warped_path_list =  pair_detail["fluid_path"]
    l_warped_path_list =  pair_detail["lfluid_path"]
    source_save_path = os.path.join(output_path,"source.png")
    lsource_save_path = os.path.join(output_path,"lsource.png")
    target_save_path = os.path.join(output_path,"target.png")
    momentum_save_path = os.path.join(output_path,"momentum.png")
    warped_name_list = [get_file_name(pth) for pth in warped_path_list]
    warped_save_path_list = [os.path.join(output_path,fname) +"_warped.png" for fname in warped_name_list]
    lwarped_save_path_list = [os.path.join(output_path,fname) + "_lwarped.png" for fname in warped_name_list]
    lwarped_phi_save_path_list = [os.path.join(output_path,fname) + "_lwarpedphi.png" for fname in warped_name_list]
    lwarped_invphi_save_path_list = [os.path.join(output_path,fname) + "_lwarpedinvphi.png" for fname in warped_name_list]
    phi_save_path_list = [os.path.join(output_path,fname) + "_phi.png" for fname in warped_name_list]
    inv_phi_save_path_list = [os.path.join(output_path,fname) + "_inv_phi.png" for fname in warped_name_list]
    img_phi_save_path_list = [os.path.join(output_path,fname) + "_imgphi.png" for fname in warped_name_list]
    f = lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x))
    f_v = lambda x: np.transpose(f(x),[3,2,1,0])
    view_2d_from_3d(img=f(source_path),fpth=source_save_path)
    view_2d_from_3d(img=f(target_path),fpth=target_save_path)
    view_2d_from_3d(img=f(lsource_path),fpth=lsource_save_path)
    momentum = f_v(momentum_path)
    momentum = np.sum(momentum ** 2, 1)
    view_2d_from_3d(img=momentum, fpth=momentum_save_path,color=True)
    l = f(lsource_path)
    for i in range(len(warped_name_list)):
        warped = f(warped_path_list[i])
        view_2d_from_3d(img=warped, fpth=warped_save_path_list[i])
        view_2d_from_3d(img=f(l_warped_path_list[i]), fpth=lwarped_save_path_list[i])
        view_2d_from_3d(img=f(l_warped_path_list[i]),phi=f_v(phi_path_list[i]), fpth=lwarped_phi_save_path_list[i])
        view_2d_from_3d(phi=f_v(phi_path_list[i]), fpth=phi_save_path_list[i])
        try:
            view_2d_from_3d(img=l, phi=f_v(inv_phi_path_list[i]), fpth=lwarped_invphi_save_path_list[i])
            view_2d_from_3d(phi=f_v(inv_phi_path_list[i]), fpth=inv_phi_save_path_list[i])
        except:
            pass
        view_2d_from_3d(img =warped ,phi=f_v(phi_path_list[i]), fpth=img_phi_save_path_list[i])
