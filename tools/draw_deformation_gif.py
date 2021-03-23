import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''
from tools.draw_deformation_viewers import *
from mermaid.data_utils import *
from glob import glob



def get_image_list_to_draw(refer_folder,img_type,t_list):
    """
    we first need to get a dict, where we can get the {"pair_name": "pair":[source,target], "fluid_warped":[warped],"phi"[phi for t=1],"t":[],"linear_warped"[]}
    :param refer_folder:
    :param img_type:
    :param source_txt:
    :return:
    """
    pair_path_list = glob(os.path.join(refer_folder,"*"+img_type))
    pair_name_list = [get_file_name(path).replace(img_type.split(".")[0],"") for path in pair_path_list]
    momentum_list = [os.path.join(refer_folder,pair_name+"_0000_Momentum.nii.gz") for pair_name in pair_name_list]
    source_path_list = [os.path.join(refer_folder, pair_name+"_t_0d01_image.nii.gz")  for pair_name in pair_name_list]
    target_path_list = [os.path.join(refer_folder,pair_name+"_test_iter_0_target.nii.gz")  for pair_name in pair_name_list]
    lsource_path_list = [os.path.join(refer_folder,pair_name+"_t_2d00_label.nii.gz")  for pair_name in pair_name_list]
    ltarget_path_list = [os.path.join(refer_folder,pair_name+"_test_iter_0_target_l.nii.gz")  for pair_name in pair_name_list]
    warped_path_list = [[os.path.join(refer_folder,pair_name+"_t_{}_image.nii.gz".format(str("{:.2f}".format(t)).replace(".","d")))for t in t_list]  for pair_name in pair_name_list ]
    phi_path_list =[[path.replace("_image.nii.gz","_phi_map.nii.gz") for path in paths] for paths in warped_path_list]
    lwarped_path_list = [[warped_path.replace("image.nii.gz","label.nii.gz") for warped_path in pair_warped_path] for pair_warped_path in warped_path_list]
    dict_to_draw = {}
    for i, pair_name in enumerate(pair_name_list):
        dict_to_draw[pair_name] = {"pair_name": pair_name, "pair_path":[source_path_list[i],target_path_list[i],lsource_path_list[i],ltarget_path_list[i]]
            ,"fluid_path":warped_path_list[i], "lfluid_path":lwarped_path_list[i],"phi_path":phi_path_list[i],"t":t_list,"momentum_path":momentum_list[i]}

    return dict_to_draw


def plot_2d_from_3d(ax,img=None, phi=None,title=""):
    if img is None:
        img = np.zeros_like(phi[0])
    if phi is None:
        ImageViewer3D_Sliced(ax, img, 0, title, False)
    else:
        ImageViewer3D_Sliced_Contour(ax, img, phi, 0, title, showColorbar=False)



def read_img_phi(img_path_list, phi_path_list=None):
    f = lambda pth: sitk.GetArrayFromImage(sitk.ReadImage(pth))
    img_list = [f(pth) for pth in img_path_list]
    phi_list = None
    if phi_path_list is not None:
        phi_list = [f(pth) for pth in phi_path_list]
        phi_list = [np.transpose(phi, (3, 2, 1,0)) for phi in phi_list]
    return img_list, phi_list



"""
dict_to_draw[pair_name] = {"pair_name": pair_name, "pair_path":[source_path_list[i],target_path_list[i],lsource_path_list[i],ltarget_path_list[i]]
            ,"fluid_path":warped_path_list[i], "lfluid_path":lwarped_path_list[i],"phi_path":phi_path_list[i],"phi1":phi1_path[i],"t":t_list,"momentum_path":momentum_list[i]}
for each pair name, we have source.png, target.png, momentum.png, phi_name.png, warped_name.png, l_warped_name.png """

def save_images(dict_to_draw,output_folder):
    for pair_name, pair_detail in dict_to_draw.items():
        output_path = os.path.join(output_folder,pair_name)
        os.makedirs(output_path,exist_ok=True)
        t_list = pair_detail['t']
        t_list[20] = 0
        source_path = pair_detail["pair_path"][0]
        target_path = pair_detail["pair_path"][1]
        lsource_path = pair_detail["pair_path"][2]
        ltarget_path = pair_detail["pair_path"][3]
        momentum_path =  pair_detail["momentum_path"]
        phi_path_list = pair_detail["phi_path"]
        warped_path_list =  pair_detail["fluid_path"]
        l_warped_path_list =  pair_detail["lfluid_path"]
        warped_name_list = [get_file_name(pth) for pth in warped_path_list]
        f = lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x))
        f_v = lambda x: np.transpose(f(x),[3,2,1,0])
        folder_path = os.path.join(output_folder,pair_name)
        os.makedirs(folder_path,exist_ok=True)
        for i in range(len(warped_name_list)):

            fig, ax = plt.subplots(2, 4, figsize=(30, 20))
            plt.style.use("bmh")
            plt.suptitle('t = {:.2f}'.format(t_list[i]), fontsize=80)
            plot_2d_from_3d(ax=ax[0,0],img=f(source_path),title="source")
            plot_2d_from_3d(ax=ax[0,1],img=f(target_path),title="target")
            plot_2d_from_3d(ax=ax[0,2],img=f(lsource_path),title="source_seg")
            plot_2d_from_3d(ax=ax[0,3],img=f(ltarget_path),title="target_seg")
            momentum = f_v(momentum_path)
            momentum = np.sum(momentum ** 2, 1)
            warped = f(warped_path_list[i])
            plot_2d_from_3d(ax=ax[1,0],img=warped,title="warped")
            plot_2d_from_3d(ax=ax[1,1],img=warped, phi=f_v(phi_path_list[i]),title="deformation")
            plot_2d_from_3d(ax=ax[1,2],img=f(l_warped_path_list[i]),phi=f_v(phi_path_list[i]),title="warped_seg")
            plot_2d_from_3d(ax=ax[1,3],img =momentum,title="initial m")
            plt.clim(vmin=-1., vmax=1.)
            if folder_path is None:
                plt.show()
                plt.clf()
            else:
                fpth = os.path.join(folder_path,("t_{:.2f}".format(t_list[i])).replace(".","d"))
                fpth = fpth+".png"
                plt.savefig(fpth, dpi=100, bbox_inches='tight')
                plt.close('all')

#
#
# def save_images(dict_to_draw,output_folder):
#     for pair_name, pair_detail in dict_to_draw.items():
#
#         t_list = pair_detail['t']
#         t_list[0] = 0
#         source_path = pair_detail["pair_path"][0]
#         target_path = pair_detail["pair_path"][1]
#         lsource_path = pair_detail["pair_path"][2]
#         ltarget_path = pair_detail["pair_path"][3]
#         momentum_path =  pair_detail["momentum_path"]
#         phi_path_list = pair_detail["phi_path"]
#         warped_path_list =  pair_detail["fluid_path"]
#         l_warped_path_list =  pair_detail["lfluid_path"]
#         warped_name_list = [get_file_name(pth) for pth in warped_path_list]
#         f = lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x))
#         f_v = lambda x: np.transpose(f(x),[3,2,1,0])
#         folder_path = None
#         if output_folder:
#             folder_path = os.path.join(output_folder, pair_name)
#             os.makedirs(folder_path,exist_ok=True)
#         for i in range(len(warped_name_list)):
#
#             fig, ax = plt.subplots(1, 5, figsize=(30, 8))
#             plt.style.use("grayscale")
#             plt.suptitle('t = {:.2f}'.format(t_list[i]), fontsize=80)
#             plot_2d_from_3d(ax=ax[0],img=f(source_path),title="source")
#             plot_2d_from_3d(ax=ax[1],img=f(target_path),title="target")
#             momentum = f_v(momentum_path)
#             momentum = np.sum(momentum ** 2, 1)
#             warped = f(warped_path_list[i])
#             plot_2d_from_3d(ax=ax[2],img=warped, title="warped")
#             plot_2d_from_3d(ax=ax[3],img=warped, phi=f_v(phi_path_list[i]),title="deformation")
#             plot_2d_from_3d(ax=ax[4],img =momentum,title="momentum")
#             plt.clim(vmin=-1., vmax=1.)
#             if folder_path is None:
#                 plt.show()
#                 plt.clf()
#             else:
#                 fpth = os.path.join(folder_path,("t_{:.2f}".format(t_list[i])).replace(".","d"))
#                 fpth = fpth+".png"
#                 plt.savefig(fpth, dpi=100, bbox_inches='tight')
#                 plt.close('all')

from tools.visual_tools import *
img_type = "_t_2d00_image.nii.gz"
t_list = [-1, -0.95,-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,
          0.01, 0.05,0.1, 0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,
          1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2
        ]
#/playpen-raid/zyshen/data/oai_reg/train_with_10/momentum_lresol/9397988_image_9074437_image_0000Momentum.nii.gz
refer_folder ="/playpen-raid/zyshen/reg_clean/demo/data_aug_demo_output/learnt_lddmm_oai_interpolation/aug"
dict_to_draw = get_image_list_to_draw(refer_folder,img_type,t_list)
#output_folder= None
output_folder = "/playpen-raid1/zyshen/data/oai_reg/draw_oai_gif_interpolation_color"
save_images(dict_to_draw,output_folder)
import imageio
for pair_name, pair_detail in dict_to_draw.items():
    images = []

    for i in range(len(t_list)):
        file_path = os.path.join(output_folder, pair_name, ("t_{:.2f}".format(t_list[i])).replace(".", "d"))
        file_path = file_path + ".png"
        images.append(imageio.imread(file_path))
    for i in range(5):
        images.append(imageio.imread(file_path))
    imageio.mimsave(os.path.join(output_folder,pair_name+".gif"), images)