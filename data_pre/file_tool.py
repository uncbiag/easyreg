from easyreg.reg_data_utils import write_list_into_txt,read_txt_into_list, get_file_name
import numpy as np
import os
from glob import glob
import random
from copy import copy



def split_txt(input_txt,num_split, output_folder):
    os.makedirs(output_folder,exist_ok=True)
    pairs = read_txt_into_list(input_txt)
    output_splits = np.split(np.array(range(len(pairs))), num_split)
    output_splits = list(output_splits)
    for i in range(num_split):
        split = [pairs[ind] for ind in output_splits[i]]
        write_list_into_txt(os.path.join(output_folder, 'p{}.txt'.format(i)),split)


def get_file_list(path, ftype):
    f_pth = os.path.join(path,"**",ftype)
    file_list = glob(f_pth,recursive=True)
    file_list = [f for f in file_list]
    return file_list



def get_txt_file(path, ftype, output_txt):
    f_pth = os.path.join(path,"**",ftype)
    file_list = glob(f_pth,recursive=True)
    file_list = [[f] for f in file_list]
    write_list_into_txt(output_txt, file_list)

def get_img_label_txt_file(path, ftype,switcher, output_txt=None):
    import subprocess

    f_pth = os.path.join(path,ftype)
    file_list = glob(f_pth,recursive=True)
    file_list = [[f, f.replace(*switcher)] for f in file_list]
    for pair in file_list:
        if not os.path.isfile(pair[0]):
            print(pair[0])
        if not os.path.isfile(pair[1]):
            print(pair[1])
            cmd = "rm {}".format(pair[0])
            process = subprocess.Popen(cmd, shell=True)
            process.wait()

    write_list_into_txt(output_txt, file_list)


def transfer_txt_file_to_altas_txt_file(txt_path, atlas_path,output_txt,atlas_label_path,sever_switcher=("","")):
    """we would remove the seg info here"""
    img_label_list = read_txt_into_list(txt_path)
    img_label_list =[[pth.replace(*sever_switcher) for pth in pths] for pths in img_label_list]
    img_atlas_list = [[img_label[0],atlas_path,img_label[1],atlas_label_path] for img_label in img_label_list]
    img_atlas_list += [[atlas_path,img_label[0],atlas_label_path,img_label[1]] for img_label in img_label_list]
    write_list_into_txt(output_txt, img_atlas_list)


def compose_file_to_file_reg(source_txt, target_txt,output_txt,sever_switcher=('','')):
    source_img_label = read_txt_into_list(source_txt)
    target_img_label = read_txt_into_list(target_txt)
    num_s = len(source_img_label)
    num_t = len(target_img_label)
    pair = []
    for i in range(num_s):
        for j in range(num_t):
            line = [source_img_label[i][0],target_img_label[j][0],source_img_label[i][1],target_img_label[j][1]]
            line =[item.replace(*sever_switcher) for item in line]
            pair.append(line)

    write_list_into_txt(output_txt,pair)


def random_sample_from_txt(txt_path,num, output_path,switcher):
    pair_list = read_txt_into_list(txt_path)
    if num>0:
        sampled_list_rand = random.sample(pair_list, num)
        sampled_list = []
        for sample in sampled_list_rand:
            sampled_list.append([sample[0], sample[1], sample[2], sample[3]])
            sampled_list.append([sample[1], sample[0], sample[3], sample[2]])
    else:
        sampled_list = pair_list
    sampled_list = [[pth.replace(*switcher) for pth in pths] for pths in sampled_list]

    write_list_into_txt(output_path,sampled_list)


def get_file_txt_from_pair_txt(txt_path, output_path):
    pair_list = read_txt_into_list(txt_path)
    file_list = [[pair[0], pair[2]] for pair in pair_list]
    file_list += [[pair[1], pair[3]] for pair in pair_list]
    write_list_into_txt(output_path,file_list)


def random_sample_for_oai_inter_txt(txt_path,num_patient,num_pair_per_patient, output_path,mod,switcher):
    pair_list = read_txt_into_list(txt_path)
    num_per_m = int(num_patient/2)
    random.shuffle(pair_list)
    aug_pair_list = []
    sampled_list = []
    num_s = 0
    for i in range(len(pair_list)):
        while num_s<num_per_m:
            pair = pair_list[i]
            if mod in pair[0]:
                sampled_list.append([pair[0],pair[2]])
                sampled_list.append([pair[1],pair[3]])
                num_s += 1
            else:
                continue
    for i,sampled_source in enumerate(sampled_list):
        index = list(range(len(sampled_list)))
        index.remove(i)
        sampled_index = random.sample(index,num_pair_per_patient)
        for j in range(num_pair_per_patient):
            sampled_target = sampled_list[sampled_index[j]]
            aug_pair_list.append([sampled_source[0],sampled_target[0],sampled_source[1],sampled_target[1]])
    aug_pair_list = [[pth.replace(*switcher) for pth in pths] for pths in aug_pair_list]
    write_list_into_txt(output_path, aug_pair_list)









def get_pair_txt_for_color_net(atlas_path,atlas_label_path,inv_warped_folder,inv_w_type,output_txt):
    """the image label path is not needed for training, we use atlas_label_path to meet input format"""
    inv_warped_file_list = glob(os.path.join(inv_warped_folder,inv_w_type))
    pair_list = []
    for file in inv_warped_file_list:
        pair_list.append([atlas_path,file,atlas_label_path,atlas_label_path])
    write_list_into_txt(output_txt,pair_list)


def get_pair_txt_for_oai_reg_net(train_txt_path,warped_folder,warped_type, num_train,output_txt):
    train_pair_list = read_txt_into_list(train_txt_path)
    warped_file_list = glob(os.path.join(warped_folder,warped_type))
    name_set = [get_file_name(pair[0]).split("_")[0] for pair in train_pair_list]
    name_set = set(name_set)
    name_file_dict = {name:[] for name in name_set}
    extra_weight = 2
    for pair in train_pair_list:
        fname = get_file_name(pair[0]).split("_")[0]
        for i in range(extra_weight):
            name_file_dict[fname].append(pair[0])
            name_file_dict[fname].append(pair[1])
    for file in warped_file_list:
        fname = get_file_name(file).split("_")[0]
        name_file_dict[fname].append(file)
    num_per_patient = int(num_train/len(name_set))
    train_list = []
    for name,values in name_file_dict.items():
        num_sample = 0
        while num_sample < num_per_patient:
            pair = random.sample(name_file_dict[name],2)
            if get_file_name(pair[0])==get_file_name(pair[1]) or get_file_name(pair[0]).split("_")[1]==get_file_name(pair[1]).split("_")[1]:
                continue
            else:
                train_list.append(pair)
                num_sample += 1
    write_list_into_txt(output_txt, train_list)



def remove_label_info(pair_path_txt,output_txt):
    pair_list = read_txt_into_list(pair_path_txt)
    pair_remove_label = [[pair[0],pair[1]] for pair in pair_list]
    write_list_into_txt(output_txt, pair_remove_label)

def get_test_file_for_brainstorm_color(test_path,transfer_path,output_txt):
    #atlas_image_9023193_image_test_iter_0_warped.nii.gz
    file_label_list = read_txt_into_list(test_path)
    file_list, label_list = [file[0] for file in file_label_list],[file[1] for file in file_label_list]
    f = lambda x: "atlas_image_"+x+"_test_iter_0_warped.nii.gz"
    new_file_list = [os.path.join(transfer_path,f(get_file_name(file))) for file in file_list]
    new_file_label_list = [[new_file_list[i],label_list[i]] for i in range(len(file_label_list))]
    write_list_into_txt(output_txt,new_file_label_list)


def generate_file_for_xu():
    folder_path = "/playpen-raid1/xhs400/Research/data/r21/data/ct-cbct/images/"
    paths = glob(os.path.join(folder_path,"**","image_normalized.nii.gz"),recursive=True)
    outpath="/playpen-raid1/zyshen/debug/xu/"
    f = lambda x: "_OG" in x
    og_paths = list(filter(f,paths))
    em_paths = [og_path.replace("_OG","_EM") for og_path in og_paths]
    sm_paths = [og_path.replace("_OG","_SM") for og_path in og_paths]
    og_em_name = [[path1.split("/")[-4]+"_"+path1.split("/")[-2],path2.split("/")[-4]+"_"+path2.split("/")[-2]] for path1, path2 in zip(og_paths,em_paths)]
    og_sm_name = [[path1.split("/")[-4]+"_"+path1.split("/")[-2],path2.split("/")[-4]+"_"+path2.split("/")[-2]] for path1, path2 in zip(og_paths,sm_paths)]
    og_l_paths = [og_path.replace("image_normalized.nii.gz","SmBowel_label.nii.gz") for og_path in og_paths]
    em_l_paths =[og_path.replace("image_normalized.nii.gz","SmBowel_label.nii.gz") for og_path in em_paths]
    sm_l_paths =[og_path.replace("image_normalized.nii.gz","SmBowel_label.nii.gz") for og_path in sm_paths]
    pair_path_list = [[og_path,em_path,og_l_path,em_l_path] for og_path,em_path,og_l_path,em_l_path in zip(og_paths,em_paths,og_l_paths,em_l_paths)]
    pair_path_list += [[og_path,sm_path,og_l_path,sm_l_path] for og_path,sm_path,og_l_path,sm_l_path in zip(og_paths,sm_paths,og_l_paths,sm_l_paths)]
    fname_list = og_em_name + og_sm_name
    os.makedirs(outpath,exist_ok=True)
    pair_outpath = os.path.join(outpath,"source_target_set.txt")
    fname_outpath = os.path.join(outpath,"source_target_name.txt")
    write_list_into_txt(pair_outpath,pair_path_list)
    write_list_into_txt(fname_outpath,fname_list)




#generate_file_for_xu()


#
# test_path = "/playpen-raid/zyshen/data/oai_seg/test/file_path_list.txt"
# transfer_path = "/playpen-raid/zyshen/data/oai_reg/brainstorm/color_lrfix_test_res/reg/res/records/3D"
# output_txt = "/playpen-raid/zyshen/data//oai_reg/brainstorm/colored_test_for_seg.txt"
# get_test_file_for_brainstorm_color(test_path,transfer_path,output_txt)

# #
# test_path = "/playpen-raid/zyshen/data/oai_seg/val/file_path_list.txt"
# transfer_path = "/playpen-raid/zyshen/data/oai_reg/brainstorm/color_lrfix_val_res/reg/res/records/3D"
# output_txt = "/playpen-raid/zyshen/data//oai_reg/brainstorm/colored_val_for_seg.txt"
# get_test_file_for_brainstorm_color(test_path,transfer_path,output_txt)


# path = "/playpen-raid1/zyshen/data/oai_reg/brain_storm/data_aug_fake_img_fluidt1"
# ftype = "*image.nii.gz"
# switcher = ("image.nii.gz","label.nii.gz")
# output_path = "/playpen-raid1/zyshen/data/oai_reg/brain_storm/aug_expr/data_aug_fake_img_fluidt1/train"
# os.makedirs(output_path,exist_ok=True)
# output_path = os.path.join(output_path,'file_path_list.txt')
# get_img_label_txt_file(path,ftype,switcher,output_path)


#
#
# path = "/playpen-raid1/zyshen/data/oai_reg/brain_storm/data_aug_fake_img_fluid_sr"
# ftype = "*image.nii.gz"
# switcher = ("image.nii.gz","label.nii.gz")
# output_path = "/playpen-raid1/zyshen/data/oai_reg/brain_storm/aug_expr/data_aug_fake_img_fluid_sr/train"
# os.makedirs(output_path,exist_ok=True)
# output_path = os.path.join(output_path,'file_path_list.txt')
# get_img_label_txt_file(path,ftype,switcher,output_path)


# pair_path_txt ="/playpen-raid/zyshen/data/oai_reg/brainstorm/color_lrfix/train/pair_path_list.txt"
# output_txt = "/playpen-raid/zyshen/data/oai_reg/brainstorm/color_lrfix/test/pair_path_list.txt"
# remove_label_info(pair_path_txt,output_txt)

# pair_path_txt ="/playpen-raid/zyshen/data/oai_reg/brainstorm/val/atlas_to_val.txt"
# output_txt = "/playpen-raid/zyshen/data/oai_reg/brainstorm/val/atlas_to_val.txt"
# remove_label_info(pair_path_txt,output_txt)

#
# train_txt_path = '/playpen-raid1/zyshen/data/reg_oai_aug/train/pair_path_list.txt'
# warped_folder = '/playpen-raid1/zyshen/data/reg_oai_aug/data_aug'
# warped_type = '*_image.nii.gz'
# num_train = 2000
# output_folder = '/playpen-raid1/zyshen/data/reg_oai_aug/aug_net_cross/train'
# os.makedirs(output_folder,exist_ok=True)
# output_txt = os.path.join(output_folder,'pair_path_list.txt')
# get_pair_txt_for_oai_reg_net(train_txt_path, warped_folder, warped_type, num_train, output_txt)
#
#
#
# #
# train_txt_path = '/playpen-raid1/zyshen/data/reg_oai_aug/train/pair_path_list.txt'
# warped_folder = '/playpen-raid1/zyshen/data/reg_oai_aug/data_aug_bspline'
# warped_type = '*_image.nii.gz'
# num_train = 2000
# output_folder = '/playpen-raid1/zyshen/data/reg_oai_aug/aug_net_bspline_cross/train'
# os.makedirs(output_folder,exist_ok=True)
# output_txt = os.path.join(output_folder,'pair_path_list.txt')
# get_pair_txt_for_oai_reg_net(train_txt_path, warped_folder, warped_type, num_train, output_txt)
#

# train_txt_path = '/playpen-raid1/zyshen/data/reg_oai_aug/train/pair_path_list.txt'
# output_txt_path = '/playpen-raid1/zyshen/data/reg_oai_aug/bspline_path_list.txt'
# get_file_txt_from_pair_txt(train_txt_path,output_txt_path)




# atlas_path = '/playpen-raid/zyshen/data/oai_seg/atlas_image.nii.gz'
# atlas_label_path = '/playpen-raid/zyshen/data/oai_seg/atlas/atlas_label.nii.gz'
# inv_warped_folder ='/playpen-raid/zyshen/data/oai_reg/brainstorm/trans_lrfix_res/reg/res/records/3D'
# inv_w_type = "*atlas_image_test_iter_0_warped.nii.gz"
# output_folder = '/playpen-raid/zyshen/data/oai_reg/brainstorm/color_lrfix/train'
# os.makedirs(output_folder,exist_ok=True)
# output_txt = os.path.join(output_folder,'pair_path_list.txt')
# get_pair_txt_for_color_net(atlas_path,atlas_label_path,inv_warped_folder,inv_w_type,output_txt)
#
#
#
#

# num_sample = [50,-1,-1,-1]
# phase  = ['train','val','test','debug']
# for i, num in enumerate(num_sample):
#     txt_path = "/playpen-raid/zyshen/data/reg_debug_3000_pair_oai_reg_intra/{}/pair_path_list.txt".format(phase[i])
#     output_folder = "/playpen-raid1/zyshen/data/reg_oai_aug/{}".format(phase[i])
#     switcher=("","")
#     os.makedirs(output_folder,exist_ok=True)
#     output_path = os.path.join(output_folder,'pair_path_list.txt')
#     random_sample_from_txt(txt_path,num, output_path,switcher)



# num_split= 20
# input_txt = ''
#
# path = "/playpen-raid/zyshen/data/oai_reg/test_aug/reg/res/records/original_sz"
# ftype= "*_warped.nii.gz"
# output_txt ="/playpen-raid/zyshen/data/oai_reg/test_aug/file_path_list.txt"
# get_txt_file(path,ftype,output_txt)
#
# path = "/playpen-raid/zyshen/data/lpba_reg/test_aug/reg/res/records/original_sz"
# ftype= "*_warped.nii.gz"
# output_txt ="/playpen-raid/zyshen/data/lpba_reg/test_aug/file_path_list.txt"
# get_txt_file(path,ftype,output_txt)


# path = "/playpen-raid/zyshen/data/lpba_seg_resize/warped_img_label"
# ftype= "*_warped.nii.gz"
# switcher = ("_warped","_label")
# output_txt ="/playpen-raid/zyshen/data/lpba_reg/test_aug/file_path_label_list.txt"
# get_img_label_txt_file(path,ftype,switcher,output_txt)

#
# path = "/playpen-raid/zyshen/data/oai_seg/warped_img_label"
# ftype= "*_warped.nii.gz"
# switcher = ("_warped","_label")
# output_txt ="/playpen-raid/zyshen/data/oai_reg/test_aug/file_path_label_list.txt"
# get_img_label_txt_file(path,ftype,switcher,output_txt)


# phase= ["train","val","test","debug"]
# for p in phase:
#     txt_path = '/playpen-raid/zyshen/data/oai_seg/baseline/100case/{}/file_path_list.txt'.format(p)
#     atlas_path = '/playpen-raid/zyshen/data/oai_seg/atlas_image.nii.gz'
#     atlas_label_path = '/playpen-raid/zyshen/data/oai_seg/atlas/atlas_label.nii.gz'
#     output_txt= '/playpen-raid/zyshen/data/oai_reg/brainstorm/{}/pair_path_list.txt'.format(p)
#     #sever_switcher=('/playpen-raid/olut',"/pine/scr/z/y/zyshen/data")
#     sever_switcher=("","")
#
#     transfer_txt_file_to_altas_txt_file(txt_path, atlas_path, output_txt,atlas_label_path, sever_switcher)

# txt_path = '/playpen-raid/zyshen/data/oai_seg/baseline/100case/test/file_path_list.txt'
# atlas_path = '/playpen-raid/zyshen/data/oai_seg/atlas_image.nii.gz'
# output_txt= '/playpen-raid/zyshen/data/oai_seg/atlas/test.txt'
# txt_path = '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/25case/test/file_path_list.txt'
# atlas_path = '/playpen-raid/zyshen/data/lpba_seg_resize/atlas_image.nii.gz'
# output_txt= '/playpen-raid/zyshen/data/lpba_seg_resize/atlas/test.txt'
#transfer_txt_file_to_altas_txt_file(txt_path,atlas_path,output_txt,switcher)


#split_txt('/playpen-raid/zyshen/data/oai_seg/atlas/train.txt',4, '/playpen-raid/zyshen/data/oai_seg/atlas')


# num_c_list = [5,10,15,20,25]
# for num_c in num_c_list:
#     source_folder="/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/gen_lresol_multi_reg/{}case".format(num_c)
#     source_file_txt = os.path.join(source_folder,'file_path_list.txt')
#     get_img_label_txt_file(source_folder, "*_image.nii.gz",("_image","_label"),source_file_txt)
#     target_txt = '/playpen-raid/zyshen/data/lpba_seg_resize/test/file_path_list.txt'
#     output_txt = '/playpen-raid/zyshen/data/lpba_seg_resize/multi_reg_list_{}.txt'.format(num_c)
#     compose_file_to_file_reg(source_file_txt, target_txt,output_txt)

#
# num_c_list = [5,10,15,20,25]
# for num_c in num_c_list:
#     source_file_txt = '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/{}case/train/file_path_list.txt'.format(num_c)
#     target_txt = '/playpen-raid/zyshen/data/lpba_seg_resize/test/file_path_list.txt'
#     output_txt = '/playpen-raid/zyshen/data/lpba_seg_resize/multi_reg_list_baseline_{}.txt'.format(num_c)
#     compose_file_to_file_reg(source_file_txt, target_txt,output_txt,sever_switcher=("/playpen-raid","/pine/scr/z/y"))

# num_c_list = [5,10,15,20,25]
# for num_c in num_c_list:
#     split_txt('/playpen-raid/zyshen/data/lpba_seg_resize/multi_reg_list_{}.txt'.format(num_c),4, '/playpen-raid/zyshen/data/lpba_seg_resize/multi_reg/multi_reg_list_{}'.format(num_c))

#split_txt('/playpen-raid/zyshen/data/oai_reg/train_with_test_aug_40/test/pair_path_list_sever.txt',5,"/playpen-raid/zyshen/data/oai_reg/train_with_test_aug_40/test/opt")
