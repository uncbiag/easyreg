import os
from easyreg.reg_data_utils import write_list_into_txt, loading_img_list_from_files,generate_pair_name
from glob import glob

def generate_atlas_set(original_txt_path,atlas_path,l_atlas_path, output_path,phase='train',test_phase_path_list=None, test_phase_l_path_list=None):
    if phase!="test":
        source_path_list,target_path_list,l_source_path_list, l_target_path_list=loading_img_list_from_files(original_txt_path)
    else:
        source_path_list =test_phase_path_list
        l_source_path_list = test_phase_l_path_list
        target_path_list = []
        l_target_path_list = []
    source_path_list =source_path_list+target_path_list
    file_num = len(source_path_list)
    l_source_path_list = l_source_path_list+l_target_path_list
    target_path_list = [atlas_path for _ in range(file_num)]
    l_target_path_list = [l_atlas_path for _ in range(file_num)]
    if l_source_path_list is not None and l_target_path_list is not None:
        assert len(source_path_list) == len(l_source_path_list)
        file_list = [[source_path_list[i], target_path_list[i],l_source_path_list[i],l_target_path_list[i]] for i in range(file_num)]
    else:
        file_list = [[source_path_list[i], target_path_list[i]] for i in range(file_num)]
    output_phase_path = os.path.join(output_path,phase)
    os.makedirs(output_phase_path,exist_ok=True)
    pair_txt_path =  os.path.join(output_phase_path,'pair_path_list.txt')
    fn_txt_path =   os.path.join(output_phase_path,'pair_name_list.txt')
    fname_list = [generate_pair_name([file_list[i][0],file_list[i][1]]) for i in range(file_num)]
    write_list_into_txt(pair_txt_path,file_list)
    write_list_into_txt(fn_txt_path,fname_list)


output_path = '/playpen-raid/zyshen/oai_data/reg_test_for_atlas'
# original_txt_path_dict={'train':'/playpen-raid/zyshen/data/croped_for_reg_debug_3000_pair_oai_reg_inter/train/pair_path_list.txt',
#                         'val':'/playpen-raid/zyshen/data/croped_for_reg_debug_3000_pair_oai_reg_inter/val/pair_path_list.txt',
#                         'debug':'/playpen-raid/zyshen/data/croped_for_reg_debug_3000_pair_oai_reg_inter/debug/pair_path_list.txt'}
# atlas_path = '/playpen-raid/zyshen/oai_data/croped_atlas/atlas.nii.gz'
# l_atlas_path = '/playpen-raid/zyshen/oai_data/croped_atlas/atlas_label.nii.gz'



original_txt_path_dict={phase:'/playpen-raid/zyshen/data/croped_for_reg_debug_3000_pair_oai_reg_inter/{}/pair_path_list.txt'.format(phase) for phase in ['train','val','debug']}
atlas_path = '/playpen-raid/zyshen/oai_data/atlas/atlas.nii.gz'
l_atlas_path = '/playpen-raid/zyshen/oai_data/atlas/atlas_label.nii.gz'
test_phase_folder = "/playpen-raid/zyshen/oai_data/Nifti_rescaled"
test_phase_path_list = glob(os.path.join(test_phase_folder, "*_image.nii.gz"))
test_phase_l_path_list = glob(os.path.join(test_phase_folder, "*_label_all.nii.gz"))

resize_atlas = False

if resize_atlas:
    from tools.image_rescale import resize_input_img_and_save_it_as_tmp
    atlas_path=resize_input_img_and_save_it_as_tmp(atlas_path,is_label=False,saving_path='/playpen-raid/zyshen/oai_data/croped_atlas',fname='atlas.nii.gz')
    l_atlas_path = resize_input_img_and_save_it_as_tmp(l_atlas_path,is_label=True,saving_path='/playpen-raid/zyshen/oai_data/croped_atlas',fname='atlas_label.nii.gz')

for phase,txt_path in original_txt_path_dict.items():
    generate_atlas_set(txt_path,atlas_path,l_atlas_path,output_path,phase)
generate_atlas_set(None,atlas_path,l_atlas_path,output_path,"test",test_phase_path_list,test_phase_l_path_list)