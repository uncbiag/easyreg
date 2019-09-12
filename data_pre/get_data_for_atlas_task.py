import os
from data_pre.reg_data_utils import write_list_into_txt, get_file_name, loading_img_list_from_files


def generate_atlas_set(original_txt_path,atlas_path,l_atlas_path, output_path,phase='train'):
    source_path_list,target_path_list,l_source_path_list, l_target_path_list=loading_img_list_from_files(original_txt_path)
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
    fname_list = [get_file_name(file_list[i][0])+'_'+get_file_name(file_list[i][1]) for i in range(file_num)]
    write_list_into_txt(pair_txt_path,file_list)
    write_list_into_txt(fn_txt_path,fname_list)


output_path = '/playpen/zyshen/data/reg_test_for_atlas'
# original_txt_path_dict={'train':'/playpen/zyshen/data/croped_for_reg_debug_3000_pair_oai_reg_inter/train/pair_path_list.txt',
#                         'val':'/playpen/zyshen/data/croped_for_reg_debug_3000_pair_oai_reg_inter/val/pair_path_list.txt',
#                         'debug':'/playpen/zyshen/data/croped_for_reg_debug_3000_pair_oai_reg_inter/debug/pair_path_list.txt'}
# atlas_path = '/playpen/zyshen/oai_data/croped_atlas/atlas.nii.gz'
# l_atlas_path = '/playpen/zyshen/oai_data/croped_atlas/atlas_label.nii.gz'



original_txt_path_dict={'test':'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/test/pair_path_list.txt'}
atlas_path = '/playpen/zyshen/oai_data/atlas/atlas.nii.gz'
l_atlas_path = '/playpen/zyshen/oai_data/atlas/atlas_label.nii.gz'

resize_atlas = False

if resize_atlas:
    from tools.image_rescale import resize_input_img_and_save_it_as_tmp
    atlas_path=resize_input_img_and_save_it_as_tmp(atlas_path,is_label=False,saving_path='/playpen/zyshen/oai_data/croped_atlas',fname='atlas.nii.gz')
    l_atlas_path = resize_input_img_and_save_it_as_tmp(l_atlas_path,is_label=True,saving_path='/playpen/zyshen/oai_data/croped_atlas',fname='atlas_label.nii.gz')

for phase,txt_path in original_txt_path_dict.items():
    generate_atlas_set(txt_path,atlas_path,l_atlas_path,output_path,phase)
