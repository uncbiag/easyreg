import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
# Random test data
def get_experiment_data_from_record(order,path):
    data = np.load(path)
    return order,data
def get_experiment_data_from_record_detail(order, path):
    data_detail = np.load(path)
    data = np.mean(data_detail[:,1:],1)
    return order, data

def plot_box(data_list,name_list,label = 'Dice Score'):
    fig, ax = plt.subplots(figsize=(20, 10))
    bplot = ax.boxplot(data_list, vert=True, patch_artist=True)
    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(data_list))], )
    ax.set_xlabel('Method')
    ax.set_ylabel(label)
    plt.setp(ax, xticks=[y + 1 for y in range(len(data_list))],
             xticklabels=name_list)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig('/playpen-raid/zyshen/plots/box_plot_intra.png',dpi=300)
    #plt.clf()



def plot_trendency(data_list,name_list):
    data_mean = [np.mean(data) for data in data_list]
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(data_mean)
    plt.xticks(np.arange(len(data_mean)), name_list, rotation=45)
    plt.title('vSVF self-iter')
    #plt.xlabel('vSVF self-iter')
    plt.ylabel('Dice Score')
    plt.show()
    plt.draw() 
    fig1.savefig('/playpen-raid/zyshen/plots/trendency_intra.png',dpi=300)
    #plt.clf()

def compute_std(data_list,name_list):
    for i,name in enumerate(name_list):
        print("the mean and  std of the {}: is {} , {}".format(name, np.mean(data_list[i]), np.std(data_list[i])))

def compute_jacobi_info(data_list,name_list):
    for i,name in enumerate(name_list):
        print("the mean and  num of jacobi image of the {}: is {} , {}".format(name, np.mean(data_list[i]), np.sum(data_list[i]>0)))

def sort_jacobi_info(data_list,name_list, top_n):
    for i, name in enumerate(name_list):
        print("the length of the data is {}".format(len(data_list[i])))
        sorted_index = data_list[i].argsort()[-top_n:][::-1]
        print("for method {}, the top {} jacobi is from id {}, with value {}".format(name,top_n,sorted_index,data_list[i][sorted_index] ))


def get_list_from_dic(data_dic,use_log=False,use_perc=False):
    data_list = [None for _ in  range(len(data_dic))]
    name_list = [None for _ in range(len(data_dic))]
    for key,item in data_dic.items():
        order = data_dic[key][0]
        data = data_dic[key][1]
        if use_log:
            data = np.log10(data)
            data = data[data != -np.inf]
        if use_perc:
            data = data*100
        data_list[order]= data
        name_list[order]= key
    return data_list,name_list




def get_df_from_list(name_list, data_list1,name=''):
    data_combined1 = np.array([])
    group_list = np.array([])
    for i in range(len(name_list)):
        data1 = data_list1[i]
        tmp_data1 = np.empty(len(data1))
        tmp_data1[:] = data1[:]
        data_combined1 = np.append(data_combined1,tmp_data1)
        group_list = np.append(group_list, np.array([name_list[i]]*len(data1)))
    group_list = list(group_list)
    df = pd.DataFrame({'Group':group_list,name:data_combined1})
    return df




def get_brainstorm_res(task_type,file_type='records.npy'):
    data_dic = {}

    data_dic['brainstorm'] = get_experiment_data_from_record(inc(),
                                                             '/playpen-raid1/zyshen/data/oai_reg/brain_storm/aug_expr/data_aug_fake_img_disp/res/seg/res/records/' + file_type)
    data_dic['brainstom_real'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid1/zyshen/data/oai_reg/brain_storm/aug_expr/data_aug_real_img_disp/res/seg/res/records/' + file_type)
    data_dic['fluid_aug'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid1/zyshen/data/oai_reg/brain_storm/aug_expr/data_aug_fake_img_fluid_sr/res/seg/res/records/' + file_type)
    data_dic['fluid_aug_real_t1'] = get_experiment_data_from_record(inc(),
                                                                 '/playpen-raid1/zyshen/data/oai_reg/brain_storm/aug_expr/data_aug_real_img_fluidt1/res/seg/res/records/' + file_type)
    data_dic['fluid_aug_real'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid1/zyshen/data/oai_reg/brain_storm/aug_expr/data_aug_real_img_fluid_sr/res/seg/res/records/' + file_type)
    data_dic['upper_bound'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid1/zyshen/data/oai_reg/brain_storm/aug_expr/upperbound/res/seg/res/records/' + file_type)
    return data_dic


def get_longitudinal_reg_res(task_type, file_type = "records.npy"):
    data_dic = {}
    data_dic['affine_network'] = get_experiment_data_from_record_detail(inc(),
                                                                        '/playpen-raid/zyshen/data/reg_debug_labeled_oai_reg_intra/run_affine_net_sym_lncc_bi/records/records_detail.npy')

    data_dic['non-aug'] = get_experiment_data_from_record(inc(),
                                                                 '/playpen-raid1/zyshen/data/reg_oai_aug/svf_net_scratch_res/reg/res/records/records.npy')
    data_dic['fluid-aug'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid1/zyshen/data/reg_oai_aug/svf_lld_scratch_res/reg/res/records/records.npy')

    data_dic['vSVF-opt'] = get_experiment_data_from_record_detail(inc(),
                                                                  '/playpen-raid/zyshen/data/reg_debug_labeled_oai_reg_intra/run_baseline_svf_jacobi_more_iter_save_def_fixed/records/records_detail.npy')

    data_dic['vSVF-net'] = get_experiment_data_from_record_detail(inc(),
                                                                  '/playpen-raid/zyshen/data/reg_debug_labeled_oai_reg_intra/test_intra_mermaid_net_500inst_10reg_double_loss_step2_jacobi/records/records_detail.npy')

    return data_dic



def get_lpba_post_dic(task_type,file_type='records.npy'):
    data_dic = {}


    if task_type=='post_aug_10':
        data_dic['5 patients'] = get_experiment_data_from_record(inc(),
                                                                 '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/10_v2/5case/seg/res/records/' + file_type)
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/10_v2/10case/seg/res/records/' + file_type)
        data_dic['15 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/10_v2/15case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/10_v2/20case/seg/res/records/' + file_type)
        data_dic['25 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/10_v2/25case/seg/res/records/' + file_type)

    if task_type=='post_aug_20':
        data_dic['5 patients'] = get_experiment_data_from_record(inc(),
                                                             '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/20_v2/5case/seg/res/records/' + file_type)
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/20_v2/10case/seg/res/records/' + file_type)
        data_dic['15 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/20_v2/15case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/20_v2/20case/seg/res/records/' + file_type)
        data_dic['25 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/20_v2/25case/seg/res/records/' + file_type)


    if task_type=='post_aug_30':
        data_dic['5 patients'] = get_experiment_data_from_record(inc(),
                                                             '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/30_v2/5case/seg/res/records/' + file_type)
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/30_v2/10case/seg/res/records/' + file_type)
        data_dic['15 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/30_v2/15case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/30_v2/20case/seg/res/records/' + file_type)
        data_dic['25 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/30_v2/25case/seg/res/records/' + file_type)

    if task_type == 'post_aug_10_t0':
        data_dic['5 patients'] = get_experiment_data_from_record(inc(),
                                                             '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/10_v2_w0d2/5case/seg/res/records/' + file_type)
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/10_v2_w0d2/10case/seg/res/records/' + file_type)
        data_dic['15 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/10_v2_w0d2/15case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/10_v2_w0d2/20case/seg/res/records/' + file_type)
        data_dic['25 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/10_v2_w0d2/25case/seg/res/records/' + file_type)

    if task_type == 'post_aug_20_t0':
        data_dic['5 patients'] = get_experiment_data_from_record(inc(),
                                                             '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/20_v2_w0d2/5case/seg/res/records/' + file_type)
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/20_v2_w0d2/10case/seg/res/records/' + file_type)
        data_dic['15 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/20_v2_w0d2/15case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/20_v2_w0d2/20case/seg/res/records/' + file_type)
        data_dic['25 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/20_v2_w0d2/25case/seg/res/records/' + file_type)

    if task_type == 'post_aug_30_t0':
        data_dic['5 patients'] = get_experiment_data_from_record(inc(),
                                                             '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/30_v2_w0d2/5case/seg/res/records/' + file_type)
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/30_v2_w0d2/10case/seg/res/records/' + file_type)
        data_dic['15 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/30_v2_w0d2/15case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/30_v2_w0d2/20case/seg/res/records/' + file_type)
        data_dic['25 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/30_v2_w0d2/25case/seg/res/records/' + file_type)

    return data_dic


def get_oai_post_dic(task_type,file_type='records.npy'):
    data_dic = {}


    if task_type=='post_aug_10':
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                                 '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/10/10case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/10/20case/seg/res/records/' + file_type)
        data_dic['30 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/10/30case/seg/res/records/' + file_type)
        data_dic['40 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/10/40case/seg/res/records/' + file_type)
        data_dic['60 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/10/60case/seg/res/records/' + file_type)

    if task_type == 'post_aug_20':
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/20/10case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/20/20case/seg/res/records/' + file_type)
        data_dic['30 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/20/30case/seg/res/records/' + file_type)
        data_dic['40 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/20/40case/seg/res/records/' + file_type)
        data_dic['60 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/20/60case/seg/res/records/' + file_type)

    if task_type == 'post_aug_30':
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/30/10case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/30/20case/seg/res/records/' + file_type)
        data_dic['30 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/30/30case/seg/res/records/' + file_type)
        data_dic['40 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/30/40case/seg/res/records/' + file_type)
        data_dic['60 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/30/60case/seg/res/records/' + file_type)

    if task_type=='post_aug_10_t0':
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                                 '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/10_w0d2/10case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/10_w0d2/20case/seg/res/records/' + file_type)
        data_dic['30 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/10_w0d2/30case/seg/res/records/' + file_type)
        data_dic['40 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/10_w0d2/40case/seg/res/records/' + file_type)
        data_dic['60 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/10_w0d2/60case/seg/res/records/' + file_type)

    if task_type == 'post_aug_20_t0':
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/20_w0d2/10case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/20_w0d2/20case/seg/res/records/' + file_type)
        data_dic['30 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/20_w0d2/30case/seg/res/records/' + file_type)
        data_dic['40 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/20_w0d2/40case/seg/res/records/' + file_type)
        data_dic['60 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/20_w0d2/60case/seg/res/records/' + file_type)

    if task_type == 'post_aug_30_t0':
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/30_w0d2/10case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/30_w0d2/20case/seg/res/records/' + file_type)
        data_dic['30 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/30_w0d2/30case/seg/res/records/' + file_type)
        data_dic['40 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/30_w0d2/40case/seg/res/records/' + file_type)
        data_dic['60 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/30_w0d2/60case/seg/res/records/' + file_type)

    return data_dic




def get_lpba_dic(task_type,file_type='records.npy'):
    data_dic = {}
    if task_type=='aug_2d':
        data_dic['5 patients_aug'] = get_experiment_data_from_record(inc(),
                                                                   '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/5case/best2_aug/seg/res/records/'+file_type)
        data_dic['10 patients_aug'] = get_experiment_data_from_record(inc(),
                                                                        '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/10case/best2_aug/seg/res/records/'+file_type)
        data_dic['15 patients_aug'] = get_experiment_data_from_record(inc(),
                                                                        '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/15case/best2_aug/seg/res/records/'+file_type)
        data_dic['20 patients_aug'] = get_experiment_data_from_record(inc(),
                                                                        '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/20case/best2_aug/seg/res/records/'+file_type)
        data_dic['25 patients_aug'] = get_experiment_data_from_record(inc(),
                                                                        '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/25case/best2_aug/seg/res/records/'+file_type)


    if task_type=='base':
        data_dic['5 patients'] = get_experiment_data_from_record(inc(),
                                                                 '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/5case/best2/seg/res/records/' + file_type)
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/10case/best2/seg/res/records/' + file_type)
        data_dic['15 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/15case/best2/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/20case/best2/seg/res/records/' + file_type)
        data_dic['25 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/25case/best2/seg/res/records/' + file_type)

    if task_type=='aug_aug':
        data_dic['5 patients'] = get_experiment_data_from_record(inc(),
                                                             '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/20_v2/5case/seg/res/records/' + file_type)
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/20_v2/10case/seg/res/records/' + file_type)
        data_dic['15 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/20_v2/15case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/20_v2/20case/seg/res/records/' + file_type)
        data_dic['25 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_aug_300/20_v2/25case/seg/res/records/' + file_type)

    if task_type=='bspline':
        data_dic['5 patients'] = get_experiment_data_from_record(inc(),
                                                                 '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_bspline/5case/seg/res/records/' + file_type)
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_bspline/10case/seg/res/records/' + file_type)
        data_dic['15 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_bspline/15case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_bspline/20case/seg/res/records/' + file_type)
        data_dic['25 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_bspline/25case/seg/res/records/' + file_type)

    if task_type=='rand':
        data_dic['5 patients'] = get_experiment_data_from_record(inc(),
                                                                 '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_rand/5case/seg/res/records/' + file_type)
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_rand/10case/seg/res/records/' + file_type)
        data_dic['15 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_rand/15case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_rand/20case/seg/res/records/' + file_type)
        data_dic['25 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_rand/25case/seg/res/records/' + file_type)
    if task_type=='aug_1d':
        data_dic['5 patients'] = get_experiment_data_from_record(inc(),
                                                                 '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_1d/5case/seg/res/records/' + file_type)
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_1d/10case/seg/res/records/' + file_type)
        data_dic['15 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_1d/15case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_1d/20case/seg/res/records/' + file_type)
        data_dic['25 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_1d/25case/seg/res/records/' + file_type)
    if task_type=='atlas':
        data_dic['5 patients'] = get_experiment_data_from_record(inc(),
                                                                 '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_atlas/5case/seg/res/records/' + file_type)
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_atlas/10case/seg/res/records/' + file_type)
        data_dic['15 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_atlas/15case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_atlas/20case/seg/res/records/' + file_type)
        data_dic['25 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/lpba_seg_resize/baseline/aug/sever_res/gen_lresol_atlas/25case/seg/res/records/' + file_type)

    return data_dic





def get_oai_dic(task_type,file_type='records.npy'):
    data_dic = {}
    if task_type=='aug_2d':
        data_dic['10 patients_aug'] = get_experiment_data_from_record(inc(),
                                                                   '/playpen-raid/zyshen/data/oai_seg/baseline/10case/best4_aug/seg/res/records/'+file_type)
        data_dic['20 patients_aug'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/20case/best3_aug/seg/res/records/' + file_type)
        data_dic['30 patients_aug'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/30case/best3_aug/seg/res/records/' + file_type)
        data_dic['40 patients_aug'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/40case/best3_aug/seg/res/records/' + file_type)
        data_dic['60 patients_aug'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/60case/best3_aug/seg/res/records/' + file_type)
        # data_dic['80 patients_aug'] = get_experiment_data_from_record(inc(),
        #                                                           '/playpen-raid/zyshen/data/oai_seg/baseline/80case/best3_aug/seg/res/records/' + file_type)
        # data_dic['100 patients_aug'] = get_experiment_data_from_record(inc(),
        #                                                           '/playpen-raid/zyshen/data/oai_seg/baseline/100case/best3_aug/seg/res/records/' + file_type)


    if task_type=='base':
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/10case/best3/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/20case/best3/seg/res/records/' + file_type)
        data_dic['30 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/30case/best3/seg/res/records/' + file_type)
        data_dic['40 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/40case/best3/seg/res/records/' + file_type)
        data_dic['60 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/60case/best3/seg/res/records/' + file_type)
        # data_dic['80 patients'] = get_experiment_data_from_record(inc(),
        #                                                           '/playpen-raid/zyshen/data/oai_seg/baseline/80case/best3/seg/res/records/' + file_type)
        # data_dic['100 patients'] = get_experiment_data_from_record(inc(),
        #                                                            '/playpen-raid/zyshen/data/oai_seg/baseline/100case/best3/seg/res/records/' + file_type)

    if task_type == 'aug_aug_1d':
        data_dic['10 patients_aug'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/20/10case/seg/res/records/' + file_type)
        data_dic['20 patients_aug'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/20/20case/seg/res/records/' + file_type)
        data_dic['30 patients_aug'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/20/30case/seg/res/records/' + file_type)
        data_dic['40 patients_aug'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/20/40case/seg/res/records/' + file_type)
        data_dic['60 patients_aug'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/test_ensemble_1d/20/60case/seg/res/records/' + file_type)
        # data_dic['80 patients_aug'] = get_experiment_data_from_record(inc(),
        #                                                           '/playpen-raid/zyshen/data/oai_seg/baseline/80case/best3_aug3/seg/res/records/' + file_type)
        # data_dic['100 patients_aug'] = get_experiment_data_from_record(inc(),
        #                                                           '/playpen-raid/zyshen/data/oai_seg/baseline/100case/best3_aug3/seg/res/records/' + file_type)

    # if task_type == 'aug_aug_1d':
    #     data_dic['10 patients_aug'] = get_experiment_data_from_record(inc(),
    #                                                               '/playpen-raid/zyshen/data/oai_seg/baseline/10case/best3_aug3/seg/res/records/' + file_type)
    #     data_dic['20 patients_aug'] = get_experiment_data_from_record(inc(),
    #                                                               '/playpen-raid/zyshen/data/oai_seg/baseline/20case/best3_aug3/seg/res/records/' + file_type)
    #     data_dic['30 patients_aug'] = get_experiment_data_from_record(inc(),
    #                                                               '/playpen-raid/zyshen/data/oai_seg/baseline/30case/best3_aug3/seg/res/records/' + file_type)
    #     data_dic['40 patients_aug'] = get_experiment_data_from_record(inc(),
    #                                                               '/playpen-raid/zyshen/data/oai_seg/baseline/40case/best3_aug3/seg/res/records/' + file_type)
    #     data_dic['60 patients_aug'] = get_experiment_data_from_record(inc(),
    #                                                               '/playpen-raid/zyshen/data/oai_seg/baseline/60case/best3_aug3/seg/res/records/' + file_type)
    #     # data_dic['80 patients_aug'] = get_experiment_data_from_record(inc(),
    #     #                                                           '/playpen-raid/zyshen/data/oai_seg/baseline/80case/best3_aug_1d/seg/res/records/' + file_type)
    #     # data_dic['100 patients_aug'] = get_experiment_data_from_record(inc(),
    #     #                                                           '/playpen-raid/zyshen/data/oai_seg/baseline/100case/best3_aug_1d/seg/res/records/' + file_type)

    if task_type=='bspline':
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/gen_lresol_bspline/10case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/gen_lresol_bspline/20case/seg/res/records/' + file_type)
        data_dic['30 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/gen_lresol_bspline/30case/seg/res/records/' + file_type)
        data_dic['40 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/gen_lresol_bspline/40case/seg/res/records/' + file_type)
        data_dic['60 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/gen_lresol_bspline/60case/seg/res/records/' + file_type)
        # data_dic['80 patients'] = get_experiment_data_from_record(inc(),
        #                                                           '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/gen_lresol_bspline/80case/seg/res/records/' + file_type)
        # data_dic['100 patients'] = get_experiment_data_from_record(inc(),
        #                                                            '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/gen_lresol_bspline/100case/seg/res/records/' + file_type)

    if task_type=='rand':
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/gen_lresol_rand/10case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/gen_lresol_rand/20case/seg/res/records/' + file_type)
        data_dic['30 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/gen_lresol_rand/30case/seg/res/records/' + file_type)
        data_dic['40 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/gen_lresol_rand/40case/seg/res/records/' + file_type)
        data_dic['60 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/gen_lresol_rand/60case/seg/res/records/' + file_type)
        # data_dic['80 patients'] = get_experiment_data_from_record(inc(),
        #                                                           '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/gen_lresol_rand/80case/seg/res/records/' + file_type)
        # data_dic['100 patients'] = get_experiment_data_from_record(inc(),
        #                                                            '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/gen_lresol_rand/100case/seg/res/records/' + file_type)

    if task_type == 'aug_1d':
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/gen_lresol_1d/10case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/gen_lresol_1d/20case/seg/res/records/' + file_type)
        data_dic['30 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/gen_lresol_1d/30case/seg/res/records/' + file_type)
        data_dic['40 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/gen_lresol_1d/40case/seg/res/records/' + file_type)
        data_dic['60 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/gen_lresol_1d/60case/seg/res/records/' + file_type)
        # data_dic['80 patients'] = get_experiment_data_from_record(inc(),
        #                                                           '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/gen_lresol_rand/80case/seg/res/records/' + file_type)
        # data_dic['100 patients'] = get_experiment_data_from_record(inc(),
        #                                                            '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res/gen_lresol_rand/100case/seg/res/records/' + file_type)

    return data_dic



def get_oai_dic2(task_type,file_type='records.npy'):
    data_dic = {}
    if task_type == 'bspline':
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_bspline/10case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_bspline/20case/seg/res/records/' + file_type)
        data_dic['30 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_bspline/30case/seg/res/records/' + file_type)
        data_dic['40 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_bspline/40case/seg/res/records/' + file_type)
        data_dic['60 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_bspline/60case/seg/res/records/' + file_type)
        data_dic['80 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_bspline/80case/seg/res/records/' + file_type)
        data_dic['100 patients'] = get_experiment_data_from_record(inc(),
                                                               '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_bspline/100case/seg/res/records/' + file_type)

    if task_type == 'aug_2d':
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_aug/10case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_aug/20case/seg/res/records/' + file_type)
        data_dic['30 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_aug/30case/seg/res/records/' + file_type)
        data_dic['40 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_aug/40case/seg/res/records/' + file_type)
        data_dic['60 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_aug/60case/seg/res/records/' + file_type)
        data_dic['80 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_aug/80case/seg/res/records/' + file_type)
        data_dic['100 patients'] = get_experiment_data_from_record(inc(),
                                                               '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_aug/100case/seg/res/records/' + file_type)

    if task_type == 'base':
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/base/10case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/base/20case/seg/res/records/' + file_type)
        data_dic['30 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/base/30case/seg/res/records/' + file_type)
        data_dic['40 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/base/40case/seg/res/records/' + file_type)
        data_dic['60 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/base/60case/seg/res/records/' + file_type)
        data_dic['80 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/base/80case/seg/res/records/' + file_type)
        data_dic['100 patients'] = get_experiment_data_from_record(inc(),
                                                               '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/base/100case/seg/res/records/' + file_type)


    if task_type == 'aug_1d':
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_1d/10case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_1d/20case/seg/res/records/' + file_type)
        data_dic['30 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_1d/30case/seg/res/records/' + file_type)
        data_dic['40 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_1d/40case/seg/res/records/' + file_type)
        data_dic['60 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_1d/60case/seg/res/records/' + file_type)
        data_dic['80 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_1d/80case/seg/res/records/' + file_type)
        data_dic['100 patients'] = get_experiment_data_from_record(inc(),
                                                                   '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_1d/100case/seg/res/records/' + file_type)


    if task_type == 'atlas':
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_atlas/10case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_atlas/20case/seg/res/records/' + file_type)
        data_dic['30 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_atlas/30case/seg/res/records/' + file_type)
        data_dic['40 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_atlas/40case/seg/res/records/' + file_type)
        data_dic['60 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_atlas/60case/seg/res/records/' + file_type)
        data_dic['80 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_atlas/80case/seg/res/records/' + file_type)
        data_dic['100 patients'] = get_experiment_data_from_record(inc(),
                                                                   '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_atlas/100case/seg/res/records/' + file_type)

    if task_type == 'rand':
        data_dic['10 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_rand/10case/seg/res/records/' + file_type)
        data_dic['20 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_rand/20case/seg/res/records/' + file_type)
        data_dic['30 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_rand/30case/seg/res/records/' + file_type)
        data_dic['40 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_rand/40case/seg/res/records/' + file_type)
        data_dic['60 patients'] = get_experiment_data_from_record(inc(),
                                                              '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_rand/60case/seg/res/records/' + file_type)
        data_dic['80 patients'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_rand/80case/seg/res/records/' + file_type)
        data_dic['100 patients'] = get_experiment_data_from_record(inc(),
                                                                   '/playpen-raid/zyshen/data/oai_seg/baseline/aug/sever_res_par/gen_lresol_rand/100case/seg/res/records/' + file_type)

    return data_dic



def draw_histogram(name_list, data_list1, data_list2, label="Jacobi Distribution",fpth=None):
    n_bins = 10

    fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(8, 5))
    ax0, ax1,= axes.flatten()

    ax0.hist(data_list1, n_bins, histtype='bar',label=name_list,range=[0, 4])
    ax0.set_title('Longitudinal logJacobi-Iteration Distribution (176 samples)')
    ax0.legend(prop={'size': 10},loc=2)
    ax1.hist(data_list2, n_bins, histtype='bar',label=name_list,range=[0, 4])
    ax1.set_title('Cross subject logJacobi-Iteration Distribution (300 samples)')
    ax1.legend(prop={'size': 10},loc=2)

    fig.tight_layout()
    if fpth is not None:
        plt.savefig(fpth,dpi=500, bbox_inches = 'tight')
        plt.close('all')
    else:
        plt.show()
        plt.clf()


def draw_single_boxplot(name_list,data_list,label ='Dice Score',titile=None, fpth=None ,data_name= None,title=None):
    df = get_df_from_list(name_list,data_list,name=data_name)
    df = df[['Group', data_name]]
    dd = pd.melt(df, id_vars=['Group'], value_vars=[data_name], var_name='task')
    fig, ax = plt.subplots(figsize=(12, 7))
    #fig, ax = plt.subplots(figsize=(12, 12))
    sn=sns.boxplot(x='Group', y='value', data=dd,ax=ax)
    sn.set_title(title,fontsize=50)
    #sns.palplot(sns.color_palette("Set2"))
    sn.set_xlabel('')
    sn.set_ylabel(label)
    # plt.xticks(rotation=45)
    ax.yaxis.grid(True)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
    if fpth is not None:
        plt.savefig(fpth,dpi=500, bbox_inches = 'tight')
        plt.close('all')
    else:
        plt.show()
        plt.clf()


def draw_group_boxplot(name_list,data_list1,data_list2, label ='Dice Score',titile=None, fpth=None):
    df = get_df_from_list(name_list,data_list1,data_list2)
    df = df[['Group', 'Longitudinal', 'Cross-subject']]
    dd = pd.melt(df, id_vars=['Group'], value_vars=['Longitudinal', 'Cross-subject'], var_name='task')
    fig, ax = plt.subplots(figsize=(15, 8))
    sn=sns.boxplot(x='Group', y='value', data=dd, hue='task', palette='Set2',ax=ax)
    #sns.palplot(sns.color_palette("Set2"))
    sn.set_xlabel('')
    sn.set_ylabel(label)
    # plt.xticks(rotation=45)
    ax.yaxis.grid(True)
    leg=plt.legend(prop={'size': 18},loc=4)
    leg.get_frame().set_alpha(0.2)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    if fpth is not None:
        plt.savefig(fpth,dpi=500, bbox_inches = 'tight')
        plt.close('all')
    else:
        plt.show()
        plt.clf()

def plot_group_trendency(trend_name, trend1, trend2,label='Average Dice', title=None,rotation_on = True,fpth=None):
    trend1_mean = [np.mean(data) for data in trend1]
    trend2_mean = [np.mean(data) for data in trend2]
    max_len = max(len(trend1),len(trend2))
    t = list(range(max_len))
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:red'
    #ax1.set_xlabel('step')
    ax1.set_ylabel(label, color=color)
    ln1 = ax1.plot(t, trend1_mean, color=color,linewidth=3.0, label='custom')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(label, color=color)  # we already handled the x-label with ax1
    ln2 = ax2.plot(t, trend2_mean, color=color, linewidth=3.0,label='aug')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.xticks(t, trend_name, rotation=45)
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    leg = ax1.legend(lns, labs, loc=0,prop={'size': 20})

    #leg = plt.legend(loc='best')
    #get the individual lines inside legend and set line width
    for line in leg.get_lines():
        line.set_linewidth(4)
    # get label texts inside legend and set font size
    for text in leg.get_texts():
        text.set_fontsize('x-large')


    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label,ax2.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()+ ax2.get_yticklabels()):
        item.set_fontsize(18)
    for tick in ax1.get_xticklabels():
        rotation = 0
        if rotation_on:
            rotation = 30
            tick.set_rotation(rotation)
    plt.title(title, fontsize=20)
    if fpth is not None:
        plt.savefig(fpth,dpi=500, bbox_inches = 'tight')
        plt.close('all')
    else:
        plt.show()
        plt.clf()
    #fig.tight_layout()  # otherwise the right y-label is slightly clipped


def draw_plots(trend_name, name_list, trend_list,label='Average Dice', title=None,rotation_on = True,fpth=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    trend_mean_list = [[np.mean(data) for data in trend] for trend in trend_list]
    trend_std_list = [[np.std(data) for data in trend] for trend in trend_list]
    max_len = max([len(trend) for trend in trend_list])
    t = list(range(max_len))


    for i in range(len(trend_list)):
        plt.errorbar(t,trend_mean_list[i], yerr=0, label=name_list[i])

    plt.legend(loc='lower right',fontsize=18)
    if rotation_on:
        plt.xticks(t, trend_name, rotation=20)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)

    ax.set_ylabel(label, fontsize=20)
    plt.title(title, fontsize=20)
    if fpth is not None:
        plt.savefig(fpth, dpi=500, bbox_inches='tight')
        plt.close('all')
    else:
        plt.show()
        plt.clf()

order = -1

def inc():
    global order
    order +=1
    return order




# data_list1, name_list = get_list_from_dic(get_oai_dic2(task_type='base',file_type = 'records.npy'))
# order = -1
# data_list2, _ = get_list_from_dic(get_oai_dic2(task_type='bspline',file_type = 'records.npy'))  #records_jacobi_num
# order = -1
# data_list3, _ = get_list_from_dic(get_oai_dic2(task_type='aug_2d',file_type = 'records.npy'))  #records_jacobi_num
# order =-1
# data_list4, _ = get_list_from_dic(get_oai_dic2(task_type='aug_1d',file_type = 'records.npy'))  #records_jacobi_num
# order = -1
# data_list5, _ = get_list_from_dic(get_oai_dic2(task_type='rand',file_type = 'records.npy'))  #records_jacobi_num
# order = -1
# data_list6, _ = get_list_from_dic(get_oai_dic2(task_type='atlas',file_type = 'records.npy'))  #records_jacobi_num
# order = -1
# label = 'Dice'
# title = 'OAI'
# fpath = "/playpen-raid/zyshen/debug/aug_res_plot/lpba40_seg.png"
# #plot_group_trendency(name_list, data_list1, data_list2,label, title,rotation_on=True,fpth=None)
# draw_plots(name_list,['non-aug','bspline','2d','1d',"rand","atlas"], [data_list1, data_list2,data_list3, data_list4,data_list5,data_list6],label, title,rotation_on=True,fpth=None)
#
#

#
data_list1, name_list = get_list_from_dic(get_lpba_dic(task_type='base',file_type = 'records.npy'))
order = -1
data_list2, _ = get_list_from_dic(get_lpba_dic(task_type='bspline',file_type = 'records.npy'))  #records_jacobi_num
order = -1
data_list3, _ = get_list_from_dic(get_lpba_dic(task_type='aug_2d',file_type = 'records.npy'))  #records_jacobi_num
order =-1
data_list4, _ = get_list_from_dic(get_lpba_dic(task_type='aug_aug',file_type = 'records.npy'))  #records_jacobi_num
order = -1
# data_list5, _ = get_list_from_dic(get_lpba_dic(task_type='rand',file_type = 'records.npy'))  #records_jacobi_num
# order = -1
label = 'Dice'
title = 'LPBA40'
fpath = "/playpen-raid/zyshen/debug/aug_res_plot/lpba40_seg.png"
#plot_group_trendency(name_list, data_list1, data_list2,label, title,rotation_on=True,fpth=None)
draw_plots(name_list,['non-aug','bspline','pre-aug','post-aug'], [data_list1, data_list2,data_list3, data_list4],label, title,rotation_on=True,fpth=fpath)



data_list1, name_list = get_list_from_dic(get_lpba_post_dic(task_type='post_aug_10',file_type = 'records.npy'))
order = -1
data_list2, _ = get_list_from_dic(get_lpba_post_dic(task_type='post_aug_20',file_type = 'records.npy'))  #records_jacobi_num
order =-1
data_list3, _ = get_list_from_dic(get_lpba_post_dic(task_type='post_aug_30',file_type = 'records.npy'))  #records_jacobi_num
order = -1
label = 'Dice'
title = 'LPBA40'
fpath = "/playpen-raid/zyshen/debug/aug_res_plot/ablation_lpba_post.png"
draw_plots(name_list,['10 times','20 times','30 times'], [ data_list1,data_list2,data_list3],label, title,rotation_on=True,fpth=fpath)


# #
# # data_list1, name_list = get_list_from_dic(get_lpba_post_dic(task_type='post_aug_10_t0',file_type = 'records.npy'))
# # order = -1
# # data_list2, _ = get_list_from_dic(get_lpba_post_dic(task_type='post_aug_20_t0',file_type = 'records.npy'))  #records_jacobi_num
# # order =-1
# # data_list3, _ = get_list_from_dic(get_lpba_post_dic(task_type='post_aug_30_t0',file_type = 'records.npy'))  #records_jacobi_num
# # order = -1
# # label = 'Dice'
# # title = 'LPBA40'
# # fpath = "/playpen-raid/zyshen/debug/aug_res_plot/ablation_lpba_post_t0.png"
# # draw_plots(name_list,['post_aug_10_t0','post_aug_20_t0','post_aug_30_t0'], [ data_list1,data_list2,data_list3],label, title,rotation_on=True,fpth=None)
#
#
#

data_list1, name_list = get_list_from_dic(get_oai_post_dic(task_type='post_aug_10',file_type = 'records.npy'))
order = -1
data_list2, _ = get_list_from_dic(get_oai_post_dic(task_type='post_aug_20',file_type = 'records.npy'))  #records_jacobi_num
order =-1
data_list3, _ = get_list_from_dic(get_oai_post_dic(task_type='post_aug_30',file_type = 'records.npy'))  #records_jacobi_num
order = -1
label = 'Dice'
title = 'OAI'
fpath = "/playpen-raid/zyshen/debug/aug_res_plot/ablation_oai_post.png"
draw_plots(name_list,['10 times','20 times','30 times'], [data_list1,data_list2,data_list3],label, title,rotation_on=True,fpth=fpath)

#
#
# # data_list1, name_list = get_list_from_dic(get_oai_post_dic(task_type='post_aug_10_t0',file_type = 'records.npy'))
# # order = -1
# # data_list2, _ = get_list_from_dic(get_oai_post_dic(task_type='post_aug_20_t0',file_type = 'records.npy'))  #records_jacobi_num
# # order =-1
# # data_list3, _ = get_list_from_dic(get_oai_post_dic(task_type='post_aug_30_t0',file_type = 'records.npy'))  #records_jacobi_num
# # order = -1
# # label = 'Dice'
# # title = 'OAI'
# # fpath = "/playpen-raid/zyshen/debug/aug_res_plot/ablation_oai_post_t0.png"
# # draw_plots(name_list,['post_aug_10_t0','post_aug_20_t0','post_aug_30_t0'], [data_list1,data_list2,data_list3],label, title,rotation_on=True,fpth=None)
# #
# #
#
#
data_list1, name_list = get_list_from_dic(get_lpba_dic(task_type='base',file_type = 'records.npy'))
order = -1
data_list2, _ = get_list_from_dic(get_lpba_dic(task_type='aug_1d',file_type = 'records.npy'))  #records_jacobi_num
order =-1
data_list3, _ = get_list_from_dic(get_lpba_dic(task_type='aug_2d',file_type = 'records.npy'))  #records_jacobi_num
order = -1
label = 'Dice'
title = 'LPBA40'
#plot_group_trendency(name_list, data_list1, data_list2,label, title,rotation_on=True,fpth=None)
fpath = "/playpen-raid/zyshen/debug/aug_res_plot/ablation_dim_lpba.png"
draw_plots(name_list,['K = 1','K = 2'], [ data_list2,data_list3],label, title,rotation_on=True,fpth=fpath)


data_list1, name_list = get_list_from_dic(get_oai_dic(task_type='base',file_type = 'records.npy'))
order = -1
data_list2, _ = get_list_from_dic(get_oai_dic(task_type='aug_1d',file_type = 'records.npy'))  #records_jacobi_num
order =-1
data_list3, _ = get_list_from_dic(get_oai_dic(task_type='aug_2d',file_type = 'records.npy'))  #records_jacobi_num
order = -1
label = 'Dice'
title = 'OAI'
#plot_group_trendency(name_list, data_list1, data_list2,label, title,rotation_on=True,fpth=None)
fpath = "/playpen-raid/zyshen/debug/aug_res_plot/ablation_dim_oai.png"
draw_plots(name_list,['K = 1','K = 2'], [ data_list2,data_list3],label, title,rotation_on=True,fpth=fpath)




# data_list1, name_list = get_list_from_dic(get_longitudinal_reg_res("",""),use_perc=True)
# order = -1
#
# fpath = "/playpen-raid/zyshen/debug/aug_res_plot/longtitudinal.png"
# draw_single_boxplot(name_list,data_list1,label="Dice",fpth=fpath,data_name='synth',title='Performance on OAI')
#
# compute_std(data_list1, name_list)



data_list1, name_list = get_list_from_dic(get_oai_dic(task_type='base',file_type = 'records.npy'))
order = -1
data_list2, _ = get_list_from_dic(get_oai_dic(task_type='bspline',file_type = 'records.npy'))  #records_jacobi_num
order = -1
data_list3, _ = get_list_from_dic(get_oai_dic(task_type='aug_1d',file_type = 'records.npy'))  #records_jacobi_num
order = -1

data_list4, _ = get_list_from_dic(get_oai_dic(task_type='aug_aug_1d',file_type = 'records.npy'))  #records_jacobi_num
order = -1

label = 'Dice'
title = 'OAI'
fpath = "/playpen-raid/zyshen/debug/aug_res_plot/oai_seg.png"
draw_plots(name_list,['non-aug','bspline','pre-aug','post-aug'], [data_list1, data_list2,data_list3,data_list4],label, title,rotation_on=True,fpth=fpath)




#
# data_list1, name_list = get_list_from_dic(get_oai_dic(task_type='aug',file_type = 'records.npy'))
# order = -1
# data_list2, _ = get_list_from_dic(get_oai_dic(task_type='aug_aug',file_type = 'records.npy'))
# order = -1
# data_list3, _ = get_list_from_dic(get_oai_dic(task_type='aug_1d',file_type = 'records.npy'))  #records_jacobi_num
# order =-1
# data_list4, _ = get_list_from_dic(get_oai_dic(task_type='aug_aug_1d',file_type = 'records.npy'))  #records_jacobi_num
# order = -1
# label = 'Dice'
# title = 'OAI'
# #plot_group_trendency(name_list, data_list1, data_list2,label, title,rotation_on=True,fpth=None)
# fpath = "/playpen-raid/zyshen/debug/aug_res_plot/ablation_dim_oai.png"
# draw_plots(name_list,['pre_aug_2d','post_aug_2d',"pre_aug_1d","post_aug_1d"], [ data_list1,data_list2,data_list3,data_list4],label, title,rotation_on=True,fpth=None)
#

data_list1, name_list = get_list_from_dic(get_brainstorm_res(task_type='',file_type = 'records.npy'))
order = -1
# # data_list2, _ = get_list_from_dic(get_oai_dic(task_type='1d',file_type = 'records.npy'))  #records_jacobi_num
# # order =-1
#
compute_std(data_list1, name_list)
# # print( "now compute the cross subject ")
# # compute_std(data_list2, name_list)