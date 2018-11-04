"""
=================================
Box plots with custom fill colors
=================================

This plot illustrates how to create two types of box plots
(rectangular and notched), and how to fill them with custom
colors by accessing the properties of the artists of the
box plots. Additionally, the ``labels`` parameter is used to
provide x-tick labels for each sample.

A good general reference on boxplots and their history can be found
here: http://vita.had.co.nz/papers/boxplots.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Random test data
def get_experiment_data_from_record(order,path):
    data = np.load(path)
    return order,data
def get_experiment_data_from_record_detail(order, path):
    data_detail = np.load(path)
    data = np.mean(data_detail[:,1:],1)
    return order, data

def plot_box(data_list,name_list):
    fig, ax = plt.subplots(figsize=(20, 10))
    bplot = ax.boxplot(data_list, vert=True, patch_artist=True, return_type='axes')
    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(data_list))], )
    ax.set_xlabel('Method')
    ax.set_ylabel('Dice Score')
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
    fig1.savefig('/playpen/zyshen/plots/box_plot_intra.png',dpi=300)
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
    fig1.savefig('/playpen/zyshen/plots/trendency_intra.png',dpi=300)
    #plt.clf()

def compute_std(data_list,name_list):
    for i,name in enumerate(name_list):
        print("the mean and  std of the {}: is {} , {}".format(name, np.mean(data_list[i]), np.std(data_list[i])))


def get_list_from_dic(data_dic):
    data_list = [None for _ in  range(len(data_dic))]
    name_list = [None for _ in range(len(data_dic))]
    for key,item in data_dic.items():
        order = data_dic[key][0]
        data = data_dic[key][1]
        data_list[order]= data
        name_list[order]= key
    return data_list,name_list



def get_df_from_list(name_list, data_list1,data_list2):
    data_combined1 = np.array([])
    data_combined2 = np.array([])
    group_list = np.array([])
    for i in range(len(name_list)):
        data1 = data_list1[i]
        data2 = data_list2[i]
        if len(data1)!= len(data2):
            print("Warning the data1, data2 not consistant, the expr name is {}, len of data1 is {}, len of data2 is {}".format(name_list[i],len(data1),len(data2)))
        max_len = max(len(data1),len(data2))
        tmp_data1 = np.empty(max_len)
        tmp_data2 = np.empty(max_len)
        tmp_data1[:]= np.nan
        tmp_data2[:]= np.nan
        tmp_data1[:len(data1)] = data1
        tmp_data2[:len(data2)] = data2
        data_combined1 = np.append(data_combined1,tmp_data1)
        data_combined2 = np.append(data_combined2, tmp_data2)
        group_list = np.append(group_list, np.array([name_list[i]]*max_len))
    group_list = list(group_list)

    df = pd.DataFrame({'Group':group_list,'longitudinal':data_combined1, 'cross-subject':data_combined2})
    return df




def get_res_dic(draw_intra, draw_trendency):
    data_dic = {}
    if draw_intra:
        if not draw_trendency:
            #data_dic['af_ants'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_ants_affine_bi/records/records.npy')
            data_dic['affine_niftyreg'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_niftyreg_affine_jacobi/records/records_detail.npy')
            data_dic['affine_opt'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_baseline_affine_lncc_bi/records/records_detail.npy')
            #data_dic['affine_ncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_affine_net_single_bi/records/records_detail.npy')
            #data_dic['affine_cycle_step3'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_affine_net_cycle_step3_bi/records/records_detail.npy')
            #data_dic['affine_cycle_ncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_affine_net_cycle_bi/records/records_detail.npy')
            #data_dic['affine_sym_ncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_affine_net_sym_bi/records/records_detail.npy')
            data_dic['affine_sym_lncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_affine_net_sym_lncc_bi/records/records_detail.npy')

            data_dic['demons'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_demons_jacobi/records/records_detail.npy')
            data_dic['syn_ants'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_ants_refine_jacobi/records/records_detail.npy')
            data_dic['niftyreg_nmi'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_niftyreg_bspline_nmi_10_jacobi_save_img/records/records_detail.npy')
            data_dic['niftyreg_lncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_niftyreg_bspline_interv10_jacobi_save_img/records/records_detail.npy')
            data_dic['svf_opt'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_baseline_svf_jacobi_new2_moreiter_fixed/records/records_detail.npy')
            data_dic['AVSM'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/train_mermaid_net_reisd_2_4step_lncc_recbi/records/records_detail.npy')
            #data_dic['affine_svf_sym_lncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_baseline_svf_lncc_bilncc/records/records_detail.npy')
        else:
            data_dic['vSVF_iter1'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_mermaid_net_reisd_lncc_bi/records/records_detail.npy')
            data_dic['vSVF_iter2'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_mermaid_net_reisd_2intra_2step_lncc_recbi/records/records_detail.npy')
            data_dic['vSVF_iter3'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_mermaid_net_reisd_2intra_3step_lncc_recbi/records/records_detail.npy')
            data_dic['vSVF_iter4'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_mermaid_net_reisd_2intra_4step_lncc_recbi/records/records_detail.npy')
            data_dic['vSVF_iter5'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_mermaid_net_reisd_2intra_5step_lncc_recbi/records/records_detail.npy')
            data_dic['vSVF_iter6'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_mermaid_net_reisd_2intra_6step_lncc_recbi/records/records_detail.npy')
            data_dic['vSVF_iter7'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_mermaid_net_reisd_2intra_7step_lncc_recbi/records/records_detail.npy')



    else:
        if not draw_trendency:
            #data_dic['af_ants'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_ants_affine_bi/records/records.npy')
            data_dic['affine_niftyreg'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_niftyreg_affine_jacobi/records/records_detail.npy')
            data_dic['affine_opt'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_baseline_affine_recbi/records/records_detail.npy')
            #data_dic['affine_ncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_affine_net_single_recbi/records/records_detail.npy')
            #data_dic['affine_cycle_step3'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_affine_net_cycle_step3_recbi/records/records_detail.npy')
            #data_dic['affine_cycle_ncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_affine_net_cycle_recbi/records/records_detail.npy')
            #data_dic['affine_sym_ncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_affine_net_sym_recbi/records/records_detail.npy')
            data_dic['affine_sym_lncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_affine_net_sym_from_intra_recbi/records/records_detail.npy')

            data_dic['demons'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_demons_jacobi/records/records_detail.npy')
            data_dic['syn_ants'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_ants_refine_jacobi/records/records_detail.npy')
            data_dic['niftyreg_nmi'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_niftyreg_bspline_nmi_10_jacobi_save_img/records/records_detail.npy')
            #data_dic['niftyreg_improve'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_niftyreg_bspline_interv20_bi/records/records.npy')
            data_dic['niftyreg_lncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_niftyreg_bspline_interv10_jacobi_save_img/records/records_detail.npy')
            data_dic['svf_opt'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_baseline_svf_jacobi_new2_moreiter_fixed/records/records_detail.npy')
            data_dic['AVSM'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/test_intra_mermaid_net_500thisinst_10reg_double_loss_jacobi/records/records_detail.npy')
            #data_dic['affine_svf_sym_lncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_mermaid_net_reisd_sym_lncc_recbi/records/records_detail.npy')
            #data_dic['affine_svf_sym_lncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_baseline_svf_lncc_bilncc/records/records_detail.npy')
        else:
            data_dic['vSVF_iter1'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_mermaid_net_resid_lncc_recbi/records/records_detail.npy')
            data_dic['vSVF_iter2'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_mermaid_net_reisd_2_2step_lncc_recbi/records/records_detail.npy')
            data_dic['vSVF_iter3'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_mermaid_net_reisd_2_3step_lncc_recbi/records/records_detail.npy')
            data_dic['vSVF_iter4'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_mermaid_net_reisd_2_4step_lncc_recbi/records/records_detail.npy')
            data_dic['vSVF_iter5'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_mermaid_net_reisd_2_5step_lncc_recbi/records/records_detail.npy')
            data_dic['vSVF_iter6'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_mermaid_net_reisd_2_6step_lncc_recbi/records/records_detail.npy')
            data_dic['vSVF_iter7'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_mermaid_net_reisd_2_7step_lncc_recbi/records/records_detail.npy')

    return data_dic




def get_jacobi_dic(draw_intra, draw_trendency):
    data_dic = {}
    if draw_intra:

            data_dic['demons'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_demons_jacobi/records/records_jacobi_num.npy')
            data_dic['syn_ants'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_ants_refine_jacobi/records/records_jacobi_num.npy')
            data_dic['niftyreg_nmi'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_niftyreg_bspline_nmi_10_jacobi_save_img/records/records_jacobi_num.npy')
            data_dic['niftyreg_lncc'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_niftyreg_bspline_interv10_jacobi_save_img/records/records_jacobi_num.npy')
            data_dic['svf_opt'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_baseline_svf_jacobi_new2_moreiter_fixed/records/records_jacobi_num.npy')
            #data_dic['AVSM'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/train_mermaid_net_reisd_2_4step_lncc_recbi/records/records_jacobi.npy')
    else:
            data_dic['demons'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_demons_jacobi/records/records_jacobi_num.npy')
            data_dic['syn_ants'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_ants_refine_jacobi/records/records_jacobi_num.npy')
            data_dic['niftyreg_nmi'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_niftyreg_bspline_nmi_10_jacobi_save_img/records/records_jacobi_num.npy')
            data_dic['niftyreg_lncc'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_niftyreg_bspline_interv10_jacobi_save_img/records/records_jacobi_num.npy')
            data_dic['svf_opt'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_baseline_svf_jacobi_new2_moreiter_fixed/records/records_jacobi_num.npy')
            data_dic['AVSM'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/test_intra_mermaid_net_500thisinst_10reg_double_loss_jacobi/records/records_jacobi_num.npy')

    return data_dic


def get_multi_step_affine_dic(draw_intra):
    data_dic = {}
    if draw_intra:
        data_dic['iter2'] = get_experiment_data_from_record_detail(inc(),
                                                                        '/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/multi_step_compare_affine/run_affine_net_sym_lncc_multi_step_record_jacobi_2/records/records_detail.npy')
        data_dic['iter3'] = get_experiment_data_from_record_detail(inc(),
                                                                        '/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/multi_step_compare_affine/run_affine_net_sym_lncc_multi_step_record_jacobi_3/records/records_detail.npy')
        data_dic['iter4'] = get_experiment_data_from_record_detail(inc(),
                                                                        '/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/multi_step_compare_affine/run_affine_net_sym_lncc_multi_step_record_jacobi_4/records/records_detail.npy')
        data_dic['iter5'] = get_experiment_data_from_record_detail(inc(),
                                                                        '/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/multi_step_compare_affine/run_affine_net_sym_lncc_multi_step_record_jacobi_5/records/records_detail.npy')
        data_dic['iter6'] = get_experiment_data_from_record_detail(inc(),
                                                                        '/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/multi_step_compare_affine/run_affine_net_sym_lncc_multi_step_record_jacobi_6/records/records_detail.npy')
        data_dic['iter7'] = get_experiment_data_from_record_detail(inc(),
                                                                        '/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/multi_step_compare_affine/run_affine_net_sym_lncc_multi_step_record_jacobi_7/records/records_detail.npy')
        data_dic['iter8'] = get_experiment_data_from_record_detail(inc(),
                                                                        '/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/multi_step_compare_affine/run_affine_net_sym_lncc_multi_step_record_jacobi_8/records/records_detail.npy')

    else:
        data_dic['iter2'] = get_experiment_data_from_record_detail(inc(),
                                                                        '/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/multi_step_compare_affine/run_affine_net_sym_lncc_multi_step_record_jacobi_2/records/records_detail.npy')
        data_dic['iter3'] = get_experiment_data_from_record_detail(inc(),
                                                                        '/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/multi_step_compare_affine/run_affine_net_sym_lncc_multi_step_record_jacobi_3/records/records_detail.npy')
        data_dic['iter4'] = get_experiment_data_from_record_detail(inc(),
                                                                        '/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/multi_step_compare_affine/run_affine_net_sym_lncc_multi_step_record_jacobi_4/records/records_detail.npy')
        data_dic['iter5'] = get_experiment_data_from_record_detail(inc(),
                                                                        '/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/multi_step_compare_affine/run_affine_net_sym_lncc_multi_step_record_jacobi_5/records/records_detail.npy')
        data_dic['iter6'] = get_experiment_data_from_record_detail(inc(),
                                                                        '/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/multi_step_compare_affine/run_affine_net_sym_lncc_multi_step_record_jacobi_6/records/records_detail.npy')
        data_dic['iter7'] = get_experiment_data_from_record_detail(inc(),
                                                                        '/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/multi_step_compare_affine/run_affine_net_sym_lncc_multi_step_record_jacobi_7/records/records_detail.npy')
        data_dic['iter8'] = get_experiment_data_from_record_detail(inc(),
                                                                        '/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/multi_step_compare_affine/run_affine_net_sym_lncc_multi_step_record_jacobi_8/records/records_detail.npy')


    return data_dic



def get_sym_dic(draw_intra):
    data_dic = {}
    if draw_intra:
        #data_dic['demons'] = get_experiment_data_from_record(inc(),
        #                                                     '/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_demons_jacobi/records/records_jacobi_num.npy')
        data_dic['syn_ants'] = get_experiment_data_from_record(inc(),
                                                               '/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_ants_refine_jacobi/records/records_jacobi_num.npy')
        data_dic['niftyreg_nmi'] = get_experiment_data_from_record(inc(),
                                                                   '/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_niftyreg_bspline_nmi_10_jacobi_save_img/records/records_jacobi_num.npy')
        data_dic['niftyreg_lncc'] = get_experiment_data_from_record(inc(),
                                                                    '/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_niftyreg_bspline_interv10_jacobi_save_img/records/records_jacobi_num.npy')
        data_dic['svf_opt'] = get_experiment_data_from_record(inc(),
                                                              '/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_baseline_svf_jacobi_new2_moreiter_fixed/records/records_jacobi_num.npy')
        # data_dic['AVSM'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/train_mermaid_net_reisd_2_4step_lncc_recbi/records/records_jacobi.npy')
    else:
        # data_dic['demons'] = get_experiment_data_from_record(inc(),
        #                                                      '/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_demons_jacobi/records/records_jacobi_num.npy')
        data_dic['syn_ants'] = get_experiment_data_from_record(inc(),
                                                               '/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_ants_refine_jacobi/records/records_jacobi_num.npy')
        data_dic['niftyreg_nmi'] = get_experiment_data_from_record(inc(),
                                                                   '/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_niftyreg_bspline_nmi_10_jacobi_save_img/records/records_jacobi_num.npy')
        data_dic['niftyreg_lncc'] = get_experiment_data_from_record(inc(),
                                                                    '/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_niftyreg_bspline_interv10_jacobi_save_img/records/records_jacobi_num.npy')
        data_dic['svf_opt'] = get_experiment_data_from_record(inc(),
                                                              '/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_baseline_svf_jacobi_new2_moreiter_fixed/records/records_jacobi_num.npy')
        # data_dic['AVSM'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/train_mermaid_net_reisd_2_4step_lncc_recbi/records/records_jacobi.npy')

    return data_dic





def draw_group_boxplot(name_list,data_list1,data_list2):
    df = get_df_from_list(name_list,data_list1,data_list2)
    df = df[['Group', 'longitudinal', 'cross-subject']]
    dd = pd.melt(df, id_vars=['Group'], value_vars=['longitudinal', 'cross-subject'], var_name='task')
    fig, ax = plt.subplots(figsize=(15, 8))
    sn=sns.boxplot(x='Group', y='value', data=dd, hue='task', palette='Set2',ax=ax)
    #sns.palplot(sns.color_palette("Set2"))
    sn.set_xlabel('Method')
    sn.set_ylabel('Dice Score')
    # plt.xticks(rotation=45)
    ax.yaxis.grid(True)
    leg=plt.legend(prop={'size': 18},loc=4)
    leg.get_frame().set_alpha(0.2)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    plt.show()


def plot_group_trendency(trend_name, trend1, trend2):
    trend1_mean = [np.mean(data) for data in trend1]
    trend2_mean = [np.mean(data) for data in trend2]
    max_len = max(len(trend1),len(trend2))
    t = list(range(max_len))
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:red'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Average Dice', color=color)
    ln1 = ax1.plot(t, trend1_mean, color=color,linewidth=3.0, label='Longitudinal')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Average Dice', color=color)  # we already handled the x-label with ax1
    ln2 = ax2.plot(t, trend2_mean, color=color, linewidth=3.0,label='Cross-subject')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.xticks(t, trend_name, rotation=45)
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    leg = ax1.legend(lns, labs, loc=0)


    #leg = plt.legend(loc='best')
    #get the individual lines inside legend and set line width
    for line in leg.get_lines():
        line.set_linewidth(4)
    # get label texts inside legend and set font size
    for text in leg.get_texts():
        text.set_fontsize('x-large')


    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label,ax2.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()+ ax2.get_yticklabels()):
        item.set_fontsize(15)
    for tick in ax1.get_xticklabels():
        tick.set_rotation(30)

    #fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

order = -1

def inc():
    global order
    order +=1
    return order



draw_trendency = True
draw_boxplot = False

##################################Get Data ##############################################################


# # get dice box plot data

# data_list1, name_list = get_list_from_dic(get_res_dic(draw_intra=True, draw_trendency=False))
# order = -1
# data_list2, _ = get_list_from_dic(get_res_dic(draw_intra=False, draw_trendency=False))
# order = -1



# ##  get multi-step svf trend data
# data_list1, name_list = get_list_from_dic(get_res_dic(draw_intra=True, draw_trendency=True))
# order = -1
# data_list2, _ = get_list_from_dic(get_res_dic(draw_intra=False, draw_trendency=True))
# order = -1
## get multi-step affine trend data

data_list1, name_list = get_list_from_dic(get_multi_step_affine_dic(draw_intra=True))
order = -1
data_list2, _ = get_list_from_dic(get_multi_step_affine_dic(draw_intra=False))
order = -1



# if not draw_trendency:
#     plot_box(data_list1, name_list)
# else:
#     plot_trendency(data_list1,name_list)


######################################################compute mean and std ##################################3


compute_std(data_list1, name_list)
print( "now compute the cross subject ")
compute_std(data_list2, name_list)
if draw_boxplot:
    draw_group_boxplot(name_list,data_list1,data_list2)
if draw_trendency:
    plot_group_trendency(name_list, data_list1, data_list2)





##############################################Jacobian###############################################
# print("Now lets compute jacobi")
# order = -1
# jacobi_list1, jacobi_name_list = get_list_from_dic(get_jacobi_dic(draw_intra=True, draw_trendency=False))
# order = -1
# jacobi_list2, _ = get_list_from_dic(get_jacobi_dic(draw_intra=False, draw_trendency=False))
# compute_std(jacobi_list1, jacobi_name_list)
# print( "now compute the cross subject ")
# compute_std(jacobi_list2, jacobi_name_list)
# #draw_group_boxplot(jacobi_name_list, jacobi_list1, jacobi_list2)