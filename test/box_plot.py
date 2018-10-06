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
    bplot = ax.boxplot(data_list, vert=True, patch_artist=True)
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
    return data_list, data_dic

data_dic = {}

order=-1
draw_intra = False
draw_trendency = False
def inc():
    global order
    order +=1
    return order




if draw_intra:
    if not draw_trendency:
        #data_dic['af_ants'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_ants_affine_bi/records/records.npy')
        data_dic['affine_niftyreg'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_niftyreg_affine_bi/records/records.npy')
        data_dic['affine_opt'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_baseline_affine_lncc_bi/records/records_detail.npy')
        data_dic['affine_ncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_affine_net_single_bi/records/records_detail.npy')
        #data_dic['affine_cycle_step3'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_affine_net_cycle_step3_bi/records/records_detail.npy')
        data_dic['affine_cycle_ncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_affine_net_cycle_bi/records/records_detail.npy')
        data_dic['affine_sym_ncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_affine_net_sym_bi/records/records_detail.npy')
        data_dic['affine_sym_lncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_affine_net_sym_lncc_bi/records/records_detail.npy')

        data_dic['syn_ants'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_ants_refine_bi/records/records.npy')
        data_dic['demons'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_demons_dev1_recbi/records/records.npy')
        data_dic['niftyreg_nmi'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_niftyreg_bspline_nmi_bi/records/records.npy')
        data_dic['niftyreg_lncc'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_niftyreg_bspline_interv10_bi/records/records.npy')
        data_dic['svf_opt'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_baseline_svf_lncc_bi/records/records_detail.npy')
        data_dic['ASVM'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/train_mermaid_net_reisd_2_4step_lncc_recbi/records/records_detail.npy')
        #data_dic['affine_svf_sym_lncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_baseline_svf_lncc_bilncc/records/records_detail.npy')
    else:
        data_dic['vSVF_iter1'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_mermaid_net_reisd_lncc_bi/records/records.npy')
        data_dic['vSVF_iter2'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_mermaid_net_reisd_2intra_2step_lncc_recbi/records/records_detail.npy')
        data_dic['vSVF_iter3'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_mermaid_net_reisd_2intra_3step_lncc_recbi/records/records_detail.npy')
        data_dic['vSVF_iter4'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_mermaid_net_reisd_2intra_4step_lncc_recbi/records/records_detail.npy')
        data_dic['vSVF_iter5'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_mermaid_net_reisd_2intra_5step_lncc_recbi/records/records_detail.npy')
        data_dic['vSVF_iter6'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_mermaid_net_reisd_2intra_6step_lncc_recbi/records/records_detail.npy')
        data_dic['vSVF_iter7'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_mermaid_net_reisd_2intra_7step_lncc_recbi/records/records_detail.npy')



else:
    if not draw_trendency:
        #data_dic['af_ants'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_ants_affine_bi/records/records.npy')
        data_dic['affine_niftyreg'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_niftyreg_affine_recbi/records/records.npy')
        data_dic['affine_opt'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_baseline_affine_recbi/records/records_detail.npy')
        data_dic['affine_ncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_affine_net_single_recbi/records/records_detail.npy')
        #data_dic['affine_cycle_step3'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_affine_net_cycle_step3_recbi/records/records_detail.npy')
        data_dic['affine_cycle_ncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_affine_net_cycle_recbi/records/records_detail.npy')
        data_dic['affine_sym_ncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_affine_net_sym_recbi/records/records_detail.npy')
        data_dic['affine_sym_lncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_affine_net_sym_lncc_step7_recbi/records/records_detail.npy')

        data_dic['syn_ants'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_ants_refine_recbi/records/records.npy')
        data_dic['demons'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_demons_dev1_recbi/records/records.npy')
        data_dic['niftyreg_nmi'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_niftyreg_affine_recbi/records/records.npy')
        #data_dic['niftyreg_improve'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_niftyreg_bspline_interv20_bi/records/records.npy')
        data_dic['niftyreg_lncc'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_niftyreg_bspline_interv10_recbi/records/records.npy')
        data_dic['svf_opt'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_baseline_svf_recbi/records/records_detail.npy')
        data_dic['ASVM'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_mermaid_net_reisd_2_7step_lncc_recbi/records/records_detail.npy')
        #data_dic['affine_svf_sym_lncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_mermaid_net_reisd_sym_lncc_recbi/records/records_detail.npy')
        #data_dic['affine_svf_sym_lncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_baseline_svf_lncc_bilncc/records/records_detail.npy')
    else:
        data_dic['vSVF_iter1'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_mermaid_net_resid_lncc_recbi/records/records.npy')
        data_dic['vSVF_iter2'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_mermaid_net_reisd_2_2step_lncc_recbi/records/records_detail.npy')
        data_dic['vSVF_iter3'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_mermaid_net_reisd_2_3step_lncc_recbi/records/records_detail.npy')
        data_dic['vSVF_iter4'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_mermaid_net_reisd_2_4step_lncc_recbi/records/records_detail.npy')
        data_dic['vSVF_iter5'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_mermaid_net_reisd_2_5step_lncc_recbi/records/records_detail.npy')
        data_dic['vSVF_iter6'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_mermaid_net_reisd_2_6step_lncc_recbi/records/records_detail.npy')
        data_dic['vSVF_iter7'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_mermaid_net_reisd_2_7step_lncc_recbi/records/records_detail.npy')





data_list, name_list = get_list_from_dic(data_dic)
if not draw_trendency:
    plot_box(data_list, name_list)
else:
    plot_trendency(data_list,name_list)
compute_std(data_list, name_list)




#
# np.random.seed(123)
# all_data = [np.random.normal(0, std, 100) for std in range(1, 4)]
#
# fig, ax = plt.subplots(figsize=(9, 4))
#
# # rectangular box plot
# bplot = ax.boxplot(all_data,
#                          vert=True,   # vertical box aligmnent
#                          patch_artist=True)   # fill with color
#
#  # fill with color
# colors = ['pink', 'lightblue', 'lightgreen']
#
# # fill with colors
# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)
#
# # adding horizontal grid lines
# ax.yaxis.grid(True)
# ax.set_xticks([y+1 for y in range(len(all_data))], )
# ax.set_xlabel('xlabel')
# ax.set_ylabel('ylabel')
#
# # add x-tick labels
# plt.setp(ax, xticks=[y+1 for y in range(len(all_data))],
#          xticklabels=['x1', 'x2', 'x3', 'x4'])
#
# plt.show()