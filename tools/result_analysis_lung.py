
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

def plot_box(data_list,name_list,label = 'Dice Score'):
    fig, ax = plt.subplots(figsize=(22, 10))
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



def get_df_from_double_list(name_list, data_list1,data_list2):
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

    df = pd.DataFrame({'Group':group_list,'Longitudinal':data_combined1, 'Cross-subject':data_combined2})
    return df




def get_res_dic():
    data_dic = {}

    #data_dic['af_ants'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_ants_affine_bi/records/records.npy')
    data_dic['affine_NiftyReg'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_niftyreg_affine_jacobi/records/records_detail.npy')
    data_dic['affine_opt'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_baseline_affine_recbi/records/records_detail.npy')
    #data_dic['affine_ncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_affine_net_single_recbi/records/records_detail.npy')
    #data_dic['affine_cycle_step3'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_affine_net_cycle_step3_recbi/records/records_detail.npy')
    #data_dic['affine_cycle_ncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_affine_net_cycle_recbi/records/records_detail.npy')
    #data_dic['affine_sym_ncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_affine_net_sym_recbi/records/records_detail.npy')
    data_dic['affine_net'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_affine_net_sym_from_intra_recbi/records/records_detail.npy')

    data_dic['Demons'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_demons_en2en1p2_jacobi/records/records_detail.npy')
    data_dic['SyN'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_ants_ncc/records/records_detail.npy')
    data_dic['NiftyReg_NMI'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_niftyreg_bspline_nmi_10_jacobi_save_img/records/records_detail.npy')
    #data_dic['niftyreg_improve'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_niftyreg_bspline_interv20_bi/records/records.npy')
    data_dic['NiftyReg_LNCC'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_niftyreg_bspline_bpsline_10_constrain_jacobi_save_img/records/records_detail.npy')
    data_dic['vSVF-opt'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_baseline_svf_jacobi_more_iter_save_def_fixed/records/records_detail.npy')
    data_dic['LDDMM-opt'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/lddmm_accumiter_absw_maskm_b2_fixed/reg/res/records/records_detail.npy')
    data_dic['RDMM-opt'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/rdmm_iter_wkw_formul_025_1_omt_2step_200sym_minstd_005_allinterp_maskv_epdffix/reg/res/records/records_detail.npy')
    data_dic['VoxelMorph(with aff)'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/vm_miccal_setting_zeroboundary_withbothlambda100sigma002withenlargedflowreg_symjacobi/reg/res/records/records_detail.npy')
    data_dic['vSVF-net'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/test_intra_mermaid_net_500thisinst_10reg_double_loss_jacobi/records/records_detail.npy')
    data_dic['LDDMM-net'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/reg_lddmm_2step_sym500/reg/res/records/records_detail.npy')
    data_dic['RDMM-net'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/rdmm_learn_sm006_sym_500_thisis_std005/reg/res/records/records_detail.npy')
    #data_dic['AVSM (3step)'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/test_intra_mermaid_net_500thisinst_10reg_double_loss_step3_jacobi/records/records_detail.npy')
    #data_dic['VoxelMorph(w/o aff)'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/vm_miccal_setting_zeroboundary_withbothlambda100sigma002withenlargedflowreg_withoutaffine_symjacobi/reg/res/records/records_detail.npy')
    #data_dic['affine_svf_sym_lncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_mermaid_net_reisd_sym_lncc_recbi/records/records_detail.npy')
    #data_dic['affine_svf_sym_lncc'] = get_experiment_data_from_record_detail(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_baseline_svf_lncc_bilncc/records/records_detail.npy')

    return data_dic




def get_jacobi_dic():
    data_dic = {}
    data_dic['Demons'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_demons_en2en1p2_jacobi/records/records_jacobi.npy')
    data_dic['SyN'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_ants_ncc/records/records_jacobi.npy')
    data_dic['NiftyReg_NMI'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_niftyreg_bspline_nmi_10_jacobi_save_img/records/records_jacobi.npy')
    data_dic['NiftyReg_LNCC'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_niftyreg_bspline_bpsline_10_constrain_jacobi_save_img/records/records_jacobi.npy')
    data_dic['vSVF-opt'] = get_experiment_data_from_record(inc(),
                                                                 '/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_baseline_svf_jacobi_more_iter_save_def_fixed/records/records_jacobi.npy')
    data_dic['LDDMM-opt'] = get_experiment_data_from_record(inc(),
                                                                   '/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/lddmm_accumiter_absw_maskm_b2_fixed/reg/res/records/records_jacobi.npy')
    data_dic['RDMM-opt'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/rdmm_iter_wkw_formul_025_1_omt_2step_200sym_minstd_005_allinterp_maskv_epdffix/reg/res/records/records_jacobi.npy')
    data_dic['VoxelMorph(with aff)'] = get_experiment_data_from_record(inc(),
                                                                              '/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/vm_miccal_setting_zeroboundary_withbothlambda100sigma002withenlargedflowreg_symjacobi/reg/res/records/records_jacobi.npy')
    data_dic['vSVF-net'] = get_experiment_data_from_record(inc(),
                                                                 '/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/test_intra_mermaid_net_500thisinst_10reg_double_loss_jacobi/records/records_jacobi.npy')
    data_dic['LDDMM-net'] = get_experiment_data_from_record(inc(),
                                                                   '/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/reg_lddmm_2step_sym500/reg/res/records/records_jacobi.npy')
    data_dic['RDMM-net'] = get_experiment_data_from_record(inc(),
                                                                  '/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/reg_adpt_lddamm_wkw_formul_05_1_omt_2step_200sym_minstd_005_sm008_allinterp_maskv_epdffix/reg/res/records/records_jacobi.npy')
    # data_dic['AVSM (3step)'] = get_experiment_data_from_record(inc(),
    #                                                    '/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/test_intra_mermaid_net_500thisinst_10reg_double_loss_step3_jacobi/records/records_jacobi.npy')

    return data_dic





def get_syth_dice():
    data_dic = {}
    # data_dic['svf'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/syn_data/rdmm_comparison/svf_reg1_range10_10_lncc_nomask_epdffix/svf_reg1_range10_10_lncc_nomask_epdffix/score/score.npy')
    # data_dic['lddmm'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/syn_data/rdmm_comparison/lddmm_reg10_range10_10_lncc_nomask/lddmm_reg10_range10_10_lncc_nomask/score/score.npy')
    # data_dic['rdmm'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/syn_data/rdmm_comparison/rddmm_maxstd008_omt_03_200iter_abs_reg10_range10_10_200/score/score.npy')
    data_dic['vSVF'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/syn_data/svf_400/score/score.npy')
    data_dic['LDDMM'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/syn_data/lddmm_400/score/score.npy')
    data_dic['RDMM'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/syn_data/rdmm_400/score/score.npy')

    return data_dic














###########################################

def get_omt_dice_dic(draw_intra, draw_trendency):
    data_dic = {}
    omt_list = list(np.array(range(1,10))/10) +[2,5,10]
    omt_key_list = [str(term) for term in omt_list]
    omt_expr_list = [str(term).replace('.','') for term in omt_key_list]
    expri_path = '/playpen/zyshen/data/syn_data/rdmm_expri_'
    for i,term in enumerate(omt_list):
        data_dic[omt_key_list[i]] = get_experiment_data_from_record(inc(),expri_path+omt_expr_list[i]+'.npy')
    return data_dic

def get_omt_average_jacobi_dic(draw_intra, draw_trendency):
    data_dic = {}
    omt_list = list(np.array(range(1,10))/10) +[2,5,10]
    omt_key_list = [str(term) for term in omt_list]
    omt_expr_list = [str(term).replace('.','') for term in omt_key_list]
    expri_path = '/playpen/zyshen/data/syn_data/rdmm_expri_'
    for i,term in enumerate(omt_list):
        data_dic[omt_key_list[i]] = get_experiment_data_from_record(inc(),expri_path+omt_expr_list[i]+'.npy')
    return data_dic



def draw_single_box_plt(name, data_list):
    pass





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
    fig, ax = plt.subplots(figsize=(22, 10))
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
        tick.set_rotation(30)
    if fpth is not None:
        plt.savefig(fpth,dpi=500, bbox_inches = 'tight')
        plt.close('all')
    else:
        plt.show()
        plt.clf()





def draw_group_boxplot(name_list,data_list1,data_list2, label ='Dice Score',titile=None, fpth=None ):
    df = get_df_from_double_list(name_list,data_list1,data_list2)
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
    ln1 = ax1.plot(t, trend1_mean, color=color,linewidth=3.0, label='Longitudinal')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(label, color=color)  # we already handled the x-label with ax1
    ln2 = ax2.plot(t, trend2_mean, color=color, linewidth=3.0,label='Cross-subject')
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














order = -1

def inc():
    global order
    order +=1
    return order


#
draw_trendency = False
draw_boxplot = False
title =None
label = "Dice Score"
##################################Get Data ##############################################################


#get dice box plot data
#
# data_list1, name_list = get_list_from_dic(get_syth_dice(),use_perc=True)
# order = -1
# fpth=None
# draw_boxplot = True
#
# os.makedirs('/playpen/zyshen/debugs/rdmm_res',exist_ok=True)
# fpth = '/playpen/zyshen/debugs/rdmm_res/syth_boxplot.png'
# draw_single_boxplot(name_list,data_list1,label=label,fpth=fpth,data_name='synth',title="Average Dice on Synthesis Data")

order = -1
data_list1, name_list = get_list_from_dic(get_res_dic(),use_perc=True)
order = -1

fpth='/playpen/zyshen/debugs/rdmm_res/oai_per.png'
draw_single_boxplot(name_list,data_list1,label=label,fpth=fpth,data_name='synth',title='Performance on OAI')

compute_std(data_list1, name_list)
data_list1, name_list = get_list_from_dic(get_res_dic(),use_perc=True)
compute_std(data_list1, name_list)

order = -1

######################################################compute mean and std ##################################3

# data_list1, name_list = get_list_from_dic(get_res_dic(draw_intra=True, draw_trendency=False),use_perc=True)
# order = -1
# data_list2, _ = get_list_from_dic(get_res_dic(draw_intra=False, draw_trendency=False),use_perc=True)
# order = -1
# compute_std(data_list1, name_list)
# print( "now compute the cross subject ")
# compute_std(data_list2, name_list)

#
# # #################################################### plot boxplot
# if draw_boxplot:
#     draw_group_boxplot(name_list,data_list1,data_list2,label=label)
# #
# ####################################################3 plot trend
# if draw_trendency:
#     plot_group_trendency(name_list, data_list1, data_list2,label, title)


# #
#
# #
# #############################################Jacobian###############################################
# print("Now lets compute jacobi for different methods")
# order = -1
# jacobi_list1, jacobi_name_list = get_list_from_dic(get_jacobi_dic(draw_intra=True, draw_trendency=False))
# order = -1
# jacobi_list2, _ = get_list_from_dic(get_jacobi_dic(draw_intra=False, draw_trendency=False))
# compute_std(jacobi_list1, jacobi_name_list)
# compute_jacobi_info(jacobi_list1, jacobi_name_list)
#
# print( "now compute the cross subject ")
# compute_std(jacobi_list2, jacobi_name_list)
#compute_jacobi_info(jacobi_list2, jacobi_name_list)
#draw_group_boxplot(jacobi_name_list, jacobi_list1, jacobi_list2)

# print("Now lets do jacobi statistic")
# order = -1
# jacobi_list1, jacobi_name_list = get_list_from_dic(get_group_jacobi_dic(draw_intra=True, draw_trendency=False),use_log=True)
# order = -1
# jacobi_list2, _ = get_list_from_dic(get_group_jacobi_dic(draw_intra=False, draw_trendency=False),use_log=True)
#
#
# draw_histogram(jacobi_name_list, jacobi_list1, jacobi_list2)
#
# print("Now lets sort the jacobi")
# order = -1
# data_dic ={}
# jacobi_name_list=['step_6']
# data_dic['step6'] = get_experiment_data_from_record(inc(),
#                                                     '/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/visualize_jacobi/records/records_jacobi.npy')
# jacobi_list2, _ = get_list_from_dic(data_dic)
# #sort_jacobi_info(jacobi_list1, jacobi_name_list,7)
# print( "now compute the cross subject ")
# sort_jacobi_info(jacobi_list2, jacobi_name_list,7)
#

