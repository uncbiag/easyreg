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


"""
Background

1 Lateral_Ventricle_Left

2 Lateral_Ventricle_Right

3 Third_Ventricle

4 Fourth_Ventricle

5 Nucleus_Accumbens_Left

6 Nucleus_Accumbens_Right

7 Caudate_Left

8 Caudate_Right

9 Putamen_Left

10 Putamen_Right

11 GP_Left

12 GP_Right

13 Brainstem

14 Thalamus_Left

15 Thalamus_Right

16 Ventral_Diencephalon_Left

17 Ventral_Diencephalon_Right

18 Inf_Lat_Ven_-_Left

19 Inf_Lat_Ven_-Right

20 Hippocampus_Left

21 Hippocampus_Right

22 Amygdala_Left

23 Amygdala_Right


"""

"""

[
"Lateral Ventricle",
"Nucleus Accumbens",
"Caudate",
"Putamen",
"Globus Pallidus (GP)",
"Brainstem",
"Thalamus",
"Hippocampus",
"Amygdala" ]



"""

filtered_struct_name_list = [
"Lateral Ventricle",
"Nucleus Accumbens",
"Caudate",
"Putamen",
"Globus Pallidus (GP)",
"Brainstem",
"Thalamus",
"Hippocampus",
"Amygdala" ]

filtered_index_list = [[0,1],[4,5],[6,7],[8,9],[10,11],[12],[13,14],[19,20],[21,22]]

struct_name_list =  [
    "Lateral_Ventricle_Left",
    "Lateral_Ventricle_Right",
    "Third_Ventricle",
    "Fourth_Ventricle",
    "Nucleus_Accumbens_Left",
    "Nucleus_Accumbens_Right",
    "Caudate_Left",
    "Caudate_Right",
    "Putamen_Left",
    "Putamen_Right",
    "GP_Left",
    "GP_Right",
    "Brainstem",
    "Thalamus_Left",
    "Thalamus_Right",
    "Ventral_Diencephalon_Left",
    "Ventral_Diencephalon_Right",
    "Inf_Lat_Ven-Left",
    "Inf_Lat_Ven-Right",
    "Hippocampus_Left",
    "Hippocampus_Right",
    "Amygdala_Left",
    "Amygdala_Right"
]

surf_res=[
[0.93,0.947,0.826,0.888,0.842,0.811,0.922,0.917,0.932,0.935,0.835,0.865,0.851,0.896,0.916,0.853,0.895,0.781,0.751,0.843,0.885,0.807,0.81],
[0.946,0.951,0.818,0.844,0.881,0.808,0.899,0.915,0.903,0.894,0.868,0.865,0.886,0.924,0.927,0.898,0.914,0.798,0.815,0.884,0.874,0.881,0.794],
[0.963,0.955,0.916,0.848,0.825,0.828,0.912,0.926,0.912,0.907,0.898,0.88,0.886,0.923,0.924,0.882,0.893,0.833,0.721,0.88,0.898,0.875,0.829],
[0.97,0.946,0.824,0.906,0.828,0.81,0.942,0.944,0.946,0.94,0.9,0.872,0.874,0.904,0.93,0.908,0.917,0.796,0.799,0.895,0.9,0.877,0.88],
[0.902,0.877,0.768,0.839,0.784,0.596,0.886,0.895,0.88,0.867,0.837,0.838,0.727,0.882,0.881,0.828,0.834,0.728,0.529,0.797,0.82,0.763,0.741],
[0.898,0.909,0.835,0.872,0.82,0.794,0.846,0.871,0.86,0.898,0.852,0.853,0.824,0.879,0.904,0.837,0.866,0.725,0.641,0.761,0.717,0.784,0.794],
[0.916,0.904,0.867,0.868,0.81,0.761,0.904,0.896,0.919,0.892,0.767,0.841,0.903,0.916,0.915,0.829,0.849,0.792,0.757,0.887,0.869,0.855,0.828],
[0.857,0.86,0.791,0.862,0.772,0.823,0.865,0.858,0.913,0.917,0.843,0.902,0.887,0.913,0.91,0.877,0.884,0.819,0.733,0.857,0.874,0.801,0.802],
[0.859,0.821,0.83,0.899,0.834,0.824,0.895,0.898,0.892,0.917,0.881,0.88,0.831,0.916,0.901,0.917,0.889,0.571,0.457,0.819,0.819,0.829,0.812],
[0.905,0.917,0.812,0.859,0.746,0.76,0.898,0.901,0.916,0.925,0.876,0.864,0.904,0.902,0.896,0.865,0.858,0.817,0.783,0.894,0.895,0.84,0.827],
[0.872,0.931,0.853,0.774,0.856,0.853,0.914,0.918,0.924,0.915,0.81,0.826,0.915,0.919,0.915,0.88,0.865,0.641,0.718,0.856,0.894,0.827,0.791]
]


def get_experiment_data_by_class_from_record_details(path):
    data_detail = np.load(path)
    data_nobg = data_detail[:,1:]
    num_class = data_nobg.shape[1]
    data_dict = {}
    for c in range(num_class):
        data_dict[struct_name_list[c]] = (inc(), data_nobg[:,c])
    return data_dict


def compute_std(data_list,name_list):
    for i,name in enumerate(name_list):
        print("the mean and  std of the {}: is {:.2f} , {:.2f}".format(name, np.mean(data_list[i]), np.std(data_list[i])))

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



def get_df_from_group_list(compared_name_list, xaxis_name_list, expri_data_list):

    combined_dict = {}
    for i in range(len(compared_name_list)):
        data_combined = []
        group_list = np.array([])
        max_len = max(len(expri_data) for expri_data in expri_data_list[i])
        for j, data_list in enumerate(expri_data_list[i]):
            data_len = len(data_list)
            tmp_data = np.empty(max_len)
            tmp_data[:]= np.nan
            tmp_data[:data_len] = data_list
            data_combined = np.append(data_combined,tmp_data)
            group_list = np.append(group_list, np.array([xaxis_name_list[j]]*max_len))
        combined_dict[compared_name_list[i]] = data_combined
    group_list = list(group_list)
    combined_dict.update({"Group": group_list})
    df = pd.DataFrame(combined_dict)
    return df




record_path_list = ['/playpen-raid1/zyshen/data/brain_35/non_resize/custom_seg_res/seg/res/records/records_detail.npy',
    '/playpen-raid1/zyshen/data/brain_35/non_resize/seg_aug_train_bspline_res/seg/res/records/records_detail.npy',
    '/playpen-raid1/zyshen/data/brain_35/non_resize/seg_aug_train_and_test_res_trainedk2testk2/seg/res/records/records_detail.npy',
    '/playpen-raid1/zyshen/data/brain_35/non_resize/seg_aug_train_and_test_res_trainedk2testk2_r0d2/seg/res/records/records_detail.npy']
def get_res_dic():
    data_dic = {}
    #data_dic['af_ants'] = get_experiment_data_from_record(inc(),'/playpen/zyshen/data/reg_debug_labeled_oai_reg_inter/run_ants_affine_bi/records/records.npy')
    data_dic['non_aug'] = get_experiment_data_from_record_detail(inc(),record_path_list[0])
    data_dic['bspline'] = get_experiment_data_from_record_detail(inc(),record_path_list[1])
    data_dic['train_aug'] = get_experiment_data_from_record_detail(inc(),record_path_list[2])
    data_dic['train_test_aug'] = get_experiment_data_from_record_detail(inc(),record_path_list[3])

    return data_dic





def get_res_dic_by_class(method_record_path):
    data_dic= get_experiment_data_by_class_from_record_details(method_record_path)
    return data_dic

















def draw_single_boxplot(name_list,data_list,label ='Dice Score',fpth=None ,data_name= None,title=None):
    df = get_df_from_list(name_list,data_list,name=data_name)
    df = df[['Group', data_name]]
    dd = pd.melt(df, id_vars=['Group'], value_vars=[data_name], var_name='task')
    fig, ax = plt.subplots(figsize=(10, 10))
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
        item.set_fontsize(30)
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
    if fpth is not None:
        plt.savefig(fpth,dpi=500, bbox_inches = 'tight')
        plt.close('all')
    else:
        plt.show()
        plt.clf()





def draw_group_boxplot(compared_name_list,xaxis_name_list,expri_data_list, label ='Dice Score',title=None, fpth=None ):
    df = get_df_from_group_list(compared_name_list,xaxis_name_list,expri_data_list)
    df = df[['Group']+compared_name_list]
    dd = pd.melt(df, id_vars=['Group'], value_vars=compared_name_list, var_name='task')
    fig, ax = plt.subplots(figsize=(20, 8))
    sn=sns.boxplot(x='Group', y='value', data=dd, hue='task', palette='Set2',ax=ax)
    #sns.palplot(sns.color_palette("Set2"))
    sn.set_xlabel('')
    sn.set_ylabel(label,fontsize=150)
    sn.set_title(title,fontsize=300)

    # plt.xticks(rotation=45)
    ax.yaxis.grid(True)
    leg=plt.legend(prop={'size': 18},loc=4)
    leg.get_frame().set_alpha(0.2)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    for tick in ax.get_xticklabels():
        tick.set_rotation(15)
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


def get_index(item_list, index):
    return [item_list[idx] for idx in index]

def get_merge_expri_data(expri_data_list,index_group_list):
    expri_data_merged_list = []
    for expri_data in expri_data_list:
        expri_data_merge = []
        for index_group in index_group_list:
            expri_data_merge_tmp = 0
            for index in index_group:
                expri_data_merge_tmp += expri_data[index]
            expri_data_merge.append(expri_data_merge_tmp/len(index_group))
        expri_data_merged_list.append(expri_data_merge)
    return expri_data_merged_list


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
#draw_single_boxplot(name_list,data_list1,label=label,fpth=fpth,data_name='synth',title="Average Dice on Synthesis Data")

order = -1
data_list1, name_list = get_list_from_dic(get_res_dic(),use_perc=True)
order = -1



fpth=None #'/playpen/zyshen/debugs/rdmm_res/oai_per.png'
#draw_single_boxplot(name_list,data_list1,label=label,fpth=fpth,data_name='synth',title='Performance on 20 train, 11 test')

# compute_std(data_list1, name_list)
# data_list1, name_list = get_list_from_dic(get_res_dic(),use_perc=True)
# compute_std(data_list1, name_list)

order = -1




expri_data_list = []
name_list = []
for record_path in record_path_list:
    data_list , name_list = get_list_from_dic(get_res_dic_by_class(record_path),use_perc=True)
    expri_data_list.append(data_list)
    order = -1


struct_name_list= name_list
surf_data = np.array(surf_res).transpose()*100
surf_data = [surf_sub_data for surf_sub_data in surf_data]
expri_data_list.append(surf_data)

data_average_list = [np.array(expri_data).mean(0) for expri_data in expri_data_list]
index = [4,0,3]
name_list = ["non aug","bspline","ana_train_aug","anatomical aug","fastsurfer"]
#draw_single_boxplot(get_index(name_list,index),get_index(data_average_list,index),label=label,fpth=fpth,data_name='synth',title='Average over all structures')

compute_std(get_index(data_average_list,index), get_index(name_list,index))
filtered_expri_data = get_merge_expri_data(expri_data_list,filtered_index_list)
#draw_group_boxplot(get_index(name_list,index),filtered_struct_name_list, get_index(filtered_expri_data,index), label, title="Selected structures")


selected_expri_name_list = get_index(name_list,index)
selected_filtered_expri_data_list =  get_index(filtered_expri_data,index)

stuc_name_str="Method".ljust(20)
for struc_name in filtered_struct_name_list:
    stuc_name_str += struc_name.ljust(20)
print(stuc_name_str)


for i, selected_filtered_expri_data in enumerate(selected_filtered_expri_data_list):
    res_str = selected_expri_name_list[i].ljust(20)

    for struct in selected_filtered_expri_data:
        res_str +="  {:.2f} ({:.2f})  ".format(np.mean(struct),np.std(struct))
    print(res_str)



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



surf_complete_data =[
[1.0,0.93,0.947,0.826,0.888,0.842,0.811,0.922,0.917,0.932,0.935,0.835,0.865,0.851,0.896,0.916,0.853,0.895,0.781,0.751,0.843,0.885,0.807,0.81],
[1.0,0.946,0.951,0.818,0.844,0.881,0.808,0.899,0.915,0.903,0.894,0.868,0.865,0.886,0.924,0.927,0.898,0.914,0.798,0.815,0.884,0.874,0.881,0.794],
[1.0,0.963,0.955,0.916,0.848,0.825,0.828,0.912,0.926,0.912,0.907,0.898,0.88,0.886,0.923,0.924,0.882,0.893,0.833,0.721,0.88,0.898,0.875,0.829],
[1.0,0.97,0.946,0.824,0.906,0.828,0.81,0.942,0.944,0.946,0.94,0.9,0.872,0.874,0.904,0.93,0.908,0.917,0.796,0.799,0.895,0.9,0.877,0.88],
[0.999,0.902,0.877,0.768,0.839,0.784,0.596,0.886,0.895,0.88,0.867,0.837,0.838,0.727,0.882,0.881,0.828,0.834,0.728,0.529,0.797,0.82,0.763,0.741],
[0.999,0.898,0.909,0.835,0.872,0.82,0.794,0.846,0.871,0.86,0.898,0.852,0.853,0.824,0.879,0.904,0.837,0.866,0.725,0.641,0.761,0.717,0.784,0.794],
[1.0,0.916,0.904,0.867,0.868,0.81,0.761,0.904,0.896,0.919,0.892,0.767,0.841,0.903,0.916,0.915,0.829,0.849,0.792,0.757,0.887,0.869,0.855,0.828],
[1.0,0.857,0.86,0.791,0.862,0.772,0.823,0.865,0.858,0.913,0.917,0.843,0.902,0.887,0.913,0.91,0.877,0.884,0.819,0.733,0.857,0.874,0.801,0.802],
[0.999,0.859,0.821,0.83,0.899,0.834,0.824,0.895,0.898,0.892,0.917,0.881,0.88,0.831,0.916,0.901,0.917,0.889,0.571,0.457,0.819,0.819,0.829,0.812],
[1.0,0.905,0.917,0.812,0.859,0.746,0.76,0.898,0.901,0.916,0.925,0.876,0.864,0.904,0.902,0.896,0.865,0.858,0.817,0.783,0.894,0.895,0.84,0.827],
[1.0,0.872,0.931,0.853,0.774,0.856,0.853,0.914,0.918,0.924,0.915,0.81,0.826,0.915,0.919,0.915,0.88,0.865,0.641,0.718,0.856,0.894,0.827,0.791]
]

name_list = ["non_aug","bspline","ana_train_aug","ana_train_test_aug"]
res_dict = {name_list[i]:np.load(record_path_list[i]) for i in range(len(name_list))}
res_dict['fast_surfer'] = np.array(surf_complete_data)
res_dict["structure_id"] = ["background"]+struct_name_list
res_dict["case_id"] = [
"190031",
"153025",
"131217",
"130013",
"366446",
"159340",
"198451",
"672756",
"138534",
"163129",
"199655"]
import pickle
# f = open("/playpen-raid1/zyshen/data/brain_35/non_resize/res.pkl","wb")
# pickle.dump(res_dict,f)
# f.close()



with open('/playpen-raid1/zyshen/data/brain_35/non_resize/res.pkl', 'rb') as f:
    data_load = pickle.load(f)
print()