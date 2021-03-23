import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import time
# Random test data
def get_experiment_data_from_record(order,path):
    data = np.load(path)
    return order,data
def get_experiment_data_from_record_detail(order, path):
    data_detail = np.load(path)
    data = np.mean(data_detail[:,1:],1)
    return order, data

def collect_experiment_res(order,expri_folder):
    from glob import glob
    record_path_list = glob(os.path.join(expri_folder,"**","records_detail.npy"),recursive=True)
    data_list= []
    for record_path in record_path_list:
        order, data = get_experiment_data_from_record_detail(order, record_path)
        data_list.append(data)
    return np.array(data_list).squeeze()







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




def get_full_resol_res(expri_folder_dict):
    name_list = []
    data_list = []
    for name, expri_folder in expri_folder_dict.items():
        data = collect_experiment_res(inc(),expri_folder)
        name_list.append(name)
        data_list.append(data)
    return data_list, name_list



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





expri_folder_dict = {
    #"ants_inter":"/playpen-raid/zyshen/llf_mount/data/oai_data/expri/ants_inter",
    "demons_inter":"/playpen-raid/zyshen/llf_mount/data/oai_data/expri/demons_inter",
    "nifty_reg_inter":"/playpen-raid/zyshen/llf_mount/data/oai_data/expri/nifty_reg_inter",
    "avsm_inter":"/playpen-raid/zyshen/oai_data/expri/avsm_inter"
}
data_list, name_list = get_full_resol_res(expri_folder_dict)
order = -1
# data_list5, _ = get_list_from_dic(get_lpba_dic(task_type='rand',file_type = 'records.npy'))  #records_jacobi_num
# order = -1
label = 'Dice'
title = 'OAI Cross Object'
fpath = "/playpen-raid/zyshen/oai_data/expri/inter.png"
#draw_plots(name_list,name_list, data_list,label, title, rotation_on=True,fpth=None)
draw_single_boxplot(name_list,data_list,label=label,fpth=None,data_name='inter',title=title)



compute_std(data_list, name_list)



expri_folder_dict = {
    #"ants_atlas":"/playpen-raid/zyshen/llf_mount/data/oai_data/expri/ants_inter",
    "demons_atlas":"/playpen-raid/zyshen/llf_mount/data/oai_data/expri/demons_atlas",
    "nifty_reg_atlas":"/playpen-raid/zyshen/llf_mount/data/oai_data/expri/nifty_reg_atlas",
    "avsm_atlas":"/playpen-raid/zyshen/oai_data/expri/avsm_atlas"
}
data_list, name_list = get_full_resol_res(expri_folder_dict)
order = -1
# data_list5, _ = get_list_from_dic(get_lpba_dic(task_type='rand',file_type = 'records.npy'))  #records_jacobi_num
# order = -1
label = 'Dice'
title = 'OAI Atlas'
fpath = "/playpen-raid/zyshen/oai_data/expri/atlas.png"
#draw_plots(name_list,name_list, data_list,label, title, rotation_on=True,fpth=None)
draw_single_boxplot(name_list,data_list,label=label,fpth=None,data_name='atlas',title=title)
compute_std(data_list, name_list)
