import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def get_df_from_list(name_list, data_list1,data_list2,cat_name=[]):
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

    df = pd.DataFrame({'Group':group_list,cat_name[0]:data_combined1, cat_name[1]:data_combined2})
    return df





def draw_group_boxplot(name_list,data_list1,data_list2, label ='Daily cost',titile=None, fpth=None ,cat_name=[]):
    df = get_df_from_list(name_list,data_list1,data_list2,cat_name)
    df = df[['Group', cat_name[0], cat_name[1]]]
    dd = pd.melt(df, id_vars=['Group'], value_vars=[ cat_name[0], cat_name[1]], var_name='cat')
    fig, ax = plt.subplots(figsize=(10, 8))
    sn=sns.boxplot(x='Group', y='value', data=dd, hue='cat', palette='Set2',ax=ax)
    #sns.palplot(sns.color_palette("Set2"))
    sn.set_xlabel('')
    sn.set_ylabel(label)
    # plt.xticks(rotation=45)
    ax.yaxis.grid(True)
    leg=plt.legend(prop={'size': 18},loc=0)
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
#
# name_list = ['Thursday','Friday','Saturday','Sunday']
# cat_name = ['Student(Area A)','Student(Area B)']
#
# data_list1= [np.clip(np.random.normal(20,10,100),3,None),np.clip(np.random.normal(25,15,100),5,None),
#             np.clip(np.random.normal(36,25,100),4,None),np.clip(np.random.normal(35,20,100),10,None)]
#
# data_list2= [np.clip(np.random.normal(30,15,100),2,None),np.clip(np.random.normal(35,20,100),7,None),
#             np.clip(np.random.normal(60,30,100),10,None),np.clip(np.random.normal(45,30,100),14,None)]
#
# draw_group_boxplot(name_list,data_list1,data_list2, label ='Daily cost',titile=None, fpth=None,cat_name=cat_name )




name_list = ['Method A','Method B','Method C']
cat_name = ['Group 1','Group 2' ]

data_list1= [np.clip(np.random.normal(80,2,100),3,None),np.clip(np.random.normal(78,3,100),0,None),
            np.clip(np.random.normal(67,10,100),4,None)]

data_list2= [np.clip(np.random.normal(70,5,100),2,None),np.clip(np.random.normal(79,4,100),7,None),
            np.clip(np.random.normal(57,8,100),14,None)]

draw_group_boxplot(name_list,data_list1,data_list2, label ='Performance',titile=None, fpth=None,cat_name=cat_name )
