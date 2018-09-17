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
def get_experiment_data_from_record(path):
    data = np.load(path)
    return data
def get_experiment_data_from_record_detail(path):
    data_detail = np.load(path)
    data = np.mean(data_detail[:,1:],1)
    return data

def plot_box(data_list,name_list):
    fig, ax = plt.subplots(figsize=(9, 4))
    bplot = ax.boxplot(data_list, vert=True, patch_artist=True)
    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(data_list))], )
    ax.set_xlabel('xlabel')
    ax.set_ylabel('ylabel')
    plt.setp(ax, xticks=[y + 1 for y in range(len(data_list))],
             xticklabels=name_list)
    plt.show()


def get_list_from_dic(data_dic):
    data_list = []
    name_list = []
    for key,item in data_dic.items():
        data_list += [data_dic[key]]
        name_list += [key]
    return data_list, data_dic

data_dic = {}


data_dic['affine_opt'] = get_experiment_data_from_record_detail('/playpen/zyshen/data/reg_debug_labeled_oai_reg_intra/run_baseline_affine_lncc_bi/records/records_detail.npy')
# data_dic['svf_opt'] = get_experiment_data_from_record_detail()
# data_dic['affine_nifty_reg_nmi'] = get_experiment_data_from_record()
# data_dic['bspline_nifty_reg_nmi'] = get_experiment_data_from_record()
# data_dic['affine_nifty_reg_lncc'] = get_experiment_data_from_record()
# data_dic['bspline_nifty_reg_lncc'] = get_experiment_data_from_record()
# data_dic['affine_network_3step_lncc'] = get_experiment_data_from_record_detail()
# data_dic['affine_network_5step_lncc']= get_experiment_data_from_record_detail()
# data_dic['svf_network_lncc'] = get_experiment_data_from_record()

data_list, name_list = get_list_from_dic(data_dic)
plot_box(data_list, name_list)









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