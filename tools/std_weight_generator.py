import numpy as np

random = np.random.RandomState(2016)
std_list = np.array(range(2,10))/100
print(std_list)

def gaussian_stds_generater(random,num=4):
    candid_list = []
    candid_list.append(random.uniform(0.01, 0.1))
    while len(candid_list)<num:
        std_add = np.random.choice(std_list)
        candid_list.append(candid_list[-1]+std_add)
    return candid_list


def gaussian_weight_generater(random,num=4):
    candid_list = []
    while len(candid_list)<num:
        candid_list.append(random.uniform(0.1, 1))
    candid_list= np.array(candid_list)
    candid_list = candid_list/candid_list.sum()
    return list(candid_list)



num=4
num_std=15
num_weight=10
std_matrix = np.zeros((num_std,num))
weight_matrix = np.zeros((num_std*num_weight,num))
for i in range(num_std):
    std_res = gaussian_stds_generater(random,num)
    std_matrix[i,:]= std_res
    for j in range(num_weight):
        idx= i* num_weight + j
        weight_res = gaussian_weight_generater(random, num)
        weight_matrix[idx,:] = weight_res

np.save('stds.npy',std_matrix)
np.save('weights.npy',weight_matrix)

