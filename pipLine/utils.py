import torch
import numpy as np


def get_pair(data, pair= True, target=None):
    if pair:
        return data[:,:,0], data[:,:,1]
    else:
        return data[:,:,0], target

def organize_data(moving, target, sched='depth_concat'):
    if sched == 'depth_concat':
        input = torch.cat([moving, target], dim=1)
    elif sched == 'width_concat':
        input = torch.cat((moving, target), dim=3)
    return input