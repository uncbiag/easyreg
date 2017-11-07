import torch
import numpy as np
import skimage

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


def save_image_with_scale(path, variable):
      arr = variable.cpu().data.numpy()
      arr = np.clip(arr, -1., 1.)
      arr = (arr+1.)/2 * 255.
      arr = arr.astype(np.uint8)
      skimage.io.imsave(path, arr)


def save_result(path, appendix, moving, target, reproduce):
    for i in range(moving.size(0)):
      save_image_with_scale(path+appendix+"_{:02d}_moving.tif".format(i), moving[i,0,...])
      save_image_with_scale(path+appendix+"_{:02d}_target.tif".format(i), target[i,0,...])
      save_image_with_scale(path+appendix+"_{:02d}_reproduce.tif".format(i),reproduce[i,0,...])