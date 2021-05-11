import os
import argparse
import numpy as np
import random
parser = argparse.ArgumentParser(description='Data Organizer and Preprocessor')
parser.add_argument('--dataset_path', required=True, help='path to the root of the dataset')
parser.add_argument('--im2im', action='store_true')
parser.add_argument('--atlas', action='store_true')

parser.add_argument('--atlas_image_path', type=str, help='absolute path to atlas')
parser.add_argument('--atlas_label_path', type=str, help='absolute path to atlas')

parser.add_argument('--task_type', type=str, help='type of the task, can be either reg or seg', required=True)

parser.add_argument('--train_size', type=int, default=70)
parser.add_argument('--test_size', type=int, default=20)
parser.add_argument('--val_size', type=int, default=10)
parser.add_argument('--output_root_path', type=str)
parser.add_argument('--data_task_name', type=str)
parser.add_argument('--seed', type=int, default=1773)







# Data organization

# dataset_path / images 
# dataset_path / labels

# OR

# dataset_path / train / images
# dataset_path / train / labels

# dataset_path / test / images
# dataset_path / test / labels

# dataset_path / val / images
# dataset_path / val / labels

opt = parser.parse_args()
def clean_ds_store(dataset_path, mode=''):
    os.remove(os.path.join(opt.dataset_path, mode, 'images', '.DS_Store'))
    os.remove(os.path.join(opt.dataset_path, mode, 'labels', '.DS_Store'))

def check_if_equal_label_pairs(dataset_path, mode=''):
    try:
        print("Trying to clean up DS.Store files")
        clean_ds_store(dataset_path, mode)
    except:
        print("No DS.Store file found")
    if len(os.listdir(os.path.join(opt.dataset_path, mode, 'images'))) == len(os.listdir(os.path.join(opt.dataset_path, mode, 'labels'))):
        return True
    return False    



if opt.task_type != 'seg' and opt.task_type != 'reg':
    print("Invalid task type {}, it can be either seg or reg".format(opt.task_type))
    exit(1)

if opt.train_size + opt.test_size + opt.val_size != 100:
    print('Train, test, val split sum is {}%, returning error'.format(opt.train_size + opt.test_size + opt.val_size))
    exit(1)


curr_dir = os.getcwd()
file_list_paths = os.path.join(curr_dir, opt.output_root_path, opt.data_task_name)
os.makedirs(os.path.join(file_list_paths, 'train'), exist_ok=True)
os.makedirs(os.path.join(file_list_paths, 'test'), exist_ok=True)
os.makedirs(os.path.join(file_list_paths, 'val'), exist_ok=True)

file_names = dict()
file_names['train'] = {}
file_names['test'] = {}
file_names['val'] = {}
file_names['all'] = {}

already_splitted = False
 
# We do the split
if 'images' in os.listdir(opt.dataset_path) and 'labels' in os.listdir(opt.dataset_path):
    if not check_if_equal_label_pairs(opt.dataset_path):
        print('Unequal number of labels and images in dataset!')
        exit(1)



# It has been already splitted  
elif check_if_equal_label_pairs(opt.dataset_path, 'train') and check_if_equal_label_pairs(opt.dataset_path, 'test') and check_if_equal_label_pairs(opt.dataset_path, 'val'):
    already_splitted = True

else:
    print(''' # Data organization should be like following: \n
            \n
            dataset_path / images \n
            dataset_path / labels \n
            \n  
            OR\n
            \n
            dataset_path / train / images \n
            dataset_path / train / labels \n
            \n
            dataset_path / test / images \n
            dataset_path / test / labels \n
            \n
            dataset_path / val / images \n
            dataset_path / val / labels''')
    exit(1)

# We are splitting
# Assumption, when we sort, the labels and the images are matching.
if not already_splitted:
    file_names['all']['images'] = np.array(sorted(os.listdir(os.path.join(opt.dataset_path, 'images'))))
    file_names['all']['labels'] = np.array(sorted(os.listdir(os.path.join(opt.dataset_path, 'labels'))))
    indices = list(range(len(file_names['all']['images'])))
    random.shuffle(indices)
    file_names['all']['images'] = file_names['all']['images'][indices]
    file_names['all']['labels'] = file_names['all']['labels'][indices]
    len_of_dataset = len(indices)
    file_names['train']['images'] = file_names['all']['images'][:int(len(indices)*opt.train_size/100)]
    file_names['train']['labels'] = file_names['all']['labels'][:int(len(indices)*opt.train_size/100)]
    file_names['test']['images'] = file_names['all']['images'][int(len(indices)*opt.train_size/100):int(len(indices)*(opt.train_size+opt.test_size)/100)]
    file_names['test']['labels'] = file_names['all']['labels'][int(len(indices)*opt.train_size/100):int(len(indices)*(opt.train_size+opt.test_size)/100)]
    file_names['val']['images'] = file_names['all']['images'][-int(len(indices)*opt.val_size/100):]
    file_names['val']['labels'] = file_names['all']['labels'][-int(len(indices)*opt.val_size/100):]

    for mode in ['train', 'test', 'val']:
        file_names[mode]['images'] = [os.path.join(opt.dataset_path, 'images', path) for path in file_names[mode]['images']]
        file_names[mode]['labels'] = [os.path.join(opt.dataset_path, 'labels', path) for path in file_names[mode]['labels']]

    print(file_names['train']['images'])




else:
    for mode in ['train', 'test', 'val']:
        file_names[mode]['images'] = [os.path.join(opt.dataset_path, mode, 'images', path) for path in sorted(os.listdir(os.path.join(opt.dataset_path, mode, 'images')))]
        file_names[mode]['labels'] = [os.path.join(opt.dataset_path, mode, 'labels', path) for path in sorted(os.listdir(os.path.join(opt.dataset_path, mode, 'labels')))]
    





if opt.task_type == 'seg':
    # Create files
    for mode in ['train', 'test', 'val']:
        with open(os.path.join(file_list_paths, mode, 'file_path_list.txt'), 'w+') as f:
            for img, label in zip(file_names[mode]['images'], file_names[mode]['labels']):
                f.write('{} {}\n'.format(img, label))






    for mode in ['train', 'test', 'val']:
        with open(os.path.join(file_list_paths, mode, 'file_name_list.txt'), 'w+') as f:
            for img, label in zip(file_names[mode]['images'], file_names[mode]['labels']):
                f.write('img_{}\n'.format(img.split('.')[0].split('/')[-1]))

elif opt.task_type == 'reg':
    if not opt.atlas:
        for mode in ['train', 'test', 'val']:
            with open(os.path.join(file_list_paths, mode, 'pair_path_list.txt'), 'w+') as f:
                for img1, label1 in zip(file_names[mode]['images'], file_names[mode]['labels']):
                    for img2, label2 in zip(file_names[mode]['images'], file_names[mode]['labels']):
                        if img1 == img2:
                            continue
                        f.write('{} {} {} {}\n'.format(img1, img2, label1, label2))

        for mode in ['train', 'test', 'val']:
            with open(os.path.join(file_list_paths, mode, 'pair_name_list.txt'), 'w+') as f:
                for img1, label1 in zip(file_names[mode]['images'], file_names[mode]['labels']):
                    for img2, label2 in zip(file_names[mode]['images'], file_names[mode]['labels']):
                        if img1 == img2:
                            continue
                        f.write('{}_{}\n'.format(img1.split('.')[0].split('/')[-1], img2.split('.')[0].split('/')[-1]))
                        
    else:
        for mode in ['train', 'test', 'val']:
            with open(os.path.join(file_list_paths, mode, 'pair_path_list.txt'), 'w+') as f:
                for img1, label1 in zip(file_names[mode]['images'], file_names[mode]['labels']):
                    img2 = opt.atlas_image_path
                    f.write('{} {} {} {}\n'.format(img1, img2, label1, label1))

        for mode in ['train', 'test', 'val']:
            with open(os.path.join(file_list_paths, mode, 'pair_name_list.txt'), 'w+') as f:
                for img1, label1 in zip(file_names[mode]['images'], file_names[mode]['labels']):
                    img2 = 'atlas'
                    f.write('{}_{}\n'.format(img1.split('.')[0].split('/')[-1], img2.split('.')[0].split('/')[-1]))

        










