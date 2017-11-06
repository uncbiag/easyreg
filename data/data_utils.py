import h5py


def read_file(path, type='h5py'):
    if type == 'h5py':
        f = h5py.File(path, 'r')
        data = f.attrs['data']
        info = f.attrs['info']
        f.close()
        return {'data':data, 'info': info}


def write_file(path, dic, type='h5py'):
    if type == 'h5py':
        f = h5py.File(path, 'w')
        f.attrs['data']= dic['data']
        f.attrs['info'] = dic['info'] if 'info' in dic else None
        f.close()

def save_to_h5py(path, img_pair_list, info):
    dic = {'data': img_pair_list, 'info': info}
    write_file(path, dic, type='h5py')
