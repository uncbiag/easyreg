import h5py as hp
path = '/playpen/zyshen/debugs/lung_example/test.h5'

f = hp.File(path)
data = {}
for key in f.keys():
    data[key] = f[key][:]

pass
f.close()
