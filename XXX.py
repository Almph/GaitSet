import numpy as np
import xarray as xr
# a='a.npy'
# b=[1,2,3]
# np.save(a, b)
# c = np.load('a.npy')
# print(c)

frame_list=[np.random.rand(64, 44) for _ in range(9)]
num_list=['00'+str(i) for i in range(9)]
data_dict=xr.DataArray(frame_list, coords={'frame': num_list}, dims=['frame', 'H', 'W'])

print(data_dict.coords['frame'].values.tolist())
