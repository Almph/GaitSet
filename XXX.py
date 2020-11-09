import numpy as np
import xarray as xr
import torch
from datetime import datetime

# a='a.npy'
# b=[1,2,3]
# np.save(a, b)
# c = np.load('a.npy')
# print(c)

# frame_list=[np.random.rand(64, 44) for _ in range(9)]
# num_list=['00'+str(i) for i in range(9)]
# data_dict=xr.DataArray(frame_list, coords={'frame': num_list}, dims=['frame', 'H', 'W'])

# print(type(data_dict.values))

# t=[np.asarray([[i, j] for i in range(3)]) for j in range(3)]
# print(t)
# print(t[1])

# a=np.random.rand(3, 2, 2)
# q=[a,a,a]
# c=np.concatenate(q, 0)
# print(c.shape)

# a=[[j] for j in range(3)]
# print(a)

# a=np.random.rand(3, 2, 2)
# b=np.pad(a, ((0, 2),(0, 0),(0, 0)), 'constant', constant_values=0)
# print(b.shape)
# print(b)

# a=torch.nn.Linear(3, 4)
# b=torch.optim.Adam([{'params': a.parameters()}], lr = 0.001)
# for q in b.param_groups:
#     # print(type(q['params'][0]))
#     print('>>>', q)

a=datetime.now()
print(a)