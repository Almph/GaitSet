import numpy as np
import xarray as xr
import torch
from datetime import datetime
import cv2

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

# a=datetime.now()
# print(a)

# a=torch.randint(0, 10, size=(1, 2, 1, 2, 2))
# print(a)
# b=torch.max(a, 1)
# print(b[0])

# a=torch.rand(2, 3, 4)
# print(a.shape)
# print(a.size())
# print(a.size(0))
# b=np.random.rand(5, 6, 7)
# print(b.shape)
# print(b.shape[0])
# print(b.size)

# a=torch.rand(2,3,4)
# b=a.view(-1)
# c=a.tolist()
# print(b)
# print(c)

# a=np.random.rand(2,3,4)
# #b=a.view(-1)
# c=a.tolist()
# #print(b)
# print(c)

# a=torch.randint(0,1, size=(2,2))
# b=a+2
# print(b)

# a=torch.randint(3,10, size=(2,3,3)).view(-1)
# b=torch.randint(0,2, size=(2,3,3)).bool().view(-1)
# print(a)
# print(b)
# c=torch.masked_select(a, b).view(2,3,-1)
# print(c)

# a=torch.randint(0,3, size=(3,3))
# b=(a != 0)
# print(b)
# print(b.sum())

# a=torch.tensor(0.)
# b=torch.tensor(0.)
# print(a/b)

# a=torch.rand(2,3)
# b=a.mean()
# print(b)
# print(b.mean())

a=torch.rand(2,3)
b=np.mean(np.asarray(a))
print(b)