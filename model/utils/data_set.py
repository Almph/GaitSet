import torch.utils.data as tordata
import numpy as np
import os.path as osp
import os
import pickle
import cv2
import xarray as xr


class DataSet(tordata.Dataset):
    def __init__(self, seq_dir, label, seq_type, view, cache, resolution):
        self.seq_dir = seq_dir
        #每个数据所在的文件夹的列表。

        self.view = view
        self.seq_type = seq_type
        self.label = label
        #上面几个列表长度都是label数量*type数量*view数量。

        self.cache = cache
        #规定数据是否缓存，默认是True。

        self.resolution = int(resolution)
        #64，本来是字符串的64。
        self.cut_padding = int(float(resolution)/64*10)
        #图的两边切掉10像素宽度的黑条。

        self.data_size = len(self.label)
        #数据的数量，即一共有多少序列。

        self.data = [None] * self.data_size
        self.frame_set = [None] * self.data_size

        self.label_set = set(self.label)
        self.seq_type_set = set(self.seq_type)
        self.view_set = set(self.view)
        #长度依次为训练或测试人数、10和11。

        _ = np.zeros((len(self.label_set),
                      len(self.seq_type_set),
                      len(self.view_set))).astype('int')
        _ -= 1
        #用于记录所有训练或测试数据的下标。

        self.index_dict = xr.DataArray(
            _,
            coords={'label': sorted(list(self.label_set)),
                    'seq_type': sorted(list(self.seq_type_set)),
                    'view': sorted(list(self.view_set))},
            dims=['label', 'seq_type', 'view'])
        #给三个维度的坐标命名。

        for i in range(self.data_size):
            _label = self.label[i]
            _seq_type = self.seq_type[i]
            _view = self.view[i]
            self.index_dict.loc[_label, _seq_type, _view] = i
        #生成一个下标字典。
        print('DataSet define done.')#for test

    def load_all_data(self):
        for i in range(self.data_size):
            self.load_data(i)

    def load_data(self, index):
        return self.__getitem__(index)

    def __loader__(self, path):
        return self.img2xarray(
            path)[:, :, self.cut_padding:-self.cut_padding].astype(
            'float32') / 255.0
            #序列长度*64*44，
            #数据类型被转换为32位浮点数。

    def __getitem__(self, index):
        # pose sequence sampling
        #核心方法，torch的data_loader会自动调用。

        if not self.cache:
            #如果数据不缓存，但一般都会缓存。

            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            #这里_path形如'your_data_path/002/bg-02/018'。
            #data里封装了一个序列的数据：
            #[xrDataArray(nparray(64*44), nparray(64*44), ..., nparray(64*44))]
            #1*序列长度*64*44。

            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
            #['001', '002', ..., '']
            #这里面记录着当前序列每一帧的序号

        elif self.data[index] is None:
            #如果要求缓存且当前下标的数据还没有加载到数据集（的.data属性）里面。
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            #一个整数列表，包着一个集合，记录着帧数。

            frame_set = list(set.intersection(*frame_set))
            #一个整数列表，记录着帧数。

            self.data[index] = data
            #存入这个数据集。
            #self.data是一个五维的列表。
            #[[xrarray(nparray(64*44))], [], ..., []]。

            self.frame_set[index] = frame_set
            #某一数据对应的帧数列表。

        else:
            data = self.data[index]
            frame_set = self.frame_set[index]
            #如果已经缓存好，就直接读取。

        return data, frame_set, self.view[
            index], self.seq_type[index], self.label[index],

    def img2xarray(self, file_path):
        imgs = sorted(list(os.listdir(file_path)))
        #['000', '001', '002', ..., '']长度不定。

        #for test:
        for i in imgs:
            fi=osp.join(file_path, i)
            print('img_path:', fi)
            if osp.isfile(fi):
                cv2imread=cv2.imread(fi)
                print(cv2imread.shape)
        #end test.

        frame_list = [np.reshape(
            cv2.imread(osp.join(file_path, _img_path)),
            [self.resolution, self.resolution, -1])[:, :, 0]
                      for _img_path in imgs
                      if osp.isfile(osp.join(file_path, _img_path))]
        #[nparray(64*64), nparray(64*64), ..., nparray(64*64)]
        #每个nparray(64*64)是一帧。
        #cv2.imread读出来的图片是nparrary类型，形状是H*W*C，数据类型是int8。

        num_list = list(range(len(frame_list)))
        #当前序列的长度。
        #元素是整数。

        data_dict = xr.DataArray(
            frame_list,
            coords={'frame': num_list},
            dims=['frame', 'img_y', 'img_x'],
        )

        return data_dict

    def __len__(self):
        return len(self.label)
        #即self.data_size的值。
