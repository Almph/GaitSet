import torch.utils.data as tordata
import random


class TripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        #传入诸如[8,16]的batch_size。

    def __iter__(self):
        while (True):
            #sampler是个死循环。
            #使得DataLoader可以无限调用。
            sample_indices = list()
            pid_list = random.sample(
                list(self.dataset.label_set),
                self.batch_size[0])
                #label_set中没有重复元素
                #每次抽出诸如8个不同的人。
            for pid in pid_list:
                _index = self.dataset.index_dict.loc[pid, :, :].values
                #取出这某个人所有的数据。
                _index = _index[_index > 0].flatten().tolist()
                #拉成列表。
                _index = random.choices(
                    _index,
                    k=self.batch_size[1])
                #从这个人的数据里不区分类型和角度地取出16个数据。
                sample_indices += _index
            yield sample_indices
            #包含8*16=128个数据。

    def __len__(self):
        return self.dataset.data_size
