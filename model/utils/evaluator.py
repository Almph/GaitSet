import torch
import torch.nn.functional as F
import numpy as np


def cuda_dist(x, y):
    #传入两个n*(62*256)的数组。
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
        1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    #这个跟三元组损失里的batch_dist计算方法相同。
    dist = torch.sqrt(F.relu(dist))
    #欧式距离，n*n的数组。
    return dist


def evaluation(data, config):
    #传进来的参数是所有测试数据以及附带的一堆属性；
    #config是原conf['data']。
    dataset = config['dataset'].split('-')[0]
    #'CASIA'。
    feature, view, seq_type, label = data
    #data的结构如此。
    label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    #长度11，有序。
    view_num = len(view_list)
    #11。
    sample_num = len(feature)
    #测试数据总量。

    probe_seq_dict = {'CASIA': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['00']]}
    gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']]}

    num_rank = 5
    #找出最相似的前五个数据。
    acc = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num, num_rank])
    #3*11*11*5的准确度记录数组。
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        #例如p=0, probe_seq=['nm-05, 'nm-06']。
        for gallery_seq in gallery_seq_dict[dataset]:
            #仅gallery_seq=['nm-01', 'nm-02', 'nm-03', 'nm-04']。
            for (v1, probe_view) in enumerate(view_list):
                #例如v1=10, probe_view='180'。
                for (v2, gallery_view) in enumerate(view_list):
                    #例如v2=0, gallery_view='000'。
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])
                    gallery_x = feature[gseq_mask, :]
                    #取出符合当前gallery_seq和gallery_view的所有数据。
                    #n*(62*256)的二维数组。
                    gallery_y = label[gseq_mask]
                    #取出上述数据对应的label。
                    #长度为n的向量。

                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                    probe_x = feature[pseq_mask, :]
                    #取出符合当前gallery_seq和gallery_view的所有数据。
                    #n*(62*256)的二维数组。
                    probe_y = label[pseq_mask]
                    #取出上述数据对应的label。
                    #长度为n的向量。

                    #上面的代码把测试数据进行了抽取，根据他们的序列类型分成了probe和gallery。

                    dist = cuda_dist(probe_x, gallery_x)
                    #传入两个n*(62*256)的数组。
                    idx = dist.sort(1)[1].cpu().numpy()
                    #这里用1是对gallery_x的轴进行排序。
                    #排序后的结果就是，对每个probe_x，离它的欧氏距离由近到远。
                    #tensor和nparray的.sort()方法不同，np会原地修改，无返回值，
                    # torch不修改原数组，返回排序后的结果，包括排序后的数组和排序后元素原下标的新位置。

                    acc[p, v1, v2, :] = np.round(
                        np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                               0) * 100 / dist.shape[0], 2)
                    #n*5 -> n*5 -> (,5) -> (,5)
                    #np.cumsum对每行五个布尔值做累计求和，求和结果，若第一项为真，则后四项全真，
                    # 即rank-5级别的要求下准确，同时rank-1级别下准确。
                    # 同理，前两项为假而第三项为真，则后三项全真，只能在rank-3级别下准确。
                    # > 0将cumsum的结果转为布尔值，
                    #np.sum将当前probe_seq，probe_view和gallery_view下的probe数据检测准确与否求和
                    # / dist.shape[0] 求准确率，* 100变成百分数形式，np.round()保留两位小数。
                    #这个五维向量的元素依次表示rank-1到rank-5的准确率。

    return acc
