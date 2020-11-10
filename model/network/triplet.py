import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, batch_size, hard_or_full, margin):
        super(TripletLoss, self).__init__()
        self.batch_size = batch_size
        #这里传进来的是128，是batch_size的两个元素相乘的结果。
        self.margin = margin
        #超参。

    def forward(self, feature, label):
        # feature: [n, m, d], label: [n, m]
        #传进来的triplet_feature是62*batch_size*256，triplet_label是62*batch_size。
        n, m, d = feature.size()
        #这里n=62，m=bs，d=256。

        # print('n:{0}, m:{1}, d:{2}'.format(n, m, d))#for test
        # print('label:', label)#for test

        hp_mask = (label.unsqueeze(1) == label.unsqueeze(2)).bool().view(-1)
        #hard positive
        hn_mask = (label.unsqueeze(1) != label.unsqueeze(2)).bool().view(-1)
        #hard negative
        #长度为62*batch_size*batch_size的向量。
        #原本是调用.byte()方法，转为torch.uint8。现在调用.bool()，转为布尔值。

        # print('hp_mask:', len(hp_mask[hp_mask==1]))#for test
        # print('hn_mask:', len(hn_mask[hn_mask==1]))#for test

        dist = self.batch_dist(feature)
        #62*batch_size*batch_size。
        mean_dist = dist.mean(1).mean(1)
        #长度为62的向量，整个batch数据的平均距离。
        dist = dist.view(-1)
        #长度为62*batch_size*batch_size的向量。
        
        # hard
        hard_hp_dist = torch.max(torch.masked_select(dist, hp_mask).view(n, m, -1), 2)[0]
        hard_hn_dist = torch.min(torch.masked_select(dist, hn_mask).view(n, m, -1), 2)[0]
        #形状都是62*batch_size。
        #hard_hp_dist记录着每个类别类内距离的最大值，分为62个scale。
        #hard_hn_dist记录着每个类别和别的几个类类间距离的最小值，分为62个scale。

        #这里是怎么确保经过mask后的dist长度正好可以被n*m整除的呢？（n=62, m=bs）
        #经过一系列测试代码和观察，能保证按hp_mask或hn_mask取值后可以被n*m整除是因为采样用了三元组采样的原因。
        #考虑将mask拉成向量前的形状62*bs*bs：
        # 首先被n=62整除是自然的，因为62个scale上label信息完全相同，hp_mask[0]到hp_mask[61]每一片长得一模一样；
        # 而被m整除是三元组采样的原因，这保证了每次label数量都固定为bs[0]，那么
        # hp_mask[0]的元素总数为bs*bs，其中为真的元素总数为bs[1]*bs[1]*bs[0]，因为每个人只和自己的label相同；
        # hn_mask[0]的元素总数为bs*bs，其中为真的元素总数为bs[1]*bs[1]*bs[0]*(bs[0]-1)，因为每个人都和除自己以外的人label不同；
        # 上面的两个总数都可以被bs=bs[0]*bs[1]整除。
        #所以立即推，取.view(n, m, -1)之前，
        # torch.masked_select(dist, hp_mask)的长度为62*bs[1]*bs[1]*bs[0]；
        # torch.masked_select(dist, hn_mask)的长度为62*bs[1]*bs[1]*bs[0]*(bs[0]-1)；
        #取.view(n, m, -1)之后：
        # torch.masked_select(dist, hp_mask).view(n, m, -1)的形状为62*bs*bs[1]；
        # torch.masked_select(dist, hp_mask).view(n, m, -1)的形状为62*bs*(bs[1]*(bs[0]-1))。

        # print('hard_hp_dist:', hard_hp_dist.shape)#for test
        # print('hard_hn_dist:', hard_hn_dist.shape)#for test
        
        hard_loss_metric = F.relu(self.margin + hard_hp_dist - hard_hn_dist).view(n, -1)
        #形状仍为62*batch_size。
        #记录着bs个类别的硬三元组损失，分为62个scale。

        hard_loss_metric_mean = torch.mean(hard_loss_metric, 1)
        #是长度为62的向量。
        #当前batch内的平均硬三元组损失，分为62个scale。

        # non-zero full
        full_hp_dist = torch.masked_select(dist, hp_mask).view(n, m, -1, 1)
        full_hn_dist = torch.masked_select(dist, hn_mask).view(n, m, 1, -1)
        #full_hp_dist的形状为62*bs*bs[1]*1。
        #full_hn_dist的形状为62*bs*1*(bs[1]*(bs[0]-1))。

        full_loss_metric = F.relu(self.margin + full_hp_dist - full_hn_dist).view(n, -1)
        #从62*bs*bs[1]*(bs[1]*(bs[0]-1)) -> 62*~
        #bs个数据，每个数据有和bs[1]个类内数据的距离，有和(bs[0]-1)*bs[1]个数据的类间距离，
        #按照三元组损失的定义，有bs个锚点，每个锚点有bs[1]个正样本对，有(bs[0]-1)*bs[1]个负样本对，
        # 而每个锚点的三元组损失就是上面的正负样本对两两组合计算得到，所以对每个锚点，它的正负样本对组合后的
        # 结果记录在一个bs[1]*(bs[1]*(bs[0]-1))的二维数组里，加上margin，relu后再求和。
        #而一共的正负样本对的个数，是锚点数*每个锚点对应的正负样本数，即bs*bs[1]*(bs[1]*(bs[0]-1))。
        #矩阵增广后做差得到三元组损失。

        full_loss_metric_sum = full_loss_metric.sum(1)
        #沿锚点数量作和，得到形状62*bs[1]*(bs[1]*(bs[0]-1))。
        full_loss_num = (full_loss_metric != 0).sum(1).float()
        #bool值当做0、1处理，n个True的和为n。
        #形状为62*bs[1]*(bs[1]*(bs[0]-1))。
        #记录着full_loss_metric里loss不为零的个数。
        #后面可能会出现除零操作，所以要设为float。

        full_loss_metric_mean = full_loss_metric_sum / full_loss_num
        #62*bs[1]*(bs[1]*(bs[0]-1))。
        full_loss_metric_mean[full_loss_num == 0] = 0
        #torch里用tensor除零的得数为inf或nan，这里改为0。

        return full_loss_metric_mean, hard_loss_metric_mean, mean_dist, full_loss_num

    def batch_dist(self, x):
        #传进来的feature是62*batch_size*256。
        #62是scale，把每个数据看成一个256维向量，即可知dist计算的是一个batch里数据之间的距离。
        x2 = torch.sum(x ** 2, 2)
        #尺寸是62*batch_size。
        dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(1, 2) - 2 * torch.matmul(x, x.transpose(1, 2))
        #62*batch_size*batch_size。
        dist = torch.sqrt(F.relu(dist))
        return dist
