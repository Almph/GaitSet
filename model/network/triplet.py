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
        hp_mask = (label.unsqueeze(1) == label.unsqueeze(2)).byte().view(-1)
        #hard positive
        hn_mask = (label.unsqueeze(1) != label.unsqueeze(2)).byte().view(-1)
        #hard negative
        #长度为62*batch_size*batch_size的向量。

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
        #这里是怎么确保经过mask后的dist长度正好可以被n*m整除的呢？
        hard_loss_metric = F.relu(self.margin + hard_hp_dist - hard_hn_dist).view(n, -1)
        #形状仍为62*batch_size。

        hard_loss_metric_mean = torch.mean(hard_loss_metric, 1)
        #是长度为62的向量。

        # non-zero full
        full_hp_dist = torch.masked_select(dist, hp_mask).view(n, m, -1, 1)
        full_hn_dist = torch.masked_select(dist, hn_mask).view(n, m, 1, -1)
        full_loss_metric = F.relu(self.margin + full_hp_dist - full_hn_dist).view(n, -1)

        full_loss_metric_sum = full_loss_metric.sum(1)
        full_loss_num = (full_loss_metric != 0).sum(1).float()

        full_loss_metric_mean = full_loss_metric_sum / full_loss_num
        full_loss_metric_mean[full_loss_num == 0] = 0

        return full_loss_metric_mean, hard_loss_metric_mean, mean_dist, full_loss_num

    def batch_dist(self, x):
        #传进来的feature是62*batch_size*256。
        x2 = torch.sum(x ** 2, 2)
        #尺寸是62*batch_size。
        dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(1, 2) - 2 * torch.matmul(x, x.transpose(1, 2))
        #62*batch_size*batch_size。
        dist = torch.sqrt(F.relu(dist))
        return dist
