import math
import os
import os.path as osp
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata

from .network import TripletLoss
from .utils import TripletSampler

class Model:
    def __init__(self,
                 hidden_dim,
                 lr,
                 hard_or_full_trip,
                 margin,
                 num_workers,
                 batch_size,
                 restore_iter,
                 total_iter,
                 save_name,
                 train_pid_num,
                 frame_num,
                 model_name,
                 train_source,
                 test_source,
                 img_size=64):

        if 'OUMVLP' in save_name:
            from .OUMVLP_network import SetNet
            print('You use OUMVLP_network!')
            #OUMVLP用这个作为encoder导入。
        else:
            from .network import SetNet
            print('You use CASIA-B_network!')
            #CASIA-B用这个作为encoder导入。

        self.save_name = save_name
        self.train_pid_num = train_pid_num
        self.train_source = train_source
        self.test_source = test_source

        self.hidden_dim = hidden_dim
        #线性全连接层的隐藏层数量。
        self.lr = lr
        self.hard_or_full_trip = hard_or_full_trip
        #三元组损失的计算方法，trip是tripletloss的缩写。
        self.margin = margin
        #三元组损失里的超参。
        self.frame_num = frame_num
        #每个数据里抽几帧进行训练。
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.model_name = model_name
        self.P, self.M = batch_size
        #人数和每个人抽几个数据。

        self.restore_iter = restore_iter
        self.total_iter = total_iter
        #这两个变量尚不清楚。

        self.img_size = img_size
        #这里用了默认的参数64。

        self.encoder = SetNet(self.hidden_dim).float()
        #提取特征的网络。
        self.encoder = nn.DataParallel(self.encoder)
        self.triplet_loss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()
        #第一个参数给出一个batch有多少个数据，第二个参数告诉三元组损失类型，第三个参数是计算loss时的超参。
        self.triplet_loss = nn.DataParallel(self.triplet_loss)
        self.encoder.cuda()
        self.triplet_loss.cuda()
        #先并行，然后放到显卡上。

        self.optimizer = optim.Adam([
            {'params': self.encoder.parameters()},
        ], lr=self.lr)
        #优化器的优化目标是整个网络的参数。

        self.hard_loss_metric = []
        self.full_loss_metric = []
        self.full_loss_num = []
        self.dist_list = []
        #记录loss和dist的列表。

        self.mean_dist = 0.01
        #平均距离？

        self.sample_type = 'all'
        #这里是默认为all，
        #训练时设为'random'。
        #测试时在.transform()方法里又设为'all'。

    def collate_fn(self, batch):
        batch_size = len(batch)
        #8*16=128个数据。
        feature_num = len(batch[0][0])
        #len(data)，值是1，data是[xarray]，
        #就看data里面存着几个特征，这里只有一个silhouette特征。
        seqs = [batch[i][0] for i in range(batch_size)]
        #五维列表，元素是data（[xarray]）。
        frame_sets = [batch[i][1] for i in range(batch_size)]
        #二维列表，元素是frame_set=['001', '002', ..., '']，长度不定。
        view = [batch[i][2] for i in range(batch_size)]
        #一维列表，记录当前batch数据的view。
        seq_type = [batch[i][3] for i in range(batch_size)]
        #同上，记录seq_type。
        label = [batch[i][4] for i in range(batch_size)]
        #同上，记录label。
        batch = [seqs, view, seq_type, label, None]

        def select_frame(index):
            sample = seqs[index]
            #取出一个特定的数据，sample是四维的。
            frame_set = frame_sets[index]
            #取出该数据的帧数。
            if self.sample_type == 'random':
                frame_id_list = random.choices(frame_set, k=self.frame_num)
                #从给定的数据中抽出30帧。
                _ = [feature.loc[frame_id_list].values for feature in sample]
                #这里的feature是个xarray。
            else:
                _ = [feature.values for feature in sample]
                #如果不是随机取帧，如前默认为'all'，则取所有的帧。
                #.values方法仅取出值。
                #_是个四维列表，列表壳子里面是一个三维nparray。
            return _

        seqs = list(map(select_frame, range(len(seqs))))
        #seqs的内容物从xarray变成了nparray，仍然是五维的。
        #len(seqs)*1*frame_num*64*44。

        if self.sample_type == 'random':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
            #seqs仍为五维，列表壳子里装着一个nparray，这个nparray是四维的。
            #1*batch_size*frame_num*64*44，第一个1是因为j只能为0。
        else:
            #采样模式为'all'时。
            gpu_num = min(torch.cuda.device_count(), batch_size)
            #事实上这里只在测试时发生，而测试时bs=1。
            batch_per_gpu = math.ceil(batch_size / gpu_num)
            #每个gpu上的batch数量。
            batch_frames = [[
                                len(frame_sets[i])
                                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                if i < batch_size
                                ] for _ in range(gpu_num)]
            #[[45, 56, ..., 64], [], ..., []]
            #长度为gpu数量，每个元素是一个列表，长度为batch_per_gpu，存着一张gpu上所有数据的帧数。

            if len(batch_frames[-1]) != batch_per_gpu:
                #即gpu_num无法整除batch_size时，最后一张gpu上的batch数量会比batch_per_gpu少。
                for _ in range(batch_per_gpu - len(batch_frames[-1])):
                    batch_frames[-1].append(0)
                    #把帧数用0补上，让最后一张GPU上数据的数量看起来也是batch_per_gpu。

            max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])
            #哪张GPU上要跑的数据的帧数最多呢？

            seqs = [[
                        np.concatenate([
                                           seqs[i][j]
                                           for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                           if i < batch_size
                                           ], 0) for _ in range(gpu_num)]
                    for j in range(feature_num)]
            #拼接操作会把列表壳子吃掉，返回(batch_per_gpu个frame_num的和)*64*44的三维输出。
            #1*gpu_num*(batch_per_gpu个frame_num的和)*64*44。

            seqs = [np.asarray([
                                   np.pad(seqs[j][_],
                                          ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                                          'constant',
                                          constant_values=0)
                                   for _ in range(gpu_num)])
                    for j in range(feature_num)]
            #1*gpu_num*max_sum_frame*64*44。
            #(batch_per_gpu个frame_num的和)不足max_sum_frame的用全是0的帧补齐。

            batch[4] = np.asarray(batch_frames)
            #给之前预设为None的位置赋值。

        batch[0] = seqs
        #采样模式如为random，seqs是1*batch_size*frame_num*64*44。
        #采样模式如为all，seqs是1*gpu_num*max_sum_frame*64*44。
        return batch

    def fit(self):
        if self.restore_iter != 0:
            self.load(self.restore_iter)
            #加载对应的checkpoint。
            #加载网络参数和优化器参数。

        self.encoder.train()
        self.sample_type = 'random'
        #训练时采样方法强制定为random。
        #这个参数不允许从外部修改，所以是指定使用random采样。

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
            #param_group是一个字典，带有键'lr'。该循环只执行一次。

        triplet_sampler = TripletSampler(self.train_source, self.batch_size)
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_sampler=triplet_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        train_label_set = list(self.train_source.label_set)
        train_label_set.sort()
        #训练数据集的label列表。

        _time1 = datetime.now()
        #获取当前系统时间。

        for seq, view, seq_type, label, batch_frame in train_loader:
            #按如前所述，sample_type='random'，
            # 在collate_fn里seq是1*batch_size*frame_num*64*44的格式。
            self.restore_iter += 1
            self.optimizer.zero_grad()

            for i in range(len(seq)):
                seq[i] = self.np2var(seq[i]).float()
                #nparray->tensor->Variable。
                #新版本pytorch中不再鼓励使用Variable，因为tensor内置了require_grad等属性。

            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()
                #这里只有在sample_type='all'时才生效。

            feature, label_prob = self.encoder(*seq, batch_frame)
            #seq被解包后为batch_size*frame_num*64*44的格式。
            #batch_frame为None。
            #feature形状为batch_size*62*256。
            #label_prob是None。

            target_label = [train_label_set.index(l) for l in label]
            #得出当前batch数据的label在整个训练集label的位置下标。
            #长度为batch_size=128，值从0-71（005没有姓名）。
            target_label = self.np2var(np.array(target_label)).long()
            #同样转换为Variable。

            triplet_feature = feature.permute(1, 0, 2).contiguous()
            #转置后形状为62*batch_size*256，
            # 转置后在内存中连续化。

            triplet_label = target_label.unsqueeze(0).repeat(triplet_feature.size(0), 1)
            #62*128（batch_size）。

            (full_loss_metric, hard_loss_metric, mean_dist, full_loss_num
             ) = self.triplet_loss(triplet_feature, triplet_label)
            #传进去的triplet_feature是62*batch_size*256，triplet_label是62*batch_size。
            #返回值的形状依次为62*bs[1]*(bs[1]*(bs[0]-1))，其实是平均值full_loss_metric_mean，
            # 62*batch_size，其实是平均值hard_loss_metric_mean，
            # 62维的向量，和
            # 62*bs[1]*(bs[1]*(bs[0]-1))。

            if self.hard_or_full_trip == 'hard':
                loss = hard_loss_metric.mean()
            elif self.hard_or_full_trip == 'full':
                loss = full_loss_metric.mean()

            self.hard_loss_metric.append(hard_loss_metric.mean().data.cpu().numpy())
            self.full_loss_metric.append(full_loss_metric.mean().data.cpu().numpy())
            self.full_loss_num.append(full_loss_num.mean().data.cpu().numpy())
            self.dist_list.append(mean_dist.mean().data.cpu().numpy())
            #最后存的都是单个标量的平均数。

            if loss > 1e-9:
                #当损失低于或等于1e-9时，将不进行BP。
                loss.backward()
                self.optimizer.step()
            
            if self.restore_iter == 150000:
                self.optimizer.param_groups[0]['lr'] = 1e-5
                #这里依据论文针对OUMVLP设置，CASIA-B跑不了这么多次迭代。

            if self.restore_iter % 1000 == 0:
                print(datetime.now() - _time1)
                _time1 = datetime.now()
                #每1000个iter报告一次所用时间。

            if self.restore_iter % 100 == 0:
                self.save()
                #保存的模型参数和优化器参数。
                print('iter {}:'.format(self.restore_iter), end='')
                print(', hard_loss_metric={0:.8f}'.format(np.mean(self.hard_loss_metric)), end='')
                print(', full_loss_metric={0:.8f}'.format(np.mean(self.full_loss_metric)), end='')
                print(', full_loss_num={0:.8f}'.format(np.mean(self.full_loss_num)), end='')
                self.mean_dist = np.mean(self.dist_list)
                print(', mean_dist={0:.8f}'.format(self.mean_dist), end='')
                print(', lr=%f' % self.optimizer.param_groups[0]['lr'], end='')
                print(', hard or full=%r' % self.hard_or_full_trip)
                #上面打成一句话，以空格分离。
                #取100个iter里平均距离的平均值。
                sys.stdout.flush()
                #实际上没有清空输出屏幕。

                self.hard_loss_metric = []
                self.full_loss_metric = []
                self.full_loss_num = []
                self.dist_list = []
                #清空前一百个iter存储的信息。

            # Visualization using t-SNE
            # if self.restore_iter % 500 == 0:
            #     pca = TSNE(2)
            #     pca_feature = pca.fit_transform(feature.view(feature.size(0), -1).data.cpu().numpy())
            #     for i in range(self.P):
            #         plt.scatter(pca_feature[self.M * i:self.M * (i + 1), 0],
            #                     pca_feature[self.M * i:self.M * (i + 1), 1], label=label[self.M * i])
            #
            #     plt.show()

            if self.restore_iter == self.total_iter:
                break

    def ts2var(self, x):
        return autograd.Variable(x).cuda()

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))

    def transform(self, flag, batch_size=1):
        self.encoder.eval()
        #显然，这个方法只在测试时调用。
        #传入的参数依次是字符串'flag'，整数1。

        source = self.test_source if flag == 'test' else self.train_source
        #该参数是一个数据集。

        self.sample_type = 'all'
        #这里又强制限定采样方法为'all'。

        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)
        #数据集用测试集，使用顺序采样，bs为1，分配函数使用自定函数。
        #注意采样器是需要输入具体的数据集的，用来调用该数据集的.__len__()方法。

        feature_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()

        for i, x in enumerate(data_loader):
            #一次只获取一个的数据，共有test_label*11*10个数据。
            seq, view, seq_type, label, batch_frame = x
            #seqs是1*gpu_num*max_sum_frame*64*44。
            #因为sample_type变为'all'，这里batch_frame不为None。
            #batch_frame长度为gpu数量，每个元素是一个列表，长度为batch_per_gpu，存着一张gpu上所有数据的帧数。
            ##[[45, 56, ..., 64], [], ..., []]，其实是一个nparray。
            #当最后一张gpu上的数据数量不够一个batch时，batch_frame[-1]长度仍为batch_per_gpu，缺的数据用帧数为0补齐。
            #可以参考self.collate_fn里的注释。
            #因为设定了batch_size=1，所以gpu_num=1,max_sum_frame=frame_num，
            #但注意，这里的frame_num是当前数据的所有帧数，而非手动指定的30（config里的）。

            for j in range(len(seq)):
                seq[j] = self.np2var(seq[j]).float()
                #seq长度为1。
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()
            #上面两个操作将data_loader里取出的内容从nparray转为Variable。
            #print(batch_frame, np.sum(batch_frame))。

            feature, _ = self.encoder(*seq, batch_frame)
            #bs*62*256，其中bs=1。
            n, num_bin, _ = feature.size()

            feature_list.append(feature.view(n, -1).data.cpu().numpy())
            #把提出来的特征每个数据拉成一条向量并存入这个列表。
            view_list += view
            seq_type_list += seq_type
            label_list += label
            #这里的view，seq_type和label都是存着一个元素的列表，对应这当前数据的view等属性。

            # print('view:', view)#for test
            # print('seq_type:', seq_type)#for test
            # print('label:', label)#for test

        return np.concatenate(feature_list, 0), view_list, seq_type_list, label_list

    def save(self):
        os.makedirs(osp.join('checkpoint', self.model_name), exist_ok=True)
        torch.save(self.encoder.state_dict(),
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-encoder.ptm'.format(
                                self.save_name, self.restore_iter)))
        torch.save(self.optimizer.state_dict(),
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-optimizer.ptm'.format(
                                self.save_name, self.restore_iter)))

    # restore_iter: iteration index of the checkpoint to load
    def load(self, restore_iter):
        self.encoder.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-encoder.ptm'.format(self.save_name, restore_iter))))
        self.optimizer.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-optimizer.ptm'.format(self.save_name, restore_iter))))
