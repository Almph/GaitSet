import os
import os.path as osp

import numpy as np

from .data_set import DataSet


def load_data(dataset_path, resolution, dataset, pid_num, pid_shuffle, cache=True):
    seq_dir = list()
    view = list()
    seq_type = list()
    label = list()

    for _label in sorted(list(os.listdir(dataset_path))):
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        if dataset == 'CASIA-B' and _label == '005':
            continue
        label_path = osp.join(dataset_path, _label)
        #诸如'your_data_path/002'这样的文件路径。

        for _seq_type in sorted(list(os.listdir(label_path))):
            #对每个人（label）下的10个类型遍历。
            #诸如''bg-02'的类型字符串。
            seq_type_path = osp.join(label_path, _seq_type)
            #诸如'your_data_path/002/bg-02'的文件路径字符串。
            for _view in sorted(list(os.listdir(seq_type_path))):
                #诸如'018'的角度字符串。
                _seq_dir = osp.join(seq_type_path, _view)
                #诸如'your_data_path/002/bg-02/018'的文件路径字符串。
                seqs = os.listdir(_seq_dir)
                #长度为序列长度，元素为诸如'001'的轮廓文件名。
                if len(seqs) > 0:
                    #如果这个序列不为空。
                    seq_dir.append([_seq_dir])
                    #[[], ..., ['your_data_path/002/bg-02/018'], ..., []]
                    label.append(_label)
                    #['', '002', ..., '']
                    seq_type.append(_seq_type)
                    #['', 'bg-02', ..., '']
                    view.append(_view)
                    #['', '018', ..., '']
                    #上述几个列表长度相同，都是label数量*type数量*view数量。

    pid_fname = osp.join('partition', '{}_{}_{}.npy'.format(
        dataset, pid_num, pid_shuffle))
    #'partition/CASIA-B_73_False.npy'

    if not osp.exists(pid_fname):
        pid_list = sorted(list(set(label)))
        #长度为123，元素为诸如'002'的label字符串。

        if pid_shuffle:
            np.random.shuffle(pid_list)
            #将这个列表是否打乱。
        
        pid_list = [pid_list[0:pid_num], pid_list[pid_num:]]
        #[['000', '001', ..., '073'], ['074', '075', ..., '124']]

        os.makedirs('partition', exist_ok=True)
        
        np.save(pid_fname, pid_list)

    pid_list = np.load(pid_fname)
    train_list = pid_list[0]
    test_list = pid_list[1]
    train_source = DataSet(
        [seq_dir[i] for i, l in enumerate(label) if l in train_list],
        [label[i] for i, l in enumerate(label) if l in train_list],
        [seq_type[i] for i, l in enumerate(label) if l in train_list],
        [view[i] for i, l in enumerate(label)
         if l in train_list],
        cache, resolution)
    test_source = DataSet(
        [seq_dir[i] for i, l in enumerate(label) if l in test_list],
        [label[i] for i, l in enumerate(label) if l in test_list],
        [seq_type[i] for i, l in enumerate(label) if l in test_list],
        [view[i] for i, l in enumerate(label)
         if l in test_list],
        cache, resolution)

    return train_source, test_source
