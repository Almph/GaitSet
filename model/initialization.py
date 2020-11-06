# -*- coding: utf-8 -*-
# @Author  : admin
# @Time    : 2018/11/15
import os
from copy import deepcopy

import numpy as np

from .utils import load_data
from .model import Model


def initialize_data(config, train=False, test=False):
    #这里传进来的train=True。
    #
    print("Initializing data source...")
    train_source, test_source = load_data(**config['data'], cache=(train or test))
    #双**星号表示把字典解包为key=value的形式传参给函数，要求key和函数的形参名能对应起来。
    #这里cache必然是True，就是数据的缓存不管是训练还是测试都会做。
    #这里的cache是告诉后续的data_loader，数据已经缓存好存在DataSet（.data属性）里面了。
    if train:
        print("Loading training data...")
        train_source.load_all_data()
        #加载数据缓存好，这里其实与cache参数无关，不管真假都会把所有数据都加载一遍。

    if test:
        print("Loading test data...")
        test_source.load_all_data()
    
    print("Data initialization complete.")
    return train_source, test_source


def initialize_model(config, train_source, test_source):
    print("Initializing model...")
    data_config = config['data']
    model_config = config['model']
    model_param = deepcopy(model_config)
    #深拷贝。

    model_param['train_source'] = train_source
    model_param['test_source'] = test_source
    #直接把两个数据集当做模型参数传进去对模型初始化。
    model_param['train_pid_num'] = data_config['pid_num']
    #这是新加入的一个键值对。

    batch_size = int(np.prod(model_config['batch_size']))
    #用于计算batch_size[0]*batch_size[1]的值，128。

    model_param['save_name'] = '_'.join(map(str,[
        model_config['model_name'],
        data_config['dataset'],
        data_config['pid_num'],
        data_config['pid_shuffle'],
        model_config['hidden_dim'],
        model_config['margin'],
        batch_size,
        model_config['hard_or_full_trip'],
        model_config['frame_num'],
    ]))

    m = Model(**model_param)
    #将设置好的参数传入对模型初始化。

    print("Model initialization complete.")
    return m, model_param['save_name']
    #返回模型和模型名称。


def initialization(config, train=False, test=False):
    #这里传进来的train为True。
    print("Initialzing...")
    WORK_PATH = config['WORK_PATH']
    os.chdir(WORK_PATH)
    #切换到工作目录。
    #也就是存放checkpoint的目录。
    os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]
    #环境的显卡数量。
    train_source, test_source = initialize_data(config, train, test)
    #data按照train=True进行初始化。
    #两个数据集的.data方法都存好了所有需要的数据。
    return initialize_model(config, train_source, test_source)