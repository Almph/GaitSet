# -*- coding: utf-8 -*-
# @Author  : admin
# @Time    : 2018/11/15
import os
from copy import deepcopy

import numpy as np

from .utils import load_data
from .model import Model


def initialize_data(config, train=False, test=False):
    #训练和测试两个布尔值必然一真一假。除非手动指定--cache=False。
    print("Initializing data source...")
    train_source, test_source = load_data(**config['data'], cache=(train or test))
    #双**星号表示把字典解包为key=value的形式传参给函数，要求key和函数的形参名能对应起来。
    #因为训练和测试必然一真一假，这里cache必然是True。
    #这里的cache可以告诉后续调用数据集类的.__getitem()方法的data_loader，
    # 数据已经缓存好存在DataSet（.data属性）里面了，不用再从硬盘里读取第二次。

    if train:
        print("Loading training data...")
        train_source.load_all_data()
        

    if test:
        print("Loading test data...")
        test_source.load_all_data()
    #调用方法.load_all_data()后所有数据被存入.data属性里（是一个五维大列表）。
    #[[xrarray(nparray(64*44))], [], ..., []]。
    #训练时，把所有训练数据预先加载进数据集train_source.data（内存），
    #测试时，把所有测试数据预先加载进数据集test_source.data（内存）。
    
    if not (train or test):
        print("Skip data caching!")

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
    #这里计算的乘积仅用于存储模型名称。

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
    #返回模型和模型的保存名称。


def initialization(config, train=False, test=False):
    #这里传进来的train和test必有一个为True。
    #train和test后续传入cache=(train or test)参数，声明数据集是否进行了缓存（一定会缓存）。

    print("Initialzing...")
    WORK_PATH = config['WORK_PATH']
    os.chdir(WORK_PATH)
    #切换到工作目录。
    #也就是存放checkpoint的目录。
    os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]
    #环境的显卡数量。
    train_source, test_source = initialize_data(config, train, test)
    #数据集按照cache=True进行初始化。
    #训练时train_source.data存好了所有需要的数据。
    #测试时test_source.data存好了所有需要的数据。
    return initialize_model(config, train_source, test_source)