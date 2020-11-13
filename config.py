conf = {
    #当前版本加载的数据集从pk文件load，所以路径会不同。
    "WORK_PATH": "./work",
    "CUDA_VISIBLE_DEVICES": "4,5",
    "data": {
        'dataset_path': "/mnt/pami14/DATASET/GAIT/GaitAligned/64/OUMVLP/silhouettes/",
        #CASIA-B: "/mnt/pami14/DATASET/GAIT/GaitAligned/64/CASIA-loose/silhouettes/"
        #OUMVLP: "/mnt/pami14/DATASET/GAIT/GaitAligned/64/OUMVLP/silhouettes/"
        'resolution': '64',
        'dataset': 'OUMVLP',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_num': 5153,
        #ST: 24
        #MT: 62
        #LT: 73
        #OUMVLP: 5153
        'pid_shuffle': False,
    },
    "model": {
        'hidden_dim': 256,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        #默认为full。
        'batch_size': (4, 8),
        #默认为(8, 16)。
        'restore_iter': 0,
        #这个参数决定了是否加载checkpoint，以及加载第多少次iteration（类似episode）的checkpoint。
        'total_iter': 1000,
        #CASIA-B: 80000。
        #OUMVLP: 250000。
        'margin': 0.2,
        'num_workers': 3,
        'frame_num': 30,
        #每个数据中取出多少帧进行训练。
        'model_name': 'GaitSet',
    },
}
#这个版本改动了data_set.__loader__()，用于读取pk文件。
