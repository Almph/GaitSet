conf = {
    "WORK_PATH": "./work",
    "CUDA_VISIBLE_DEVICES": "0,1,3,4",
    "data": {
        'dataset_path': "/mnt/pami14/DATASET/GAIT/GaitAligned/64/OUMVLP/silhouettes/",
        #这里把CASIA-B改成OUMVLP即可更换为另一个数据集。
        'resolution': '64',
        'dataset': 'OUMVLP',
        #'CASIA-B'或'OUMVLP'。
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_num': 5153,
        #CASIA-B：73
        #OUMVLP：5153
        'pid_shuffle': False,
    },
    "model": {
        'hidden_dim': 256,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        #默认为full。
        'batch_size': (2, 4),
        #默认为(8, 16)。
        'restore_iter': 0,
        #这个参数决定了是否加载checkpoint，以及加载第多少次iteration（类似episode）的checkpoint。
        'total_iter': 2500,
        #CASIA-B为80000。
        #OUMVLP为250000。
        'margin': 0.2,
        'num_workers': 3,
        'frame_num': 30,
        #每个数据中取出多少帧进行训练。
        'model_name': 'GaitSet',
    },
}
