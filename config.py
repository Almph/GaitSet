conf = {
    "WORK_PATH": "./work",
    "CUDA_VISIBLE_DEVICES": "0,1",
    "data": {
        'dataset_path': "/mnt/pami14/DATASET/GAIT/GaitAligned/64/CASIA-B/silhouettes/",
        'resolution': '64',
        'dataset': 'CASIA-B',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_num': 73,
        #ST: 24
        #MT: 62
        #LT: 73
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
        'total_iter': 2500,
        #CASIA-B为80000。
        'margin': 0.2,
        'num_workers': 3,
        'frame_num': 30,
        #每个数据中取出多少帧进行训练。
        'model_name': 'GaitSet',
    },
}
#这个配置只适合读取图片格式，pk格式的配置打算写到另一个分支，因为要改动data_set。
