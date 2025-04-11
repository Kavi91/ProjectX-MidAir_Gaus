import os

class Parameters():
    def __init__(self, batch_size=24):
        self.n_processors = 24
        # Path settings
        self.data_dir = '/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed'
        self.image_dir = self.data_dir
        self.depth_dir = self.data_dir
        self.lidar_dir = self.data_dir
        self.lidar_subdir = 'depth_velo'
        self.pose_dir = self.data_dir
        
        # Mid-Air climate sets
        self.climate_sets = [
            'PLE_training/fall', 'PLE_training/spring', 'PLE_training/winter',
        ]
        
        self.train_traj_ids = {
            'PLE_training/spring':  ['trajectory_5000', 'trajectory_5001', 'trajectory_5002'],
            'PLE_training/fall':    ['trajectory_4000', 'trajectory_4001', 'trajectory_4002'],
            'PLE_training/winter':  ['trajectory_6000', 'trajectory_6001', 'trajectory_6002'],
        }

        self.valid_traj_ids = {
            'PLE_training/spring':  ['trajectory_5003'],
            'PLE_training/fall':    ['trajectory_4003'],
            'PLE_training/winter':  ['trajectory_6003'],
        }

        self.test_traj_ids = {
            'PLE_training/spring':  ['trajectory_5005', 'trajectory_5006'],
        }
        
        self.partition = None
        
        # Data Preprocessing
        self.resize_mode = 'rescale'
        self.img_w = 608
        self.img_h = 184

        self.img_means_03 = (0.5029287934303284, 0.49763786792755127, 0.3706325888633728)
        self.img_stds_03 = (0.255334734916687, 0.23501254618167877, 0.2485671192407608)
        self.img_means_02 = (0.502440869808197, 0.49729645252227783, 0.37063950300216675)
        self.img_stds_02 = (0.2558165192604065, 0.23529765009880066, 0.24865560233592987)
        self.depth_max = 100.0
        self.minus_point_5 = True

        self.seq_len = 5  # Fixed sequence length
        self.sample_times = 3
        self.overlap = 1

        # Batch size
        self.batch_size = batch_size

        # Data info path – filenames encode the sequence, sample, and batch settings.
        self.train_data_info_path = 'datainfo/train_df_midair_seq{}_sample{}_b{}.pickle'.format(
            self.seq_len, self.sample_times, self.batch_size)
        self.valid_data_info_path = 'datainfo/valid_df_midair_seq{}_sample{}_b{}.pickle'.format(
            self.seq_len, self.sample_times, self.batch_size)
        self.test_data_info_path = 'datainfo/test_df_midair_seq{}_sample{}_b{}.pickle'.format(
            self.seq_len, self.sample_times, self.batch_size)

        # Model settings
        self.rnn_hidden_size = 1000
        # Updated convolutional dropout values to 0.2 for less aggressive dropout
        self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
        self.rnn_dropout_out = 0.3  # Moderate dropout on RNN outputs
        self.rnn_dropout_between = 0
        self.clip = None
        self.batch_norm = True
        
        # Training settings
        self.epochs = 100
        self.pin_mem = True
        # Reduced weight decay as recommended and a modest learning rate.
        self.optim = {'opt': 'Adam', 'lr': 1e-4, 'weight_decay': 1e-5}
        
        # Modality flags – ensure these are used consistently across the code.
        self.enable_rgb = True
        self.enable_depth = True
        self.enable_lidar = False
        self.enable_imu = True
        self.enable_gps = True

        self.gps_loss_weight = 0.5
        self.l2_lambda = 0
        self.R_factor = 1
        self.T_factor = 1
        self.depth_gate_scaling = 20.0
        self.imu_gate_scaling = 15.0
        self.depth_consistency_loss_weight = 10.0

        # Pretrain and resume settings
        self.pretrained_flownet = '/home/krkavinda/DeepVO-pytorch/FlowNet_models/pytorch/flownets_bn_EPE2.459.pth'
        self.resume = False
        self.resume_t_or_v = '.train'

        # Paths for saving and loading models and records
        self.load_model_path = 'models/midair_im{}x{}_s{}_b{}_rnn{}_{}.model{}'.format(
            self.img_h, self.img_w, self.seq_len, self.batch_size, self.rnn_hidden_size,
            '_'.join([k + str(v) for k, v in self.optim.items()]), self.resume_t_or_v)
        self.load_optimizer_path = 'models/midair_im{}x{}_s{}_b{}_rnn{}_{}.optimizer{}'.format(
            self.img_h, self.img_w, self.seq_len, self.batch_size, self.rnn_hidden_size,
            '_'.join([k + str(v) for k, v in self.optim.items()]), self.resume_t_or_v)

        self.record_path = 'records/midair_im{}x{}_s{}_b{}_rnn{}_{}.txt'.format(
            self.img_h, self.img_w, self.seq_len, self.batch_size, self.rnn_hidden_size,
            '_'.join([k + str(v) for k, v in self.optim.items()]))
        self.save_model_path = 'models/midair_im{}x{}_s{}_b{}_rnn{}_{}.model'.format(
            self.img_h, self.img_w, self.seq_len, self.batch_size, self.rnn_hidden_size,
            '_'.join([k + str(v) for k, v in self.optim.items()]))
        self.save_optimzer_path = 'models/midair_im{}x{}_s{}_b{}_rnn{}_{}.optimizer'.format(
            self.img_h, self.img_w, self.seq_len, self.batch_size, self.rnn_hidden_size,
            '_'.join([k + str(v) for k, v in self.optim.items()]))

        # Create needed directories if they do not exist.
        for path in [self.record_path, self.save_model_path, self.save_optimzer_path, self.train_data_info_path, self.test_data_info_path]:
            if not os.path.isdir(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

par = Parameters()
print('Parameters initialized.')
