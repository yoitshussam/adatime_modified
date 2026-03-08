def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    
    # This maps your new dataset names to the existing config class
    # this currently remaps any dataset name to "ALL" config because there aren't other configs
    dataset_name = "ALL" 

    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]        
        

class ALL():
    def __init__(self):
        super(ALL, self).__init__()
        # data parameters
        self.num_classes = 5
        self.class_names = ['sitting','standing','lying','running','walking']
        self.sequence_len = 150

        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 18
        self.stride = 1
        self.dropout = 0.1
        # self.kernel_size=5 is now defined in cnn_blocks
        self.cnn_blocks = [
            {'kernel_size': 24, 'maxpool': False, 'dropout': True},
            {'kernel_size': 8, 'maxpool': False, 'dropout': False},
            {'kernel_size': 8, 'maxpool': False, 'dropout': False},
        ]

        # features
        self.mid_channels = 64
        # self.final_out_channels = 128
        self.final_out_channels = 96
        self.features_len = 1

        # TCN features
        self.tcn_layers = [32,64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 15# 25
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.disc_hid_dim = 100

        # ACON-specific
        self.period = 64
        self.avg_mode = 10
        self.fft_normalize = False

        # RAINCOAT-specific
        self.fourier_modes = 64
        self.kernel_size = 5
        # out_dim = fourier_modes*2 (freq amp+phase) + final_out_channels*features_len (temporal)
        self.out_dim = self.fourier_modes * 2 + self.final_out_channels * self.features_len

        # SSSS_TSA-specific
        self.temp = 5.0  # temperature for channel attention softmax

        # CLUDA-specific
        self.cluda_K = 4096             # queue size (original 24576, reduced for smaller datasets)
        self.cluda_m = 0.999            # momentum for key encoder
        self.cluda_T = 0.07             # contrastive temperature
        self.cluda_num_neighbors = 1    # nearest neighbours for cross-domain CL
        self.cluda_mlp_hidden_dim = 256 # hidden dim for projector / discriminator
        self.cluda_use_batch_norm = True


# class ALL():
#     def __init__(self):
#         super(ALL, self).__init__()
#         # data parameters
#         self.num_classes = 5
#         self.class_names = ['sitting','standing','lying','running','walking']
#         self.sequence_len = 150

#         self.shuffle = True
#         self.drop_last = True
#         self.normalize = True

#         # model configs
#         self.input_channels = 18
#         self.kernel_size = 5
#         self.stride = 1
#         self.dropout = 0.1

#         # features
#         self.mid_channels = 64
#         # self.final_out_channels = 128
#         self.final_out_channels = 96
#         self.features_len = 1

#         # TCN features
#         self.tcn_layers = [32,64]
#         self.tcn_final_out_channles = self.tcn_layers[-1]
#         self.tcn_kernel_size = 15# 25
#         self.tcn_dropout = 0.0

#         # lstm features
#         self.lstm_hid = 128
#         self.lstm_n_layers = 1
#         self.lstm_bid = False

#         # discriminator
#         self.DSKN_disc_hid = 128
#         self.hidden_dim = 500
#         self.disc_hid_dim = 100

