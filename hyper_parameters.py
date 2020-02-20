import argparse


class HParams:
    def __init__(self):
        self.dataset_path = './gtzan'
        self.feature_path = './dataset/feature_argument'
        self.genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae']

        # parameters for feature engineering
        self.sample_rate = 22050
        self.fft_size = 1024
        self.win_size = 1024
        self.hop_size = 512
        self.num_mels = 128
        self.feature_length = 1024  # audio length = feature-length * hop_size / sample_rate (s)

        # training parameters
        self.device = 0  # 0. CPU, 1. GPU0 2. GPU2, ...
        self.batch_size = 4
        self.num_epochs = 10
        self.learning_rate = 1e-2


hparams = HParams()
