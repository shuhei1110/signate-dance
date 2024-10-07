import os
import timm
import torch.nn as nn
from configs import base_config

class Resnet50dGRU(base_config.BaseConfig):
    def __init__(self, mode='train'):
        super().__init__()
        self.encoder = 'resnet50d'
        self.lr = 1e-4
        self.epochs = 50
        self.batch_size = 4
        self.ch_size = 3
        self.reshaped_nframe = 32
        self.reshaped_width = 256
        self.reshaped_height = 128
        self.drop_rate = 0.0
        self.output_size = 17
        self.class_names = list(range(self.output_size))

        self.save_dir = self.model_dir + 'resnet50d_gru-001/'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.inference_model = self.save_dir + 'resnet50d_gru_001_epoch20_0.7725.pth'

class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
       
        config = Resnet50dGRU()
        self.encoder = timm.create_model(config.encoder, pretrained=True, in_chans=config.ch_size, num_classes=512)
        self.gru = nn.GRU(512, 128, batch_first=True, bidirectional=False, num_layers=2, dropout=config.drop_rate)
        self.layer = nn.Sequential(
            nn.Linear(128, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(config.drop_rate),
            nn.Linear(64, 32, bias=True),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(config.drop_rate),
            nn.Linear(32, config.output_size, bias=True)
            )
       
    def forward(self, x):
        config = Resnet50dGRU()
        batch_size = x.shape[0]
        n_frames = x.shape[1]
        width = x.shape[3]
        height = x.shape[4]
        x = x.view(batch_size*n_frames, config.ch_size, width, height)
        x = self.encoder(x)
        x = x.view(batch_size, n_frames, -1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.layer(x)

        return x