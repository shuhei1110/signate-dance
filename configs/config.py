import torch

class BaseConfig():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DataConfig():
    def __init__(self):
        self.root_dir = '/app/'
        self.train_dir = self.root_dir + 'data/train/'
        self.test_dir = self.root_dir + 'data/test/'
        self.train_class = self.root_dir + 'data/train_class.csv'
        self.test_submit = self.root_dir + 'data/test_submit.csv'
        self.sample_submit = self.root_dir + 'data/sample_submit.csv'
        self.output_dir = self.root_dir + 'data/output/'

class ModelConfig():
    def __init__(self):
        self.encoder = 'resnet50d'
        self.lr = 1e-4
        self.epochs = 3
        self.batch_size = 4
        self.ch_size = 3