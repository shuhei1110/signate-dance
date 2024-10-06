import torch

class BaseConfig():
    def __init__(self):
        self.root_dir = '/app/'
        self.train_dir = self.root_dir + 'data/train/'
        self.test_dir = self.root_dir + 'data/test/'
        self.train_class = self.root_dir + 'data/train_class.csv'
        self.test_submit = self.root_dir + 'data/test_submit.csv'
        self.sample_submit = self.root_dir + 'data/sample_submit.csv'
        self.output_dir = self.root_dir + 'data/output/'
        self.model_dir = self.root_dir + 'models/'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Resnet50dGRU(BaseConfig):
    def __init__(self):
        super().__init__()
        self.encoder = 'resnet50d'
        self.lr = 1e-4
        self.epochs = 30
        self.batch_size = 4
        self.ch_size = 3
        self.drop_rate = 0.5

        self.save_dir = self.model_dir + 'resnet50d_gru-001/'