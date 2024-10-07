import os
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
