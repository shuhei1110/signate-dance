import os
import sys
import pprint

from tqdm import tqdm
from pathlib import Path
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

import cv2
import numpy as np
import pandas as pd

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils import logger, image_processor
from configs import config

print('import ok')
torch.cuda.empty_cache()

logger = logger.Logger().logger
logger.info('Stert processing...')

config = config.Resnet50dGRU()

df_train_class = pd.read_csv(config.train_class)

train_paths = [config.train_dir + train_path for train_path in os.listdir(config.train_dir)]
test_paths = [config.test_dir + test_path for test_path in os.listdir(config.test_dir)]

class CustomDataset(Dataset):
    def __init__(self, dataframe, status='train'):
        self.dataframe = dataframe
        self.status = status

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.loc[idx]
        if self.status == 'train':
            video_path = config.train_dir + row.loc['file']
        elif self.status == 'valid':
            video_path = config.train_dir + row.loc['file']
        elif self.status == 'test':
            video_path = config.test_dir + row.loc['file']
        else:
            logger.error(f'The status enterd is not one of the expected ones: {self.status}')
            sys.exit()
        images = image_processor.load_image(video_path)
        images = torch.from_numpy(images)
        images = images.permute(0, 3, 2, 1)
        images = images[torch.linspace(0, images.size(0)-1, 64).long()]
        images = F.interpolate(images, size=(256, 128), mode='bilinear', align_corners=False)

        label = row.loc['class']

        return images, label
    

file_names = df_train_class['file'].to_list()
classes = df_train_class['class'].to_list()
train_files, valid_files, train_classes, valid_classes = train_test_split(file_names, classes, test_size=0.2, random_state=42, stratify=classes)
df_train = pd.DataFrame({'file': train_files, 'class': train_classes})
df_valid = pd.DataFrame({'file': valid_files, 'class': valid_classes})

train_dataset = CustomDataset(df_train, status='train')
valid_dataset = CustomDataset(df_valid, status='valid')

train_dataloader = DataLoader(train_dataset, 
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True
                              )
valid_dataloader = DataLoader(valid_dataset, 
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True
                              )

# for images, label in train_dataloader:
#     print(images.shape)
#     break

# sys.exit()

class ClassificationModel(nn.Module):
    def __init__(self):
       super().__init__()
       
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
           nn.Linear(32, 17, bias=True)
           )
       
    def forward(self, x):
        n_frames = x.shape[1]
        width = x.shape[3]
        height = x.shape[4]
        x = x.view(config.batch_size*n_frames, config.ch_size, width, height)
        x = self.encoder(x)
        x = x.view(config.batch_size, n_frames, -1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.layer(x)

        return x

class MetricsCalculater:
    def __init__(self, mode='binary'):
        self.probabilities = []
        self.predictions = []
        self.targets = []
        self.mode = mode

    def update(self, logits, target):
        if self.mode == 'binary':
            probabilities = torch.sigmoid(logits)
            predicted = (probabilities > 0.5)
        else:
            probabilities = F.softmax(logits, dim=1)
            predicted = torch.argmax(probabilities, dim=1)
        
        self.probabilities.extend(probabilities.detach().cpu().numpy())
        self.predictions.extend(predicted.detach().cpu().numpy())
        self.targets.extend(target.detach().cpu().numpy())

    def reset(self):
        self.probabilities = []
        self.predictions = []
        self.targets = []

    def compute_accuracy(self):
        return accuracy_score(self.targets, self.predictions)

    def compute_auc(self):
        processed_probabilities = []
        for x in self.probabilities:
            processed_probabilities.append(x / x.sum())
        if self.mode == 'multi':
            return roc_auc_score(self.targets, processed_probabilities, multi_class='ovo', labels=list(range(17)))
        else:
            return roc_auc_score(self.targets, self.probabilities)

# preparation for train
model = ClassificationModel().to(config.device)

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

train_acc = MetricsCalculater('multi')
valid_acc = MetricsCalculater('multi')

valid_loss_best = float('inf')

for epoch in range(config.epochs):
    logger.info(f'Epoch [{epoch + 1}/{config.epochs}]')

    #train
    model.train()
    train_loss = 0.0
    train_acc.reset()

    for batch_idx, (images, labels) in enumerate(tqdm(train_dataloader)):
        images = images.to(config.device).float()
        labels = labels.type(torch.LongTensor).to(config.device)
        
        optimizer.zero_grad()

        predicts = model(images)
        loss = criterion(predicts, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc.update(predicts, labels)

    train_loss /= (batch_idx+1)
    logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc.compute_accuracy()}')

    # validation
    model.eval()
    valid_loss = 0.0
    valid_acc.reset()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(valid_dataloader)):
            images = images.to(config.device).float()
            labels = labels.type(torch.LongTensor).to(config.device)

            predicts = model(images)
            loss = criterion(predicts, labels)

            valid_loss += loss.item()
            valid_acc.update(predicts, labels)
        
    valid_loss /= (batch_idx+1)
    logger.info(f'Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc.compute_accuracy()}')

    if valid_loss  < valid_loss_best:
        valid_loss_best = valid_loss
        logger.info('Validation loss improved')
        torch.save(model, f'{config.encoder}_gru_001_epoch{epoch}_{valid_acc:.4f}.pth')
        





