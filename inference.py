import os
import sys
import pprint

from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import logger, image_processor
from configs.resnet50d_gru_001 import Resnet50dGRU, ClassificationModel # type: ignore

print('import ok')
torch.cuda.empty_cache()

logger = logger.Logger().logger
logger.info('Stert processing...')

config = Resnet50dGRU()

test_paths = [config.test_dir + test_path for test_path in os.listdir(config.test_dir)]

model = torch.load(config.inference_model)
model.eval()

results = []

for idx, test_path in enumerate(tqdm(test_paths)):
    images = image_processor.load_image(test_path)
    images = torch.from_numpy(images)
    # (n_frame, height, width, ch) -> (n_frame, ch, width, height)
    images = images.permute(0, 3, 2, 1)
    images = images[torch.linspace(0, images.size(0)-1, config.reshaped_nframe).long()]
    images = F.interpolate(images, size=(config.reshaped_width, config.reshaped_height), mode='bilinear', align_corners=False)
    images = images.unsqueeze(0)
    images = images.to(config.device).float()

    logits = model(images)
    probabilities = F.softmax(logits, dim=1)
    _, predicted_index = torch.max(probabilities, dim=1)
    predicted_class = config.class_names[predicted_index.item()]
    
    results.append(predicted_class)

df_test_submit = pd.read_csv(config.test_submit)
df_test_submit['class'] = results
df_test_submit.to_csv(config.output_dir+'resnet50d_gru_001_001.csv', index=False)