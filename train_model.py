#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 09:37:14 2017

@author: strong
"""
import os 
import torch
#from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils,datasets, models
from PIL import Image
import time
import torch.nn as nn
import torch.optim as optim 
from torch.optim import lr_scheduler
from torch.autograd import Variable
from densenet import densenet121
import torch.nn.functional as F
from sklearn.metrics import roc_curve 
from get_data import train_model

use_gpu = torch.cuda.is_available()
#model_ft = models.densenet121(pretrained = False)
model_ft = densenet121(pretrained = False)
#torch.nn.Linear(in_features,out_features,bias = True)
num_ftrs = model_ft.classifier.in_features

model_ft.classifier = nn.Linear(num_ftrs,1)

if use_gpu:
    model_ft = model_ft.cuda()



criterion = nn.BCELoss()

optimizer_ft = optim.Adam(model_ft.parameters(), lr = 0.001,  betas=(0.9, 0.999))

exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=2,verbose=True)

model_ft = train_model(model_ft,criterion, optimizer_ft, exp_lr_scheduler,num_epochs = 25) 