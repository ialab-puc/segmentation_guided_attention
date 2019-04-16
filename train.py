# coding: utf-8

## dependencies
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import DataLoader, ToTensor, Rescale
from sscnn import SsCnn
from rsscnn import RSsCnn


## Constants
PLACE_PULSE_PATH ='votes_clean.csv'
IMAGES_PATH= 'placepulse/'
MODEL_PATH = 'model.pth'

        
if __name__ == '__main__':

    data=PlacePulseDataset(PLACE_PULSE_PATH,IMAGES_PATH,transforms.Compose([Rescale((224,224)),ToTensor()]),'wealthy')
    len_data = len(data)
    train_len = int(len_data*0.65)
    val_len = int(len_data*0.05)
    test_len = len_data-train_len-val_len
    train,val,test = random_split(data,[train_len , val_len, test_len])
    print(len(train))
    print(len(val))
    print(len(test))
    dataloader = DataLoader(train, batch_size=32,
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val, batch_size=32,
                            shuffle=True, num_workers=4)



    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
