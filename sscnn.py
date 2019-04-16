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

from data import *

## Constants
PLACE_PULSE_PATH ='votes_clean.csv'
IMAGES_PATH= 'placepulse/'
MODEL_PATH = 'model.pth'

#SsCnn definition

class SsCnn(nn.Module):
    
    def __init__(self):
        super(SsCnn, self).__init__()
        #shouldbe vgg19
        self.cnn = models.alexnet(pretrained=True).features
        #self.cnn.train() # to finetune pretrained model
        self.fuse_conv_1 = nn.Conv2d(512,512,3)
        self.fuse_conv_2 = nn.Conv2d(512,512,3)
        self.fuse_conv_3 = nn.Conv2d(512,512,2)
        self.fuse_fc = nn.Linear(512, 2)
        self.classifier = nn.LogSoftmax(dim=1)
                    
    def forward(self,left_image, right_image):
        batch_size = left_image.size()[0]
        left = self.cnn(left_image)
        right = self.cnn(right_image)
        x = torch.cat((left,right),1)
        x = self.fuse_conv_1(x)
        x = self.fuse_conv_2(x)
        x = self.fuse_conv_3(x)
        x = x.view(batch_size,512)
        x = self.fuse_fc(x)
        x = self.classifier(x)
        return x

def update(engine, data):
    input_left, input_right, label = data['left_image'], data['right_image'], data['winner']
    input_left, input_right, label = input_left.to(device), input_right.to(device), label.to(device)
    inverse_label = label.clone()
    label[label==-1] = 0
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(input_left,input_right)

    loss = criterion(outputs, label)

    loss.backward()
    optimizer.step()
    out_loss = loss.item()
    # reverse example
    inverse_label*=-1 #swap label
    inverse_label[inverse_label==-1] = 0
    inverse_outputs = net(input_right,input_left) #pass swapped input
    inverse_loss = criterion(inverse_outputs, inverse_label)
    inverse_loss.backward()
    optimizer.step()

    return  { 'loss':loss.item(), 
            'y':label,
            'y_pred': outputs
            }

def inference(engine,data):
    with torch.no_grad():
        input_left, input_right, label = data['left_image'], data['right_image'], data['winner']
        input_left, input_right, label = input_left.to(device), input_right.to(device), label.to(device)
        
        label[label==-1] = 0
        
        # forward
        outputs = net(input_left,input_right)
        loss = criterion(outputs, label)
    return  { 'loss':loss.item(), 
            'y':label,
            'y_pred': outputs
            }

def train(device):
    # device = torch.device('cpu')
    net = SsCnn()
    net = net.to(device)

    #torch ignite resume training

    # MODEL_PATH='ss_cnn_models/test_model_4.pth'
    # OPTIMIZER_PATH='ss_cnn_models/test_optimizer_4.pth'

    # net = RSsCnn()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # net.load_state_dict(torch.load(MODEL_PATH))
    # optimizer.load_state_dict(torch.load(OPTIMIZER_PATH))
    # epoch = 1

    # net.train()
    # net = net.to(device)

    # training with torch ignite
    from ignite.engine import Engine, Events, create_supervised_evaluator
    from ignite.metrics import Accuracy,Loss, RunningAverage
    from ignite.contrib.handlers import ProgressBar
    from ignite.handlers import ModelCheckpoint

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    lamb = 0.5

    trainer = Engine(update)
    evaluator = Engine(inference)

    RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'loss')
    RunningAverage(Accuracy(output_transform=lambda x: (x['y_pred'],x['y']))).attach(trainer,'avg_acc')

    RunningAverage(output_transform=lambda x: x['loss']).attach(evaluator, 'loss')
    RunningAverage(Accuracy(output_transform=lambda x: (x['y_pred'],x['y']))).attach(evaluator,'avg_acc')


    pbar = ProgressBar(persist=False)
    pbar.attach(trainer,['loss','avg_acc'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print("Training Results - Epoch: {}  Avg Val accuracy: {:.5f} Avg Val loss: {:.5f}".format(trainer.state.epoch, metrics['avg_acc'], metrics['loss']))

    handler = ModelCheckpoint('ss_cnn_models', 'test', save_interval=1, n_saved=2, create_dir=True, save_as_state_dict=True, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {
                'model': net,
                'optimizer': optimizer,
                })
        
    trainer.run(dataloader,max_epochs=10)

def test():
    pass