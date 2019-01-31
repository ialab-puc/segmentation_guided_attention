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

## Constants
PLACE_PULSE_PATH ='votes_clean.csv'
IMAGES_PATH= 'placepulse/'
MODEL_PATH = 'model.pth'



## Data loader class

class PlacePulseDataset(Dataset):
    
    def __init__(self,csv_file,img_dir,transform=None, cat=None, equal=False):
        self.placepulse_data = pd.read_csv(csv_file)
        if cat:
            self.placepulse_data = self.placepulse_data[self.placepulse_data['category'] == cat]
        if not equal:
            self.placepulse_data = self.placepulse_data[self.placepulse_data['winner'] != 'equal']
        
        self.img_dir =  img_dir
        self.transform = transform
        self.label = {'left':1, 'right':-1,'equal':0}
    
    def __len__(self):
        return len(self.placepulse_data)
    
    def __getitem__(self,idx):
        
        if type(idx) == torch.Tensor:
            idx = idx.tolist()
        left_img_name = os.path.join(self.img_dir, '{}.jpg'.format(self.placepulse_data.iloc[idx, 0]))
        left_image = io.imread(left_img_name)
        right_img_name = os.path.join(self.img_dir, '{}.jpg'.format(self.placepulse_data.iloc[idx, 1]))
        right_image = io.imread(right_img_name)
        winner = self.label[self.placepulse_data.iloc[idx, 2]]
        cat = self.placepulse_data.iloc[idx, -1]
        sample = {'left_image': left_image, 'right_image':right_image,'winner': winner, 'cat':cat}
        if self.transform:
            sample = self.transform(sample)
        return sample

#  Transformers 

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        left_image, right_image = sample['left_image'], sample['right_image']
        
        return {'left_image': ToTensor.transform_image(left_image),
                'right_image': ToTensor.transform_image(right_image),
                'winner': sample['winner'],
                'cat': sample['cat']}
    @classmethod
    def transform_image(cls,image):
        return torch.from_numpy(image.transpose((2, 0, 1))).float()
    
class Rescale():
    
    def __init__ (self,output_size):
        self.output_size = output_size
    
    def __call__(self, sample):
        left_image, right_image = sample['left_image'], sample['right_image']
        
        return {'left_image': transform.resize(left_image,self.output_size,anti_aliasing=True,mode='constant'),
                'right_image': transform.resize(right_image,self.output_size,anti_aliasing=True,mode='constant'),
                'winner': sample['winner'],
                'cat': sample['cat']}
        

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
        
data=PlacePulseDataset(PLACE_PULSE_PATH,IMAGES_PATH,transforms.Compose([Rescale((224,224)),ToTensor()]),'lively')
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    loss.to(device)
    loss.backward()
    optimizer.step()
    out_loss = loss.item()
    # reverse example
    inverse_label*=-1 #swap label
    inverse_label[inverse_label==-1] = 0
    inverse_outputs = net(input_right,input_left) #pass swapped input
    inverse_loss = criterion(inverse_outputs, inverse_label)
    inverse_loss.to(device)
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
        loss.to(device)
        loss.backward()
    return  { 'loss':loss.item(), 
            'y':label,
            'y_pred': outputs
            }

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
    
# @trainer.on(Events.EPOCH_COMPLETED)
# def log_training_results(trainer):
#     evaluator.run(dataloader)
#     metrics = evaluator.state.metrics
#     print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(trainer.state.epoch, metrics['avg_acc'], metrics['loss']))


handler = ModelCheckpoint('ss_cnn_models', 'test', save_interval=1, n_saved=2, create_dir=True, save_as_state_dict=True, require_empty=False)
trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {
            'model': net,
            'optimizer': optimizer,
            })
    
trainer.run(dataloader,max_epochs=10)

