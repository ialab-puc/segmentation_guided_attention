import os
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms

from timeit import default_timer as timer

## Data loader class
class PlacePulseDataset(Dataset):

    def __init__(self,csv_file,img_dir,transform=None, cat=None, equal=False,logger=None,return_images=False):
        self.placepulse_data = pd.read_csv(csv_file)
        if cat:
            self.placepulse_data = self.placepulse_data[self.placepulse_data['category'] == cat]
        if not equal:
            self.placepulse_data = self.placepulse_data[self.placepulse_data['winner'] != 'equal']
        self.logger = logger
        self.img_dir =  img_dir
        self.transform = transform
        self.label = {'left':1, 'right':-1,'equal':0}
        self.return_images = return_images

    def __len__(self):
        return len(self.placepulse_data)

    def __getitem__(self,idx):
        start = timer()
        if type(idx) == torch.Tensor:
            idx = idx.tolist()
        left_img_name = os.path.join(self.img_dir, '{}.jpg'.format(self.placepulse_data.iloc[idx, 0]))
        left_image = io.imread(left_img_name)
        right_img_name = os.path.join(self.img_dir, '{}.jpg'.format(self.placepulse_data.iloc[idx, 1]))
        right_image = io.imread(right_img_name)
        winner = self.label[self.placepulse_data.iloc[idx, 2]]
        cat = self.placepulse_data.iloc[idx, -1]
        sample = {'left_image': left_image, 'right_image':right_image,'winner': winner}
        if self.transform:
            sample = self.transform(sample)
        if self.return_images:
            sample['left_image_original'] = left_image
            sample['right_image_original'] = right_image
        end = timer()
        self.logger.info(f'DATALOADER,{end-start:.4f}')
        return sample

class AdaptTransform():
    def __init__ (self,transform):
        self.transform = transform

    def __call__(self, sample):
        left_image, right_image = sample['left_image'], sample['right_image']

        return {'left_image': self.transform(left_image),
                'right_image': self.transform(right_image),
                'winner': sample['winner']}