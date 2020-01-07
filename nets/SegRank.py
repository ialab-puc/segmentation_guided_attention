

# standard imports
import torch
from torch import nn
import sys
import numpy as np
import os
from utils.ranking import compute_ranking_loss, compute_ranking_accuracy
from utils.log import epoch_log


# others
sys.path.insert(0,'segmentation')
from segmentation.networks.pspnet import Res_Deeplab

# constants
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
IGNORE_LABEL = 255
NUM_CLASSES = 19
INPUT_SIZE = '340,480'
RESTORE_FROM = 'segmentation/CS_scenes_40000.pth'

import warnings
warnings.filterwarnings("ignore") 

device = torch.device("cuda:{}".format('0') if torch.cuda.is_available() else "cpu")

class SegRank(nn.Module):
    def __init__(self,image_size=(340,480), restore=RESTORE_FROM):
        super(SegRank, self).__init__()
        self.image_h, self.image_w = image_size
        self.seg_net = Res_Deeplab(num_classes=NUM_CLASSES)
        self.seg_net.eval() # FIXME: code does not run without this
        if restore is not None: self.seg_net.load_state_dict(torch.load(restore, map_location=device))
        for param in self.seg_net.parameters():  # freeze segnet params
            param.requires_grad = False
        self.interp = lambda x: nn.functional.interpolate(x,size=image_size, mode='bilinear', align_corners=True)
        self.fc_seg = nn.Linear(NUM_CLASSES,1)
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.fc_1 = nn.Linear(self.image_h*self.image_w/16, 1000)
        self.relu = nn.ReLU()
        self.output = nn.Linear(1000, 1)

    def forward(self, left_batch, right_batch):
        return self.single_forward(left_batch), self.single_forward(right_batch)

    def single_forward(self, batch):
        batch_size = batch.size()[0]
        seg_output =  self.seg_net(batch)[0]
        seg_output = self.interp(seg_output).permute([0,2,3,1])
        x = self.fc_seg(seg_output).view(batch_size, self.image_h*self.image_w)
        x = self.pool(x)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.output(x)
        return x

if __name__ == '__main__':

    import torch.distributed as dist
    dist.init_process_group('gloo', init_method='file:///tmp/tmpfile', rank=0, world_size=1)

    h, w = map(int, INPUT_SIZE.split(','))
    model = SegRank(restore=RESTORE_FROM)
    left = torch.randn([3,h,w]).unsqueeze(0).to(device)
    right = torch.randn([3,h,w]).unsqueeze(0).to(device)
    model.eval()
    model.to(device)
    print(model(left, right))