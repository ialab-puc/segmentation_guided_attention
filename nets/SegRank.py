

# standard imports
import torch
from torch import nn
import sys
import numpy as np
import os
from utils.ranking import compute_ranking_loss, compute_ranking_accuracy
from utils.log import console_log,comet_log

# others
sys.path.append('segmentation')
from segmentation.networks.pspnet import Seg_Model

# constants
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
IGNORE_LABEL = 255
NUM_CLASSES = 19
INPUT_SIZE = '340,480'
RESTORE_FROM = 'segmentation/CS_scenes_60000.pth'

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:{}".format('0') if torch.cuda.is_available() else "cpu")

class SegRank(nn.Module):
    def __init__(self,image_size=(340,480), restore=RESTORE_FROM, n_layers=2):
        super(SegRank, self).__init__()
        self.image_h, self.image_w = image_size
        self.seg_net = Seg_Model(num_classes=NUM_CLASSES)
        self.seg_net.eval() # FIXME: code does not run without this
        self.n_layers = n_layers
        if restore is not None: self.seg_net.load_state_dict(torch.load(restore, map_location=device))

        for param in self.seg_net.parameters():  # freeze segnet params
            param.requires_grad = False

        sample = torch.randn([3,self.image_h,self.image_w]).unsqueeze(0)
        self.seg_dims = self.seg_net(sample)[0].size() # for layer size definitionlayers
        self.attentions = nn.ModuleList([nn.MultiheadAttention(NUM_CLASSES, NUM_CLASSES) for _ in range(self.n_layers)])
        self.output = nn.Linear(self.seg_dims[2]*self.seg_dims[3]*NUM_CLASSES, 1)

    def forward(self, left_batch, right_batch):
        return {
            'left': self.single_forward(left_batch),
            'right': self.single_forward(right_batch)
        }

    def single_forward(self, batch):
        batch_size = batch.size()[0]
        seg_output =  self.seg_net(batch)[0]
        seg_output_permuted = seg_output.permute([2,3,0,1])
        x = seg_output_permuted.contiguous().view(self.seg_dims[2]*self.seg_dims[3],batch_size, NUM_CLASSES)
        attn_list = []
        for attention in self.attentions:
            x, attn_weights = attention(x, x, x) #attn is size nxn , first row says importance of each pixel on calculating first pixel.
            attn_list.append(attn_weights)
        x = x.permute([1,0,2]).contiguous().view(batch_size,self.seg_dims[2]*self.seg_dims[3]*NUM_CLASSES)
        x = self.output(x)
        return {
            'output': x,
            'segmentation': seg_output,
            'attention': attn_list
        }

    def partial_eval(self):
        self.seg_net.eval()


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