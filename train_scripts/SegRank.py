

# standard imports
import torch
from torch import nn
import sys
import numpy as np
import os


# os.environ["CUDA_VISIBLE_DEVICES"]='0'
# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '29500'

# others
sys.path.insert(0,'segmentation')
from segmentation.networks.pspnet import Res_Deeplab

import torch.distributed as dist
dist.init_process_group('gloo', init_method='file:///tmp/tmpfile', rank=0, world_size=1)

# constants
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
DATA_DIRECTORY = 'pp_cropped'
DATA_LIST_PATH = './dataset/list/placepulse/test.lst'
IGNORE_LABEL = 255
NUM_CLASSES = 19
INPUT_SIZE = '340,480'
RESTORE_FROM = 'segmentation/CS_scenes_40000.pth'


device = torch.device("cuda:{}".format('0') if torch.cuda.is_available() else "cpu")
