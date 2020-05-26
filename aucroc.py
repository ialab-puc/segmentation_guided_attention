import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.distributed as dist
from sklearn.metrics import roc_auc_score, roc_curve

from nets.SegRank import SegRank
from data import PlacePulseDataset, AdaptTransform
import seg_transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dist.init_process_group('gloo', init_method='file:///tmp/tmpfile', rank=0, world_size=1)
PLACE_PULSE_PATH ='votes'
IMAGES_PATH= 'pp_cropped/'
MODELS = {
    'wealthy':'segrank_resnet_wealthy_12_model_23.pth',
    'depressing':'segrank_resnet_depressing_12_model_13.pth',
    'safety':'segrank_resnet_safety_12_model_25.pth',
    'lively':'segrank_resnet_lively_12_model_28.pth',
    'boring':'segrank_resnet_boring_12_model_17.pth',
    'beautiful': 'segrank_resnet_beautiful_12_model_36.pth'
}

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
BATCH_SIZE = 2

transformers = transforms.Compose([
        AdaptTransform(seg_transforms.ToArray()),
        AdaptTransform(seg_transforms.SubstractMean(IMG_MEAN)),
        AdaptTransform(seg_transforms.Resize((244,244))),
        AdaptTransform(seg_transforms.ToTorchDims())
        ])

for attribute, model in MODELS.items():

    dataset=PlacePulseDataset(
        f'{PLACE_PULSE_PATH}/{attribute}/val.csv',
        IMAGES_PATH,
        transform=transformers,
        return_images=True
        )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=4, drop_last=True)

    net = SegRank(image_size=(244,244))
    net.load_state_dict(torch.load(model, map_location=device))
    net.eval()
    print(f'loaded {model}')
    scores = torch.Tensor()
    classifications = torch.Tensor().long()
    for i,batch in enumerate(loader):
        input_left, input_right, label = batch['left_image'].to(device), batch['right_image'].to(device), batch['winner'].to(device)
        with torch.no_grad():
            forward_dict = net(input_left,input_right)
            output_rank_left, output_rank_right =  forward_dict['left']['output'], forward_dict['right']['output']
            diff = (output_rank_left - output_rank_right).squeeze(1)
            scores = torch.cat((scores, diff),0)
            classifications = torch.cat((classifications, label),0)
            print(f'{i}/{len(loader)}', end='\r')
    print(f'{attribute}: {roc_auc_score(classifications, scores)}')