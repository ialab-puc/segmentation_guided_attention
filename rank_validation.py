import torch
import numpy as np
import json
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch
import os
import cv2

from nets.SegRank import SegRank
from data import PlacePulseDataset, AdaptTransform
from utils.image_gen import shape_attention, masked_attention_images, get_palette, segmentation_to_image, attention_to_images
from utils.metrics import mass_center
import seg_transforms
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dist.init_process_group('gloo', init_method='file:///tmp/tmpfile', rank=0, world_size=1)
PLACE_PULSE_PATH ='votes'
IMAGES_PATH= '../datasets/placepulse/'
OUTPUT_IMAGES_PATH = '../storage/output_images'
MODELS = {
    'wealthy':'../storage/models_seg/segrank_resnet_wealthy_15_acc_model_0.6199546307884856.pth',
    'depressing': '../storage/models_seg/segrank_resnet_depressing_15_acc_model_0.6215900597907325.pth'
    #'safety': '../storage/models_seg/segrank_resnet_safety_15_model_34.pth',
    #'boring': '../storage/models_seg/segrank_resnet_boring_15_model_14.pth',
    #'lively': '../storage/models_seg/segrank_resnet_lively_15_model_20.pth',
    #'beautiful':'../storage/models_seg/segrank_resnet_beautiful_15_model_9.pth',
}

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
BATCH_SIZE = 8
N_LAYERS=1
N_HEADS=1
SOFTMAX=True

CS_CLASSES = [
    'road',
    'sidewalk',
    'building',
    'wall',
    'fence',
    'pole',
    'traffic light',
    'traffic sign',
    'vegetation',
    'terrain',
    'sky',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle'
]

transformers = transforms.Compose([
        AdaptTransform(seg_transforms.ToArray()),
        AdaptTransform(seg_transforms.SubstractMean(IMG_MEAN)),
        AdaptTransform(seg_transforms.Resize((244,244))),
        AdaptTransform(seg_transforms.ToTorchDims())
        ])

def generate_batch_stats(ids,forward_dict, output_dict, attribute, original_image):
    for i in range(len(ids)):
        if ids[i] not in output_dict:
            output_dict[ids[i]] = {'id':ids[i]}
        image_dict = output_dict[ids[i]]
        attention_map = shape_attention(forward_dict['attention'][0][i].cpu())
        if attribute not in image_dict:
            image_dict[attribute] = {}
            image_dict[attribute]['score'] = float(forward_dict['output'].squeeze().cpu().numpy()[i])
            image_dict[attribute]['object_metrics'] = process_images(attention_map, forward_dict['segmentation'][i].cpu(), original_image[i])
        if 'images' not in image_dict:
            image_dict['images'] = {}
        images = image_dict['images']

        if 'segmentation' not in images:
            seg_img = segmentation_to_image(forward_dict['segmentation'][i].cpu(), original_image[i], get_palette())
            file_path = f'{segmentations_path}/{ids[i]}.png'
            cv2.imwrite(file_path, seg_img) 
            images['segmentation'] = f'segmentations/{ids[i]}.png'

        if 'attention' not in images:
            image_dict['attention'] = {}
        attention_dict = image_dict['attention']
        if attribute not in attention_dict:
            attention_images, _ = attention_to_images(original_image[i], attention_map)
            file_path = f'{attentions_path}/{attribute}/{ids[i]}.png'
            cv2.imwrite(file_path, cv2.cvtColor(attention_images[0],cv2.COLOR_RGB2BGR))
            attention_dict[attribute] = f'attentions/{attribute}/{ids[i]}.png'

def process_images(attention_map, segmentation, original):
    seg = torch.from_numpy(np.asarray(np.argmax(segmentation, axis=0), dtype=np.uint8)).long()
    seg = torch.nn.functional.one_hot(seg, num_classes=19).permute([2,0,1]).float()
    masked, _, _, _ = masked_attention_images(original,segmentation, attention_map)
    total_seg = seg.sum()
    sums = np.fromiter(map(lambda x: x.sum(), masked),dtype=np.float)
    total = masked.sum()
    sorted_idx = sums.argsort()[::-1]
    
    metrics = {}

    for i, idx in enumerate(sorted_idx):
        sum_seg = seg[idx].sum()
        if sum_seg != 0.0:
            idx_metrics = {
                'attention': masked[idx].sum() * 100 / total,
                'segmentation': float((sum_seg * 100 / total_seg).numpy()),
                'mass_center': mass_center(seg[idx])
            }
            idx_metrics['ratio'] = idx_metrics['attention'] / idx_metrics['segmentation']
            metrics[CS_CLASSES[idx]] = idx_metrics
    return metrics

f = open('test.txt', 'w')
image_hash = {}

segmentations_path = f'{OUTPUT_IMAGES_PATH}/segmentations'
if not os.path.exists(segmentations_path):
    os.makedirs(segmentations_path)

attentions_path = f'{OUTPUT_IMAGES_PATH}/attentions'
if not os.path.exists(attentions_path):
    os.makedirs(attentions_path)

for attribute, model in MODELS.items():

    if not os.path.exists(f'{OUTPUT_IMAGES_PATH}/attentions/{attribute}'):
        os.makedirs(f'{OUTPUT_IMAGES_PATH}/attentions/{attribute}')

    dataset=PlacePulseDataset(
        f'{PLACE_PULSE_PATH}/{attribute}/val.csv',
        IMAGES_PATH,
        transform=transformers,
        equal=True,
        return_ids=True,
        return_images=True
        )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=4, drop_last=True)

    net = SegRank(image_size=(244,244), n_layers=N_LAYERS, n_heads=N_HEADS, softmax=SOFTMAX)
    net.load_state_dict(torch.load(model, map_location=device))
    net.to(device)
    net.eval()
    f.write(f'loaded {model}\n')
    f.flush()
    
    for index,batch in enumerate(loader):
        input_left = batch['left_image'].to(device)
        input_right = batch['right_image'].to(device)
        label = batch['winner'].to(device)
        left_id = batch['left_id']
        right_id = batch['right_id']
        left_original = batch['left_image_original']
        right_original = batch['right_image_original']
        with torch.no_grad():
            forward_dict = net(input_left,input_right)
        generate_batch_stats(left_id, forward_dict['left'], image_hash, attribute, left_original)
        generate_batch_stats(right_id, forward_dict['right'], image_hash, attribute, right_original)
        f.write(f'{index}/{len(loader)}\n')
        f.flush()
        if index == 2: break

    with open('output.jsonl', 'w') as outfile:
        for _id, _hash in image_hash.items():
            json.dump(_hash, outfile)
            outfile.write('\n')

f.close()
