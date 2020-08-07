import torch
import numpy as np
import json
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.distributed as dist

from nets.SegRank import SegRank
from data import PlacePulseDataset, AdaptTransform
from utils.image_gen import shape_attention, masked_attention_images
import seg_transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dist.init_process_group('gloo', init_method='file:///tmp/tmpfile', rank=0, world_size=1)
PLACE_PULSE_PATH ='votes'
IMAGES_PATH= '../datasets/placepulse/'
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

transformers = transforms.Compose([
        AdaptTransform(seg_transforms.ToArray()),
        AdaptTransform(seg_transforms.SubstractMean(IMG_MEAN)),
        AdaptTransform(seg_transforms.Resize((244,244))),
        AdaptTransform(seg_transforms.ToTorchDims())
        ])

def generate_batch_stats(ids,forward_dict, output_dict, attribute):
    for i in range(len(ids)):
        if ids[i] not in output_dict:
            output_dict[ids[i]] = {'id':ids[i]}
        image_dict = output_dict[ids[i]]
        if attribute not in image_dict:
            image_dict[attribute] = {}
            image_dict[attribute]['score'] = float(forward_dict['output'].squeeze().cpu().numpy()[i])


f = open('test.txt', 'w')
image_hash = {}
for attribute, model in MODELS.items():

    dataset=PlacePulseDataset(
        f'{PLACE_PULSE_PATH}/{attribute}/val.csv',
        IMAGES_PATH,
        transform=transformers,
        equal=True,
        return_ids=True
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
        input_left, input_right, label, left_id, right_id = batch['left_image'].to(device), batch['right_image'].to(device), batch['winner'].to(device), batch['left_id'], batch['right_id']
        with torch.no_grad():
            forward_dict = net(input_left,input_right)
        generate_batch_stats(left_id, forward_dict['left'], image_hash, attribute)
        generate_batch_stats(right_id, forward_dict['right'], image_hash, attribute)
        f.write(f'{index}/{len(loader)}\n')
        f.flush()
        if index == 2: break

    with open('output.jsonl', 'w') as outfile:
        for _id, _hash in image_hash.items():
            json.dump(_hash, outfile)
            outfile.write('\n')

f.close()
