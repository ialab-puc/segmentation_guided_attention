import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.distributed as dist

from nets.SegRank import SegRank
from data import PlacePulseDataset, AdaptTransform
import seg_transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dist.init_process_group('gloo', init_method='file:///tmp/tmpfile', rank=0, world_size=1)
PLACE_PULSE_PATH ='votes'
IMAGES_PATH= '../datasets/placepulse/'
MODELS = {
    'wealthy':'../storage/models_seg/segrank_resnet_wealthy_15_reg_model_0.6167474968710889.pth'
    #'depressing': '../storage/models_seg/segrank_resnet_depressing_15_acc_model_0.6215900597907325.pth'
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

f = open('test.txt', 'w')
df = None
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
    image_hash = {}
    for index,batch in enumerate(loader):
        input_left, input_right, label, left_id, right_id = batch['left_image'].to(device), batch['right_image'].to(device), batch['winner'].to(device), batch['left_id'], batch['right_id']
        with torch.no_grad():
            forward_dict = net(input_left,input_right)
        output_rank_left, output_rank_right =  forward_dict['left']['output'], forward_dict['right']['output']
        for i in range(len(left_id)):
            if left_id[i] not in image_hash: image_hash[left_id[i]] = output_rank_left.squeeze().cpu().numpy()[i]
            if right_id[i] not in image_hash: image_hash[right_id[i]] = output_rank_right.squeeze().cpu().numpy()[i]
        f.write(f'{index}/{len(loader)}\n')
        f.flush()
    if df is None:
        df=pd.DataFrame.from_dict(image_hash, orient='index', columns=[attribute])
    else:
        df=df.join(pd.DataFrame.from_dict(image_hash, orient='index', columns=[attribute]),how='outer')
    df.to_csv('rank.csv', index_label='id')
f.close()
