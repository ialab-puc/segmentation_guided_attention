# coding: utf-8

## dependencies
import argparse
import os
import torch
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
import torchvision.models as models

from data import PlacePulseDataset, AdaptTransform
import logging
from datetime import date
import os



#script args
def arg_parse():
    parser = argparse.ArgumentParser(description='Training place pulse')
    parser.add_argument('--cuda', help="1 to run with cuda else 0", default=1, type=bool)
    parser.add_argument('--csv', help="path to placepulse csv dirs", default="votes/", type=str)
    parser.add_argument('--dataset', help="dataset images directory path", default="placepulse/", type=str)
    parser.add_argument('--attribute', help="placepulse attribute to train on", default="wealthy", type=str,  choices=['wealthy','lively', 'depressing', 'safety','boring','beautiful'])
    parser.add_argument('--batch_size', help="batch size", default=32, type=int)
    parser.add_argument('--lr', help="learning_rate", default=0.001, type=float)
    parser.add_argument('--resume','--r', help="resume training",action='store_true')
    parser.add_argument('--wd', help="weight decay regularization", default=0.0, type=float)
    parser.add_argument('--num_workers', help="number of workers for data loader", default=4, type=int)
    parser.add_argument('--model_dir', help="directory to load and save models", default='models/', type=str)
    parser.add_argument('--model', help="model to use, sscnn or rsscnn", default='sscnn', type=str, choices=['rsscnn','sscnn','rcnn'])
    parser.add_argument('--epoch', help="epoch to load training", default=1, type=int)
    parser.add_argument('--max_epochs', help="maximum training epochs", default=10, type=int)
    parser.add_argument('--cuda_id', help="gpu id", default=0, type=int)
    parser.add_argument('--premodel', help="premodel to use, alex or vgg or dense", default='alex', type=str, choices=['alex','vgg','dense'])
    parser.add_argument('--finetune','--ft', help="1 to finetune premodel else 0", default=0, type=bool)
    parser.add_argument('--pbar','--pb', help="1 to add pbars else 0", default=0, type=bool)
    return parser

        
if __name__ == '__main__':
    parser = arg_parse()
    args = parser.parse_args()
    print(args)
    if 'logs' not in os.listdir():
        os.mkdir('logs')
    logging.basicConfig(format='%(message)s',filename=f'logs/{args.attribute}-{date.today().strftime("%d-%m-%Y")}.log')
    logger = logging.getLogger('timer')
    logger.setLevel(logging.INFO) #set the minimum level of message logging

    train=PlacePulseDataset(
        f'{args.csv}/{args.attribute}/train.csv',
        args.dataset,
        transform=transforms.Compose([
            AdaptTransform(transforms.ToPILImage()),
            # AdaptTransform(transforms.Resize((244,244))),
            AdaptTransform(transforms.RandomResizedCrop(244)),
            AdaptTransform(transforms.RandomHorizontalFlip(p=0.3)),
            AdaptTransform(transforms.ToTensor())
            ]),
        logger=logger
        )
    val=PlacePulseDataset(
        f'{args.csv}/{args.attribute}/val.csv',
        args.dataset,
        transform=transforms.Compose([
            AdaptTransform(transforms.ToPILImage()),
            AdaptTransform(transforms.Resize((244,244))),
            AdaptTransform(transforms.ToTensor())
            ]),
        logger=logger
        )
    dataloader = DataLoader(train, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers)

    if args.cuda:
        device = torch.device("cuda:{}".format(args.cuda_id) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    if args.model=="sscnn":
        from nets.sscnn import SsCnn as Net
        from nets.sscnn import train
    elif args.model=="rcnn":
        from nets.rcnn import RCnn as Net
        from nets.rcnn import train
    else:
        from nets.rsscnn import RSsCnn as Net
        from nets.rsscnn import train

    resnet18 = models.resnet101
    alexnet = models.alexnet
    vgg16 = models.vgg19
    dense = models.densenet121

    models = {
        'alex':models.alexnet,
        'vgg':models.vgg19,
        'dense':models.densenet121
    }

    net = Net(models[args.premodel], finetune=args.finetune)
    if args.resume:
        net.load_state_dict(torch.load(os.path.join(args.model_dir,'{}_{}_{}_model_{}.pth'.format(
            args.model,
            args.premodel,
            args.attribute,
            args.epoch
        ))))
    
    train(device,net,dataloader,val_loader, args, logger)


