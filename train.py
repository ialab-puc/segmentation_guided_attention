# coding: utf-8

## dependencies
import argparse
import torch
from torchvision import transforms
from torch.utils.data import random_split, DataLoader


from data import PlacePulseDataset, ToTensor, Rescale

#script args
def arg_parse():
    parser = argparse.ArgumentParser(description='Training place pulse')
    parser.add_argument('--cuda', help="1 to run with cuda else 0", default=1, type=bool)
    parser.add_argument('--csv', help="dataset csv path", default="votes_clean.csv", type=str)
    parser.add_argument('--dataset', help="dataset images directory path", default="placepulse/", type=str)
    parser.add_argument('--attribute', help="placepulse attribute to train on", default="wealthy", type=str)
    parser.add_argument('--batch_size', help="batch size", default=32, type=int)
    parser.add_argument('--lr', help="learning_rate", default=0.001, type=float)
    parser.add_argument('--resume','--r', help="resume training",action='store_true')
    parser.add_argument('--wd', help="weight decay regularization", default=0.0, type=float)
    parser.add_argument('--num_workers', help="number of workers for data loader", default=4, type=int)
    parser.add_argument('--model_dir', help="directory to load and save models", default='models/', type=str)
    parser.add_argument('--model', help="model to use, sscnn or rsscnn", default='sscnn', type=str, choices=['rscnn','scnn'])
    parser.add_argument('--epoch', help="epoch to load training", default=1, type=int)
    parser.add_argument('--max_epochs', help="maximum training epochs", default=10, type=int)
    parser.add_argument('--cuda_id', help="gpu id", default=0, type=int)
    #TODO: ADD pretrainedmodel option
    return parser

        
if __name__ == '__main__':
    parser = arg_parse()
    args = parser.parse_args()
    print(args)
    data=PlacePulseDataset(args.csv,args.dataset,transforms.Compose([Rescale((224,224)),ToTensor()]), args.attribute)
    len_data = len(data)
    train_len = int(len_data*0.65)
    val_len = int(len_data*0.05)
    test_len = len_data-train_len-val_len
    train,val,test = random_split(data,[train_len , val_len, test_len])
    print(len(train))
    print(len(val))
    print(len(test))
    dataloader = DataLoader(train, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers)

    if args.cuda:
        device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    if args.model=="sscnn":
        from sscnn import SsCnn as Net
        from sscnn import train
    else:
        from rsscnn import RSsCnn as Net
        from rsscnn import train
    
    net = Net()
    if args.resume:
        net.load_state_dict(torch.load(os.path.join(args.model_dir,f'{args.model}_{args.attribute}_model_{args.epoch}.pth')))
        epoch = args.epoch
    
    train(device,net,dataloader,val_loader, args)


