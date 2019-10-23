import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from torch.autograd import Variable
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy,Loss, RunningAverage
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint
from tensorboardX import SummaryWriter
from sklearn.metrics import label_ranking_average_precision_score as rank_score
from timeit import default_timer as timer

class RCnn(nn.Module):
    
    def __init__(self,model):
        super(RCnn, self).__init__()
        self.cnn = model(pretrained=True).features
        for param in self.cnn.parameters():  # freeze cnn params
            param.requires_grad = False
        x = torch.randn([3,244,244]).unsqueeze(0)
        output_size = self.cnn(x).size()
        self.dims = output_size[1]*2
        self.cnn_size = output_size
        self.rank_fc_1 = nn.Linear(self.cnn_size[1]*self.cnn_size[2]*self.cnn_size[3], 4096)
        self.rank_fc_out = nn.Linear(4096, 1)
        self.relu = nn.ReLU()
        self.drop  = nn.Dropout(0.3)
    
    def forward(self,left_image, right_image):
        batch_size = left_image.size()[0]
        left = self.cnn(left_image)
        right = self.cnn(right_image)
        x_rank_left = left.view(batch_size,self.cnn_size[1]*self.cnn_size[2]*self.cnn_size[3])
        x_rank_right = right.view(batch_size,self.cnn_size[1]*self.cnn_size[2]*self.cnn_size[3])
        x_rank_left = self.rank_fc_1(x_rank_left)
        x_rank_left = self.relu(x_rank_left)
        x_rank_left = self.drop(x_rank_left)
        x_rank_right = self.rank_fc_1(x_rank_right)
        x_rank_right = self.relu(x_rank_right)
        x_rank_right = self.drop(x_rank_right)
        x_rank_left = self.rank_fc_out(x_rank_left)
        x_rank_right = self.rank_fc_out(x_rank_right)
        return x_rank_left, x_rank_right



def train(device, net, dataloader, val_loader, args,logger):
    def update(engine, data):
        input_left, input_right, label = data['left_image'], data['right_image'], data['winner']
        input_left, input_right, label = input_left.to(device), input_right.to(device), label.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        label = label.float()

        start = timer()
        output_rank_left, output_rank_right = net(input_left,input_right)
        end = timer()
        logger.info(f'FORWARD,{end-start:.4f}')

        #compute ranking loss
        start = timer()
        output_rank_left = output_rank_left.view(output_rank_left.size()[0])
        output_rank_right = output_rank_right.view(output_rank_right.size()[0])
        loss = rank_crit(output_rank_left, output_rank_right, label)
        
        end = timer()
        logger.info(f'LOSS,{end-start:.4f}')
        start = timer()
        #compute ranking accuracy
        rank_pairs = np.array(list(zip(output_rank_left,output_rank_right)))
        label_matrix = label.clone().cpu().detach().numpy()
        dup = np.zeros(label_matrix.shape)
        label_matrix[label_matrix==-1] = 0
        dup[label_matrix==0] = 1
        label_matrix = np.hstack((np.array([label_matrix]).T,np.array([dup]).T))
        rank_acc =  (rank_score(label_matrix,rank_pairs) - 0.5)/0.5
        
        end = timer()
        logger.info(f'RANK-ACC,{end-start:.4f}')

        # backward step
        start = timer()
        loss.backward()
        optimizer.step()
        end = timer()
        logger.info(f'BACKWARD,{end-start:.4f}')
        
        return  { 'loss':loss.item(), 
                'rank_acc': rank_acc
                }

    def inference(engine,data):
        with torch.no_grad():
            start = timer()
            input_left, input_right, label = data['left_image'], data['right_image'], data['winner']
            input_left, input_right, label = input_left.to(device), input_right.to(device), label.to(device)
            label = label.float()
            # forward
            output_rank_left, output_rank_right = net(input_left,input_right)
            output_rank_left = output_rank_left.view(output_rank_left.size()[0])
            output_rank_right = output_rank_right.view(output_rank_right.size()[0])
            loss = rank_crit(output_rank_left, output_rank_right, label)

            rank_pairs = np.array(list(zip(output_rank_left,output_rank_right)))
            label_matrix = label.clone().cpu().detach().numpy()
            dup = np.zeros(label_matrix.shape)
            label_matrix[label_matrix==-1] = 0
            dup[label_matrix==0] = 1
            label_matrix = np.hstack((np.array([label_matrix]).T,np.array([dup]).T))
            rank_acc =  (rank_score(label_matrix,rank_pairs) - 0.5)/0.5
            end = timer()
            logger.info(f'INFERENCE,{end-start:.4f}')
            return  { 'loss':loss.item(), 
                'rank_acc': rank_acc
                }
    net = net.to(device)

    rank_crit = nn.MarginRankingLoss(reduction='mean', margin=1)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

    trainer = Engine(update)
    evaluator = Engine(inference)

    writer = SummaryWriter()
    RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x['rank_acc']).attach(trainer, 'rank_acc')

    RunningAverage(output_transform=lambda x: x['loss']).attach(evaluator, 'loss')
    RunningAverage(output_transform=lambda x: x['rank_acc']).attach(evaluator, 'rank_acc')

    pbar = ProgressBar(persist=False)
    pbar.attach(trainer,['loss', 'rank_acc'])

    pbar = ProgressBar(persist=False)
    pbar.attach(evaluator,['loss','rank_acc'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        net.eval()
        writer.add_scalars(f'{args.attribute}/Training/accuracy', {
            'rank_accuracy':trainer.state.metrics['rank_acc']
        }, trainer.state.epoch)
        writer.add_scalars(f'{args.attribute}/Training/loss', {
            'total':trainer.state.metrics['loss']
        }, trainer.state.epoch)
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        writer.add_scalars(f'{args.attribute}/Val/accuracy', {
            'rank_accuracy':metrics['rank_acc']
        }, trainer.state.epoch)
        writer.add_scalars(f'{args.attribute}/Val/loss', {
            'total':metrics['loss']
        }, trainer.state.epoch)
        trainer.state.metrics['val_acc'] = metrics['rank_acc']
        
        print("Training Results - Epoch: {}  Avg Train accuracy: {:.5f} Avg Train loss: {:.6e} Avg Train clf loss: {:.6e} Avg Train rank loss: {:.6e}".format(
                trainer.state.epoch,
                trainer.state.metrics['loss'],
                trainer.state.metrics['rank_acc'])
            )
        print("Training Results - Epoch: {}  Avg Val accuracy: {:.5f} Avg Val loss: {:.6e} Avg Val clf loss: {:.6e} Avg Val rank loss: {:.6e}".format(
                trainer.state.epoch,
                metrics['loss'],
                metrics['rank_acc'])
            )
        net.train()

    handler = ModelCheckpoint(args.model_dir, '{}_{}_{}'.format(args.model, args.premodel, args.attribute),
                                n_saved=1,
                                create_dir=True,
                                save_as_state_dict=True,
                                require_empty=False,
                                score_function=lambda engine: engine.state.metrics['val_acc'])
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {
                'model': net
                })

    if (args.resume):
        def start_epoch(engine):
            engine.state.epoch = args.epoch
        trainer.add_event_handler(Events.STARTED, start_epoch)
        evaluator.add_event_handler(Events.STARTED, start_epoch)

    trainer.run(dataloader,max_epochs=args.max_epochs)
    
if __name__ == '__main__':
    from torchviz import make_dot
    net = RSsCnn(models.alexnet)
    x = torch.randn([3,244,244]).unsqueeze(0)
    y = torch.randn([3,244,244]).unsqueeze(0)
    fwd =  net(x,y)
    print(fwd)