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
from timeit import default_timer as timer
from radam import RAdam
from utils.ranking import compute_ranking_loss, compute_ranking_accuracy
from utils.log import epoch_log
from loss import RankingLoss
class RCnn(nn.Module):

    def __init__(self,model, finetune=False):
        super(RCnn, self).__init__()
        try:
            self.cnn = model(pretrained=True).features
        except AttributeError:
            self.cnn = nn.Sequential(*list(model(pretrained=True).children())[:-1])
        if not finetune:
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
        loss = compute_ranking_loss(output_rank_left, output_rank_right, label, rank_crit)
        end = timer()

        logger.info(f'LOSS,{end-start:.4f}')

        #compute ranking accuracy
        start = timer()
        rank_acc = compute_ranking_accuracy(output_rank_left, output_rank_right, label)
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
            output_rank_left, output_rank_right = net(input_left,input_right)
            loss = compute_ranking_loss(output_rank_left, output_rank_right, label, rank_crit)
            rank_acc = compute_ranking_accuracy(output_rank_left, output_rank_right, label)
            end = timer()
            logger.info(f'INFERENCE,{end-start:.4f}')
            return  { 'loss':loss.item(),
                'rank_acc': rank_acc
                }
    net = net.to(device)
    if args.equal:
        rank_crit = RankingLoss(margin=1, tie_margin=0)
        print("using new loss")
    else:
        rank_crit = nn.MarginRankingLoss(reduction='mean', margin=1)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    #optimizer = RAdam(net.parameters(), lr=args.lr, weight_decay=args.wd)

    trainer = Engine(update)
    evaluator = Engine(inference)

    writer = SummaryWriter()
    RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x['rank_acc']).attach(trainer, 'rank_acc')

    RunningAverage(output_transform=lambda x: x['loss']).attach(evaluator, 'loss')
    RunningAverage(output_transform=lambda x: x['rank_acc']).attach(evaluator, 'rank_acc')

    if args.pbar:
        pbar = ProgressBar(persist=False)
        pbar.attach(trainer,['loss', 'rank_acc'])

        pbar = ProgressBar(persist=False)
        pbar.attach(evaluator,['loss','rank_acc'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        net.eval()
        evaluator.run(val_loader)
        trainer.state.metrics['val_acc'] = evaluator.state.metrics['rank_acc']
        net.train()
        epoch_log(
            {
                "accuracy": { 'rank_accuracy':trainer.state.metrics['rank_acc'] },
                "loss": { 'total':trainer.state.metrics['loss'] }
            },
            {
                "accuracy": { 'rank_accuracy': evaluator.state.metrics['rank_acc'] },
                "loss": { 'total':evaluator.state.metrics['loss'] }
            },
            writer,
            args.attribute,
            trainer.state.epoch
        )

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
    net = RCnn(models.resnet50)
    x = torch.randn([3,244,244]).unsqueeze(0)
    y = torch.randn([3,244,244]).unsqueeze(0)
    fwd =  net(x,y)
    print(fwd)