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

class RSsCnn(nn.Module):
    
    def __init__(self,model):
        super(RSsCnn, self).__init__()
        self.cnn = model(pretrained=True).features
        x = torch.randn([3,320,320]).unsqueeze(0)
        output_size = self.cnn(x).size()
        self.dims = output_size[1]*2
        self.cnn_size = output_size
        self.conv_factor= output_size[2] - 5 #should be 1 or 2
        self.fuse_conv_1 = nn.Conv2d(self.dims,self.dims,3)
        self.fuse_conv_2 = nn.Conv2d(self.dims,self.dims,3)
        self.fuse_conv_3 = nn.Conv2d(self.dims,self.dims,2)
        self.fuse_fc = nn.Linear(self.dims*(self.conv_factor**2), 2)
        self.classifier = nn.LogSoftmax(dim=1)
        self.rank_fc_1 = nn.Linear(self.cnn_size[1]*self.cnn_size[2]*self.cnn_size[3], 4096)
        self.rank_fc_2 = nn.Linear(4096, 4096)
        self.rank_fc_out = nn.Linear(4096, 1)
        self.conv_drop  = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.drop  = nn.Dropout(0.3)
    
    def forward(self,left_image, right_image):
        batch_size = left_image.size()[0]
        left = self.cnn(left_image)
        right = self.cnn(right_image)
        x = torch.cat((left,right),1)
        x = self.fuse_conv_1(x)
        x = self.conv_drop(x)
        x = self.fuse_conv_2(x)
        x = self.conv_drop(x)
        x = self.fuse_conv_3(x)
        x = self.conv_drop(x)
        x = x.view(batch_size,self.dims*(self.conv_factor**2))
        x_clf = self.fuse_fc(x)
        x_clf = self.classifier(x_clf)
        x_rank_left = left.view(batch_size,self.cnn_size[1]*self.cnn_size[2]*self.cnn_size[3])
        x_rank_right = right.view(batch_size,self.cnn_size[1]*self.cnn_size[2]*self.cnn_size[3])
        x_rank_left = self.rank_fc_1(x_rank_left)
        x_rank_left = self.relu(x_rank_left)
        x_rank_left = self.drop(x_rank_left)
        x_rank_right = self.rank_fc_1(x_rank_right)
        x_rank_right = self.relu(x_rank_right)
        x_rank_right = self.drop(x_rank_right)
        x_rank_left = self.rank_fc_2(x_rank_left)
        x_rank_left = self.relu(x_rank_left)
        x_rank_left = self.drop(x_rank_left)
        x_rank_right = self.rank_fc_2(x_rank_right)
        x_rank_right = self.relu(x_rank_right)
        x_rank_right = self.drop(x_rank_right)
        x_rank_left = self.rank_fc_out(x_rank_left)
        x_rank_right = self.rank_fc_out(x_rank_right)
        return x_clf,x_rank_left, x_rank_right



def train(device, net, dataloader, val_loader, args):
    def update(engine, data):
        input_left, input_right, label = data['left_image'], data['right_image'], data['winner']
        input_left, input_right, label = input_left.to(device), input_right.to(device), label.to(device)
        rank_label = label.clone()
        inverse_label = label.clone()
        label[label==-1] = 0
        # zero the parameter gradients
        optimizer.zero_grad()
        rank_label = rank_label.float()
        output_clf,output_rank_left, output_rank_right = net(input_left,input_right)

        #compute clf loss
        loss_clf = clf_crit(output_clf,label)

        #compute ranking loss
        output_rank_left = output_rank_left.view(output_rank_left.size()[0])
        output_rank_right = output_rank_right.view(output_rank_right.size()[0])
        loss_rank = rank_crit(output_rank_left, output_rank_right, rank_label)

        #compute ranking accuracy
        rank_pairs = np.array(list(zip(output_rank_left,output_rank_right)))
        label_matrix = label.clone().cpu().detach().numpy()
        dup = np.zeros(label_matrix.shape)
        dup[label_matrix==0] = 1
        label_matrix = np.hstack((np.array([label_matrix]).T,np.array([dup]).T))
        rank_acc =  (rank_score(label_matrix,rank_pairs) - 0.5)/0.5

        # backward step
        loss = loss_clf + loss_rank        
        loss.backward()
        optimizer.step()

        #swapped forward
        inverse_label*=-1 #swap label
        inverse_rank_label = inverse_label.clone()
        inverse_rank_label = inverse_rank_label.float()
        inverse_label[inverse_label==-1] = 0
        outputs, output_rank_left, output_rank_right = net(input_right,input_left) #pass swapped input
        inverse_loss_clf = clf_crit(outputs, inverse_label)
        #compute ranking loss
        output_rank_left = output_rank_left.view(output_rank_left.size()[0])
        output_rank_right = output_rank_right.view(output_rank_right.size()[0])
        inverse_loss_rank = rank_crit(output_rank_left, output_rank_right, inverse_rank_label)
        #swapped backward
        inverse_loss = inverse_loss_clf + inverse_loss_rank
        inverse_loss.backward()
        optimizer.step()

        return  { 'loss':loss.item(), 
                'loss_clf':loss_clf.item(), 
                'loss_rank':loss_rank.item(),
                'y':label,
                'y_pred': output_clf,
                'rank_acc': rank_acc
                }

    def inference(engine,data):
        with torch.no_grad():
            input_left, input_right, label = data['left_image'], data['right_image'], data['winner']
            input_left, input_right, label = input_left.to(device), input_right.to(device), label.to(device)
            rank_label = label.clone()
            label[label==-1] = 0
            rank_label = rank_label.float()
            # forward
            output_clf,output_rank_left, output_rank_right = net(input_left,input_right)
            loss_clf = clf_crit(output_clf,label)
            output_rank_left = output_rank_left.view(output_rank_left.size()[0])
            output_rank_right = output_rank_right.view(output_rank_right.size()[0])

            rank_pairs = np.array(list(zip(output_rank_left,output_rank_right)))
            label_matrix = label.clone().cpu().detach().numpy()
            dup = np.zeros(label_matrix.shape)
            dup[label_matrix==0] = 1
            label_matrix = np.hstack((np.array([label_matrix]).T,np.array([dup]).T))
            rank_acc =  (rank_score(label_matrix,rank_pairs) - 0.5)/0.5

            loss_rank = rank_crit(output_rank_left, output_rank_right, rank_label)
            loss = loss_clf + loss_rank
            return  { 'loss':loss.item(), 
                'loss_clf':loss_clf.item(), 
                'loss_rank':loss_rank.item(),
                'y':label,
                'y_pred': output_clf,
                'rank_acc': rank_acc
                }
    net = net.to(device)

    clf_crit = nn.NLLLoss()
    rank_crit = nn.MarginRankingLoss(reduction='mean', margin=1)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    lamb = Variable(torch.FloatTensor([1]),requires_grad = False).cuda()[0]

    trainer = Engine(update)
    evaluator = Engine(inference)

    writer = SummaryWriter()
    RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x['loss_clf']).attach(trainer, 'loss_clf')
    RunningAverage(output_transform=lambda x: x['loss_rank']).attach(trainer, 'loss_rank')
    RunningAverage(output_transform=lambda x: x['rank_acc']).attach(trainer, 'rank_acc')
    RunningAverage(Accuracy(output_transform=lambda x: (x['y_pred'],x['y']))).attach(trainer,'avg_acc')

    RunningAverage(output_transform=lambda x: x['loss']).attach(evaluator, 'loss')
    RunningAverage(output_transform=lambda x: x['loss_clf']).attach(evaluator, 'loss_clf')
    RunningAverage(output_transform=lambda x: x['loss_rank']).attach(evaluator, 'loss_rank')
    RunningAverage(output_transform=lambda x: x['rank_acc']).attach(evaluator, 'rank_acc')
    RunningAverage(Accuracy(output_transform=lambda x: (x['y_pred'],x['y']))).attach(evaluator,'avg_acc')

    # pbar = ProgressBar(persist=False)
    # pbar.attach(trainer,['loss','avg_acc', 'rank_acc'])

    # pbar = ProgressBar(persist=False)
    # pbar.attach(evaluator,['loss','loss_clf', 'loss_rank','avg_acc'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        net.eval()
        writer.add_scalars(f'{args.attribute}/Training/accuracy', {
            'accuracy':trainer.state.metrics['avg_acc'],
            'rank_accuracy':trainer.state.metrics['rank_acc']
        }, trainer.state.epoch)
        writer.add_scalars(f'{args.attribute}/Training/loss', {
            'total':trainer.state.metrics['loss'],
            'clf':trainer.state.metrics['loss_clf'],
            'rank':trainer.state.metrics['loss_rank']
        }, trainer.state.epoch)
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        writer.add_scalars(f'{args.attribute}/Val/accuracy', {
            'accuracy':metrics['avg_acc'],
            'rank_accuracy':metrics['rank_acc']
        }, trainer.state.epoch)
        writer.add_scalars(f'{args.attribute}/Val/loss', {
            'total':metrics['loss'],
            'clf':metrics['loss_clf'],
            'rank':metrics['loss_rank']
        }, trainer.state.epoch)
        trainer.state.metrics['val_acc'] = metrics['avg_acc']
        
        print("Training Results - Epoch: {}  Avg Train accuracy: {:.5f} Avg Train loss: {:.6e} Avg Train clf loss: {:.6e} Avg Train rank loss: {:.6e}".format(
                trainer.state.epoch,
                trainer.state.metrics['avg_acc'],
                trainer.state.metrics['loss'],
                trainer.state.metrics['loss_clf'],
                trainer.state.metrics['loss_rank'])
            )
        print("Training Results - Epoch: {}  Avg Val accuracy: {:.5f} Avg Val loss: {:.6e} Avg Val clf loss: {:.6e} Avg Val rank loss: {:.6e}".format(
                trainer.state.epoch,
                metrics['avg_acc'],
                metrics['loss'],
                metrics['loss_clf'],
                metrics['loss_rank'])
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
    x = torch.randn([3,320,320]).unsqueeze(0)
    y = torch.randn([3,320,320]).unsqueeze(0)
    fwd =  net(x,y)
    print(fwd)
