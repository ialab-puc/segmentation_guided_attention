import torchvision.models as models
import torch.nn as nn

class RSsCnn(nn.Module):
    
    def __init__(self):
        super(RSsCnn, self).__init__()
        self.cnn = models.vgg19(pretrained=True).features
        self.fuse_conv_1 = nn.Conv2d(1024,1024,3)
        self.fuse_conv_2 = nn.Conv2d(1024,1024,3)
        self.fuse_conv_3 = nn.Conv2d(1024,1024,2)
        self.fuse_fc = nn.Linear(1024*4, 2)
        self.classifier = nn.LogSoftmax(dim=1)
        self.rank_fc_1 = nn.Linear(512*7*7, 4096)
        self.rank_fc_2 = nn.Linear(4096, 1)
    
    def forward(self,left_image, right_image):
        batch_size = left_image.size()[0]
        left = self.cnn(left_image)
        right = self.cnn(right_image)
        x = torch.cat((left,right),1)
        x = self.fuse_conv_1(x)
        x = self.fuse_conv_2(x)
        print(x.size())
        x = self.fuse_conv_3(x)
        print(x.size())
        x = x.view(batch_size,1024*4)
        x_clf = self.fuse_fc(x)
        x_clf = self.classifier(x_clf)
        
        x_rank_left = left.view(batch_size,512*7*7)
        x_rank_right = right.view(batch_size,512*7*7)
        x_rank_left = self.rank_fc_1(x_rank_left)
        x_rank_right = self.rank_fc_1(x_rank_right)
        x_rank_left = self.rank_fc_2(x_rank_left)
        x_rank_right = self.rank_fc_2(x_rank_right)
        return x_clf,x_rank_left, x_rank_right

def update(engine, data):
    input_left, input_right, label = data['left_image'], data['right_image'], data['winner']
    input_left, input_right, label = input_left.to(device), input_right.to(device), label.to(device)
    rank_label = label.clone()
    label[label==-1] = 0
    # zero the parameter gradients
    optimizer.zero_grad()
    rank_label = rank_label.float()
    # forward + backward + optimize
    output_clf,output_rank_left, output_rank_right = net(input_left,input_right)

    loss_clf = clf_crit(output_clf,label)
#   print(output_rank_left, output_rank_right, rank_label)
    loss_rank = rank_crit(output_rank_left, output_rank_right, rank_label)
    loss = loss_clf + loss_rank*lamb
    loss.to(device)
    loss.backward()
    optimizer.step()
    return  { 'loss':loss.item(), 
            'loss_clf':loss_clf.item(), 
            'loss_rank':loss_rank.item(),
            'y':label,
            'y_pred': output_clf
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
        loss_rank = rank_crit(output_rank_left, output_rank_right, rank_label)
        loss = loss_clf + loss_rank*lamb
        loss.to(device)
        return  { 'loss':loss.item(), 
            'loss_clf':loss_clf.item(), 
            'loss_rank':loss_rank.item(),
            'y':label,
            'y_pred': output_clf
            }

def train():
    net = RSsCnn()
    net = net.to(device)

    #torch ignite resume training
    #MODEL_PATH='models/test_model_state_dict_4.pth'
    #OPTIMIZER_PATH='models/test_optimizer_state_dict_4.pth'

    #net = RSsCnn()
    #optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    #net.load_state_dict(torch.load(MODEL_PATH))
    #optimizer.load_state_dict(torch.load(OPTIMIZER_PATH))
    #epoch = 1

    # net.train()
    # net = net.to(device)

    # training with torch ignite
    from ignite.engine import Engine, Events, create_supervised_evaluator
    from ignite.metrics import Accuracy,Loss, RunningAverage
    from ignite.contrib.handlers import ProgressBar
    from ignite.handlers import ModelCheckpoint

    clf_crit = nn.NLLLoss()
    rank_crit = nn.MarginRankingLoss(reduction='sum')
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    lamb = 0.5



    trainer = Engine(update)
    evaluator = Engine(inference)

    RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'loss')
    RunningAverage(Accuracy(output_transform=lambda x: (x['y_pred'],x['y']))).attach(trainer,'avg_acc')

    RunningAverage(output_transform=lambda x: x['loss']).attach(evaluator, 'loss')
    RunningAverage(Accuracy(output_transform=lambda x: (x['y_pred'],x['y']))).attach(evaluator,'avg_acc')


    pbar = ProgressBar(persist=False)
    pbar.attach(trainer,['loss','avg_acc'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print("Training Results - Epoch: {}  Avg Val accuracy: {:.2f} Avg Val loss: {:.2f}".format(trainer.state.epoch, metrics['avg_acc'], metrics['loss']))
        
    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_training_results(trainer):
    #     evaluator.run(dataloader)
    #     metrics = evaluator.state.metrics
    #     print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(trainer.state.epoch, metrics['avg_acc'], metrics['loss']))


    handler = ModelCheckpoint('models', 'test', save_interval=1, n_saved=2, create_dir=True, save_as_state_dict=True, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {
                'model': net,
                'optimizer': optimizer,
                })
        
    trainer.run(dataloader,max_epochs=10)

def test():
    pass