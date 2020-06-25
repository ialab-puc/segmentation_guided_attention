import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy,Loss, RunningAverage
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint
from PIL import Image as PILImage
from utils.ranking import *
from utils.log import console_log, comet_log, comet_image_log, image_log
from utils.image_gen import get_palette
from utils.accuracy import RankAccuracy
from loss import RankingLoss, LogSumExpLoss, VarianceRegularizer, RegularizedLoss

def train(device, net, dataloader, val_loader, args, logger, experiment):
    def update(engine, data):
        input_left, input_right, label, left_original = data['left_image'], data['right_image'], data['winner'], data['left_image_original']
        input_left, input_right, label = input_left.to(device), input_right.to(device), label.to(device)
        attribute = data['attribute'].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        label = label.float()

        forward_dict = net(input_left,input_right)

        output_rank_left, output_rank_right =  forward_dict['left']['output'], forward_dict['right']['output']

        if args.attribute == 'all':
            loss = compute_multiple_ranking_loss(output_rank_left, output_rank_right, label, rank_crit, attribute)
            rank_acc = compute_multiple_ranking_accuracy(output_rank_left, output_rank_right, label, attribute)
        else:
            loss = compute_ranking_loss(output_rank_left, output_rank_right, label, rank_crit)
            rank_acc = compute_ranking_accuracy(output_rank_left, output_rank_right, label)

        # backward step
        loss.backward()
        optimizer.step()
        scheduler.step()

        if trainer.state.iteration == 1:
            segmentation = forward_dict['left'].get('segmentation',[None])[0]
            original = left_original[0]
            attention_map = forward_dict['left'].get('attention',[[None]])[0][0]
            image_log(segmentation,original,attention_map,palette,experiment,0, normalize=args.attention_normalize)

        return  { 'loss':loss.item(),
                'rank_acc': rank_acc,
                'rank_left': output_rank_left,
                'rank_right': output_rank_right,
                'label': label
                }

    def inference(engine,data):
        with torch.no_grad():
            input_left, input_right, label, left_original = data['left_image'], data['right_image'], data['winner'], data['left_image_original']
            input_left, input_right, label = input_left.to(device), input_right.to(device), label.to(device)
            attribute = data['attribute'].to(device)
            label = label.float()
            forward_dict = net(input_left,input_right)
            output_rank_left, output_rank_right =  forward_dict['left']['output'], forward_dict['right']['output']
            if args.attribute == 'all':
                loss = compute_multiple_ranking_loss(output_rank_left, output_rank_right, label, rank_crit, attribute)
                rank_acc = compute_multiple_ranking_accuracy(output_rank_left, output_rank_right, label, attribute)
            else:
                loss = compute_ranking_loss(output_rank_left, output_rank_right, label, rank_crit)
                rank_acc = compute_ranking_accuracy(output_rank_left, output_rank_right, label)

            if evaluator.state.iteration == 1:
                segmentation = forward_dict.get('segmentation',[None])[0]
                original = left_original[0]
                attention_map = forward_dict['left'].get('attention',[[None]])[0][0]
                image_log(segmentation,original,attention_map,palette,experiment,trainer.state.epoch, normalize=args.attention_normalize)

            return  { 'loss':loss.item(),
                'rank_acc': rank_acc,
                'rank_left': output_rank_left,
                'rank_right': output_rank_right,
                'label': label
                }

    net = net.to(device)
    if args.logexp:
        rank_crit = LogSumExpLoss()
        print("Using log sum exp")
    else:
        if args.equal:
            rank_crit = RankingLoss(margin=1, tie_margin=0)
            print("using tie loss")
        else:
            rank_crit = nn.MarginRankingLoss(reduction='mean', margin=1)
    if args.reg:
        print(f"using regularizer with alpha={args.alpha}")
        reg = VarianceRegularizer()
        rank_crit =  RegularizedLoss(rank_crit, reg, args.alpha)

    #optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98), eps=1e-09)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.995, last_epoch=-1)

    trainer = Engine(update)
    evaluator = Engine(inference)

    palette = get_palette(19)
    RunningAverage(output_transform=lambda x: x['loss'], device=device).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x['rank_acc'], device=device).attach(trainer, 'rank_acc')
    RankAccuracy(output_transform=lambda x: (x['rank_left'], x['rank_right'], x['label']), device=device).attach(trainer,'acc')

    RunningAverage(output_transform=lambda x: x['loss'], device=device).attach(evaluator, 'loss')
    RunningAverage(output_transform=lambda x: x['rank_acc'], device=device).attach(evaluator, 'rank_acc')
    RankAccuracy(output_transform=lambda x: (x['rank_left'], x['rank_right'], x['label']),device=device).attach(evaluator,'acc')

    if args.pbar:
        pbar = ProgressBar(persist=False)
        pbar.attach(trainer,['loss', 'rank_acc'])

        pbar = ProgressBar(persist=False)
        pbar.attach(evaluator,['loss','rank_acc'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        net.eval()
        evaluator.run(val_loader)
        trainer.state.metrics['val_acc'] = evaluator.state.metrics['acc']
        net.train()
        if hasattr(net,'partial_eval'): net.partial_eval()
        metrics = {
                'train_rank_accuracy':trainer.state.metrics['rank_acc'],
                'train_loss':trainer.state.metrics['loss'],
                'acc': trainer.state.metrics['acc'],
                'val_acc': evaluator.state.metrics['acc'],
                'val_rank_accuracy': evaluator.state.metrics['rank_acc'],
                'val_loss':evaluator.state.metrics['loss']
            }
        comet_log(
            metrics,
            experiment,
            epoch=trainer.state.epoch,
            step=trainer.state.epoch,
        )
        console_log(metrics,{},trainer.state.epoch)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_results(trainer):
        if trainer.state.iteration %100 == 0:
            metrics = {
                    'train_rank_accuracy':trainer.state.metrics['rank_acc'],
                    'train_loss':trainer.state.metrics['loss'],
                    'lr': scheduler.get_lr()
                }
            comet_log(
                metrics,
                experiment,
                step=trainer.state.iteration,
                epoch=trainer.state.epoch
            )
            console_log(
                metrics,
                {},
                trainer.state.epoch,
                step=trainer.state.iteration,
            )
    model_name = '{}_{}_{}'.format(args.model, args.premodel, args.attribute)
    if args.tag: model_name += f'_{args.tag}'
    handler = ModelCheckpoint(args.model_dir, model_name,
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
    net = SegRank(image_size=(244,244))
    x = torch.randn([3,244,244]).unsqueeze(0)
    y = torch.randn([3,244,244]).unsqueeze(0)
    fwd =  net(x,y)
    print(fwd)