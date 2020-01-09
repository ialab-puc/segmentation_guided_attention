import json

def tb_log(train_metrics,val_metrics,writer,attribute, epoch):
    for key,metric_hash in train_metrics.items():
        writer.add_scalars(f'{attribute}/Training/{key}', metric_hash, epoch)
    for key,metric_hash in val_metrics.items():
        writer.add_scalars(f'{attribute}/Val/{key}', metric_hash, epoch)

def console_log(train_metrics,val_metrics, epoch):
    print(f"Results - Epoch: {epoch}")
    print(json.dumps(train_metrics, indent=2))
    if val_metrics:
        print(f"Validation Results - Epoch: {epoch}")
        print(json.dumps(val_metrics, indent=2))

def comet_log(metrics, epoch, experiment):
    experiment.log_metrics(metrics, epoch=epoch)