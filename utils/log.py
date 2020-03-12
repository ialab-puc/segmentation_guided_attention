import json

def tb_log(train_metrics,val_metrics,writer,attribute, epoch):
    for key,metric_hash in train_metrics.items():
        writer.add_scalars(f'{attribute}/Training/{key}', metric_hash, epoch)
    for key,metric_hash in val_metrics.items():
        writer.add_scalars(f'{attribute}/Val/{key}', metric_hash, epoch)

def console_log(train_metrics,val_metrics, epoch, step=None):
    print(f"Results - Epoch: {epoch} - Step: {step}")
    print(json.dumps(train_metrics, indent=2))
    if val_metrics:
        print(f"Validation Results - Epoch: {epoch}")
        print(json.dumps(val_metrics, indent=2))

def comet_log(metrics, experiment, epoch=None, step=None):
    experiment.log_metrics(metrics, epoch=epoch, step=step)

def comet_image_log(image,image_name,experiment,epoch=None):
    experiment.log_image(image, name=image_name)