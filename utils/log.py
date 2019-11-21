import json

def epoch_log(train_metrics,val_metrics,writer,attribute, epoch):
    for key,metric_hash in train_metrics.items():
        writer.add_scalars(f'{attribute}/Training/{key}', metric_hash, epoch)
    for key,metric_hash in val_metrics.items():
        writer.add_scalars(f'{attribute}/Val/{key}', metric_hash, epoch)
    print(f"Training Results - Epoch: {epoch}")
    print(json.dumps(train_metrics, indent=2))
    print(f"Validation Results - Epoch: {epoch}")
    print(json.dumps(val_metrics, indent=2))