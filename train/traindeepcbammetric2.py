import torch
import timm
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from utils.metric2 import IoU, accuracy, precision, f1_score, recall  # Assuming you have these functions in your metric.py
from utils.lane_dataset import LaneDataset  # Assuming this is where LaneDataset is defined
import argparse
from torch.cuda.amp import GradScaler, autocast

# Setup CUDA
def setup_cuda():
    seed = 50
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Training function
def train_model(accumulation_steps=2):
    model.train()
    train_loss = 0.0
    train_iou, train_acc, train_pre, train_rec, train_f1 = 0, 0, 0, 0, 0
    scaler = GradScaler()

    optimizer.zero_grad()
    for i, (img, gt) in enumerate(tqdm(train_loader, ncols=80, desc='Training')):
        img, gt = img.to(device, dtype=torch.float), gt.to(device, dtype=torch.long)
        
        with autocast():
            logits = model(img)
            loss = loss_fn(logits, gt) / accumulation_steps
        
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        train_loss += loss.item() * accumulation_steps
        prediction = logits.argmax(axis=1)
        train_iou += IoU(prediction, gt)
        train_acc += accuracy(prediction, gt)
        train_pre += precision(prediction, gt)
        train_rec += recall(prediction, gt)
        train_f1 += f1_score(prediction, gt)

    return train_loss / len(train_loader), train_iou / len(train_loader), train_acc / len(train_loader), \
           train_pre / len(train_loader), train_rec / len(train_loader), train_f1 / len(train_loader)

# Validation function
def validate_model():
    model.eval()
    valid_loss = 0.0
    val_iou, val_acc, val_pre, val_rec, val_f1 = 0, 0, 0, 0, 0

    with torch.no_grad():
        for i, (img, gt) in enumerate(tqdm(valid_loader, ncols=80, desc='Validating')):
            img, gt = img.to(device, dtype=torch.float), gt.to(device, dtype=torch.long)
            
            with autocast():
                logits = model(img)
                loss = loss_fn(logits, gt)
            
            valid_loss += loss.item()
            prediction = logits.argmax(axis=1)
            val_iou += IoU(prediction, gt)
            val_acc += accuracy(prediction, gt)
            val_pre += precision(prediction, gt)
            val_rec += recall(prediction, gt)
            val_f1 += f1_score(prediction, gt)

    return valid_loss / len(valid_loader), val_iou / len(valid_loader), val_acc / len(valid_loader), \
           val_pre / len(valid_loader), val_rec / len(valid_loader), val_f1 / len(valid_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a deep model for iris segmentation')
    parser.add_argument('-d', '--dataset', default='dataset', type=str, help='Dataset folder')
    parser.add_argument('-e', '--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('-b', '--batch-size', default=4, type=int, help='Batch size')  # Adjusted batch size
    parser.add_argument('-i', '--img-size', default=480, type=int, help='Image size')
    parser.add_argument('-c', '--checkpoint', default='segmentattention/train/checkpoints', type=str, help='Checkpoint folder')
    parser.add_argument('-t', '--metric', default='iou', type=str, help='Metric for optimization')

    cmd_args = parser.parse_args()
    device = setup_cuda()

    train_dataset = LaneDataset(dataset_dir=cmd_args.dataset, subset='test', img_size=cmd_args.img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cmd_args.batch_size, shuffle=True, num_workers=6)

    valid_dataset = LaneDataset(dataset_dir=cmd_args.dataset, subset='test', img_size=cmd_args.img_size)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=cmd_args.batch_size, shuffle=False, num_workers=6)

    model = DeepLabV3_CBAM(n_classes=2).to(device)  # Adjust the number of classes as needed

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    train_metrics = {'loss': [], 'iou': [], 'acc': [], 'pre': [], 'rec': [], 'f1': []}
    val_metrics = {'loss': [], 'iou': [], 'acc': [], 'pre': [], 'rec': [], 'f1': []}

    max_perf = 0
    for epoch in range(cmd_args.epochs):
        train_loss, train_iou, train_acc, train_pre, train_rec, train_f1 = train_model()
        val_loss, val_iou, val_acc, val_pre, val_rec, val_f1 = validate_model()

        print('Epoch: {} \tTraining {}: {:.4f} \tValid {}: {:.4f}'.format(epoch, cmd_args.metric, train_iou, cmd_args.metric, val_iou))

        train_metrics['loss'].append(train_loss)
        train_metrics['iou'].append(train_iou)
        train_metrics['acc'].append(train_acc)
        train_metrics['pre'].append(train_pre)
        train_metrics['rec'].append(train_rec)
        train_metrics['f1'].append(train_f1)

        val_metrics['loss'].append(val_loss)
        val_metrics['iou'].append(val_iou)
        val_metrics['acc'].append(val_acc)
        val_metrics['pre'].append(val_pre)
        val_metrics['rec'].append(val_rec)
        val_metrics['f1'].append(val_f1)

        if val_iou > max_perf:
            print('Valid {} increased ({:.4f} --> {:.4f}). Model saved'.format(cmd_args.metric, max_perf, val_iou))
            torch.save(model.state_dict(), cmd_args.checkpoint + '/deeplabv3_cbam_epoch_' + str(epoch) + '_' + cmd_args.metric + '_{0:.4f}'.format(val_iou) + '.pt')
            max_perf = val_iou

    epochs_range = range(cmd_args.epochs)
    for metric_name, train_values in train_metrics.items():
        plt.figure()
        plt.plot(epochs_range, train_values, label=f'Training {metric_name}')
        plt.plot(epochs_range, val_metrics[metric_name], label=f'Validation {metric_name}')
        plt.xlabel('Epochs')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name.capitalize()} vs. Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{cmd_args.checkpoint}/{metric_name}.png')
        plt.show()
