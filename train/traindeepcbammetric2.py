import torch
import timm
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import argparse
from torch.cuda.amp import GradScaler, autocast
from utils.metric2 import accuracy, iou, f1, precision, recall  # Import your metrics
from deepcbam import DeepLabV3_CBAM

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
    train_metrics = {'iou': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
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
        prediction = logits.argmax(axis=1).cpu().numpy()
        gt = gt.cpu().numpy()
        train_metrics['iou'] += iou(prediction, gt)
        train_metrics['accuracy'] += accuracy(prediction, gt)
        train_metrics['precision'] += precision(prediction, gt)
        train_metrics['recall'] += recall(prediction, gt)
        train_metrics['f1'] += f1(prediction, gt)

    for key in train_metrics:
        train_metrics[key] /= len(train_loader)

    return train_loss / len(train_loader), train_metrics

# Validation function
def validate_model():
    model.eval()
    valid_loss = 0.0
    val_metrics = {'iou': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}

    with torch.no_grad():
        for i, (img, gt) in enumerate(tqdm(valid_loader, ncols=80, desc='Validating')):
            img, gt = img.to(device, dtype=torch.float), gt.to(device, dtype=torch.long)
            
            with autocast():
                logits = model(img)
                loss = loss_fn(logits, gt)
            
            valid_loss += loss.item()
            prediction = logits.argmax(axis=1).cpu().numpy()
            gt = gt.cpu().numpy()
            val_metrics['iou'] += iou(prediction, gt)
            val_metrics['accuracy'] += accuracy(prediction, gt)
            val_metrics['precision'] += precision(prediction, gt)
            val_metrics['recall'] += recall(prediction, gt)
            val_metrics['f1'] += f1(prediction, gt)

    for key in val_metrics:
        val_metrics[key] /= len(valid_loader)

    return valid_loss / len(valid_loader), val_metrics

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

    train_history = {'loss': [], 'iou': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    val_history = {'loss': [], 'iou': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    max_perf = 0
    for epoch in range(cmd_args.epochs):
        train_loss, train_metrics = train_model()
        val_loss, val_metrics = validate_model()

        print(f'Epoch: {epoch} \tTraining {cmd_args.metric}: {train_metrics[cmd_args.metric]:.4f} \tValid {cmd_args.metric}: {val_metrics[cmd_args.metric]:.4f}')

        train_history['loss'].append(train_loss)
        val_history['loss'].append(val_loss)
        for key in train_metrics:
            train_history[key].append(train_metrics[key])
            val_history[key].append(val_metrics[key])

        if val_metrics[cmd_args.metric] > max_perf:
            print(f'Valid {cmd_args.metric} increased ({max_perf:.4f} --> {val_metrics[cmd_args.metric]:.4f}). Model saved')
            torch.save(model.state_dict(), f"{cmd_args.checkpoint}/deeplabv3_cbam_epoch_{epoch}_{cmd_args.metric}_{val_metrics[cmd_args.metric]:.4f}.pt")
            max_perf = val_metrics[cmd_args.metric]

    epochs_range = range(cmd_args.epochs)
    for metric_name in train_history:
        plt.figure()
        plt.plot(epochs_range, train_history[metric_name], label=f'Training {metric_name}')
        plt.plot(epochs_range, val_history[metric_name], label=f'Validation {metric_name}')
        plt.xlabel('Epochs')
        plt.ylabel(metric_name.capitalize())
        plt.title(f'{metric_name.capitalize()} vs. Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{cmd_args.checkpoint}/{metric_name}.png')
        plt.show()
