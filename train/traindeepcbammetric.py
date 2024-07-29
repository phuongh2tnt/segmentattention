import torch
from tqdm import tqdm
import numpy as np
import argparse
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from torchmetrics.functional import precision as torch_precision, recall as torch_recall, f1_score as torch_f1_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Resize
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
def iou(preds, labels):
    """Calculate Intersection over Union (IoU)."""
    intersection = np.logical_and(preds, labels)
    union = np.logical_or(preds, labels)
    return np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0

def custom_precision(preds, labels):
    """Calculate Precision."""
    true_positive = np.sum((preds == 1) & (labels == 1))
    false_positive = np.sum((preds == 1) & (labels == 0))
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0

def custom_recall(preds, labels):
    """Calculate Recall."""
    true_positive = np.sum((preds == 1) & (labels == 1))
    false_negative = np.sum((preds == 0) & (labels == 1))
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0

def custom_f1_score(preds, labels):
    """Calculate F1 Score."""
    prec = custom_precision(preds, labels)
    rec = custom_recall(preds, labels)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0

def calculate_metrics(logits, gt):
    """ Calculate precision, recall, F1-score and IoU """
    seg_maps = logits.argmax(dim=1)  # Changed axis to dim for PyTorch tensors
    gt = gt.cpu().numpy()
    seg_maps = seg_maps.cpu().numpy()
    
    iou_value = iou(seg_maps, gt)
    precision_score = torch_precision(torch.tensor(seg_maps), torch.tensor(gt), task='multiclass', average='macro', num_classes=2).item()
    recall_score = torch_recall(torch.tensor(seg_maps), torch.tensor(gt), task='multiclass', average='macro', num_classes=2).item()
    f1 = torch_f1_score(torch.tensor(seg_maps), torch.tensor(gt), task='multiclass', average='macro', num_classes=2).item()

    return iou_value, precision_score, recall_score, f1

def train_model(accumulation_steps=2):
    model.train()
    train_loss = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
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
        iou, precision, recall, f1 = calculate_metrics(logits, gt)
        total_iou += iou
        total_precision += precision
        total_recall += recall
        total_f1 += f1

    return train_loss / len(train_loader), total_iou / len(train_loader), total_precision / len(train_loader), total_recall / len(train_loader), total_f1 / len(train_loader)

def validate_model():
    model.eval()
    valid_loss = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0

    with torch.no_grad():
        for i, (img, gt) in enumerate(valid_loader):
            img, gt = img.to(device, dtype=torch.float), gt.to(device, dtype=torch.long)
            
            with autocast():
                logits = model(img)
                loss = loss_fn(logits, gt)
            
            valid_loss += loss.item()
            iou, precision, recall, f1 = calculate_metrics(logits, gt)
            total_iou += iou
            total_precision += precision
            total_recall += recall
            total_f1 += f1

    return valid_loss / len(valid_loader), total_iou / len(valid_loader), total_precision / len(valid_loader), total_recall / len(valid_loader), total_f1 / len(valid_loader)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Train a deep model for iris segmentation')
    args.add_argument('-d', '--dataset', default='dataset', type=str, help='Dataset folder')
    args.add_argument('-e', '--epochs', default=10, type=int, help='Number of epochs')
    args.add_argument('-b', '--batch-size', default=4, type=int, help='Batch size')
    args.add_argument('-i', '--img-size', default=480, type=int, help='Image size')
    args.add_argument('-c', '--checkpoint', default='segmentattention/train/checkpoints', type=str, help='Checkpoint folder')
    args.add_argument('-t', '--metric', default='iou', type=str, help='Metric for optimization')

    cmd_args = args.parse_args()
    device = setup_cuda()

    from utils.lane_dataset import LaneDataset

    train_dataset = LaneDataset(dataset_dir=cmd_args.dataset, subset='test', img_size=cmd_args.img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cmd_args.batch_size,
                                               shuffle=True,
                                               num_workers=6)

    valid_dataset = LaneDataset(dataset_dir=cmd_args.dataset, subset='test', img_size=cmd_args.img_size)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=cmd_args.batch_size,
                                               shuffle=False,
                                               num_workers=6)

    model = DeepLabV3_CBAM(
        n_classes=2  # Adjust the number of classes as needed
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    max_perf = 0
    for epoch in range(cmd_args.epochs):
        train_loss, train_iou, train_precision, train_recall, train_f1 = train_model()
        valid_loss, valid_iou, valid_precision, valid_recall, valid_f1 = validate_model()

        print(f'Epoch: {epoch} \tTraining IoU: {train_iou:.4f} \tPrecision: {train_precision:.4f} \tRecall: {train_recall:.4f} \tF1: {train_f1:.4f}')
        print(f'Validation IoU: {valid_iou:.4f} \tPrecision: {valid_precision:.4f} \tRecall: {valid_recall:.4f} \tF1: {valid_f1:.4f}')

        if valid_iou > max_perf:
            print(f'Validation IoU increased ({max_perf} --> {valid_iou}). Model saved')
            torch.save(model.state_dict(), '/content/drive/My Drive/AI/deepcbam/checkpoints/deepcbam_epoch_' + str(epoch) +
                       '_' + cmd_args.metric + '_{0:.4f}'.format(valid_iou) + '.pt')
            max_perf = valid_iou

    # Plotting metrics
    epochs_list = list(range(cmd_args.epochs))

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs_list, [train_iou for _ in range(cmd_args.epochs)], label='Train IoU')
    plt.plot(epochs_list, [valid_iou for _ in range(cmd_args.epochs)], label='Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('IoU vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('/content/drive/My Drive/AI/deepcbam/trainval_IOU.png')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_list, [train_precision for _ in range(cmd_args.epochs)], label='Train Precision')
    plt.plot(epochs_list, [valid_precision for _ in range(cmd_args.epochs)], label='Validation Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Precision vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('/content/drive/My Drive/AI/deepcbam/trainval_Precision.png')

    plt.subplot(2, 2, 3)
    plt.plot(epochs_list, [train_recall for _ in range(cmd_args.epochs)], label='Train Recall')
    plt.plot(epochs_list, [valid_recall for _ in range(cmd_args.epochs)], label='Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('/content/drive/My Drive/AI/deepcbam/trainval_Recall.png')

    plt.subplot(2, 2, 4)
    plt.plot(epochs_list, [train_f1 for _ in range(cmd_args.epochs)], label='Train F1-score')
    plt.plot(epochs_list, [valid_f1 for _ in range(cmd_args.epochs)], label='Validation F1-score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('/content/drive/My Drive/AI/deepcbam/trainval_F1.png')

    plt.tight_layout()
    plt.show()
