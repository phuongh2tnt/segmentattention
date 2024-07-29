import torch
from tqdm import tqdm
import numpy as np
import argparse
from torch.optim import Adam
import utils.metrics as metrics
from torch.cuda.amp import GradScaler, autocast

# Import your DeepLabV3_CBAM class
from deepcbam import DeepLabV3_CBAM  # 

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

def train_model(accumulation_steps=2):
    model.train()
    train_loss = 0.0
    performance = 0
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
        seg_maps = logits.cpu().detach().numpy().argmax(axis=1)
        gt = gt.cpu().detach().numpy()
        performance += getattr(metrics, cmd_args.metric)(seg_maps, gt)

    return train_loss / len(train_loader), performance / len(train_loader)

def validate_model():
    model.eval()
    valid_loss = 0.0
    performance = 0

    with torch.no_grad():
        for i, (img, gt) in enumerate(valid_loader):
            img, gt = img.to(device, dtype=torch.float), gt.to(device, dtype=torch.long)
            
            with autocast():
                logits = model(img)
                loss = loss_fn(logits, gt)
            
            valid_loss += loss.item()
            seg_maps = logits.cpu().detach().numpy().argmax(axis=1)
            gt = gt.cpu().detach().numpy()
            performance += getattr(metrics, cmd_args.metric)(seg_maps, gt)

    return valid_loss / len(valid_loader), performance / len(valid_loader)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Train a deep model for iris segmentation')
    args.add_argument('-d', '--dataset', default='dataset', type=str, help='Dataset folder')
    args.add_argument('-e', '--epochs', default=100, type=int, help='Number of epochs')
    args.add_argument('-b', '--batch-size', default=4, type=int, help='Batch size')  # Adjusted batch size
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
        _, train_perf = train_model()

        _, valid_perf = validate_model()

        print('Epoch: {} \tTraining {}: {:.4f} \tValid {}: {:.4f}'.format(epoch, cmd_args.metric, train_perf,
                                                                          cmd_args.metric, valid_perf))

        if valid_perf > max_perf:
            print('Valid {} increased ({:.4f} --> {:.4f}). Model saved'.format(cmd_args.metric, max_perf, valid_perf))
            torch.save(model.state_dict(), cmd_args.checkpoint + '/deeplabv3_cbam_epoch_' + str(epoch) +
                       '_' + cmd_args.metric + '_{0:.4f}'.format(valid_perf) + '.pt')
            max_perf = valid_perf
