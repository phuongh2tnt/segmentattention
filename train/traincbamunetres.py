import torch
from tqdm import tqdm
import numpy as np
import argparse
from torch.optim import Adam
import utils.metrics as metrics
from torch.cuda.amp import GradScaler, autocast
from utils.lanedatasetv2 import LaneDataset
from rescbam import resnet18_cbam, resnet34_cbam, resnet50_cbam, resnet101_cbam, resnet152_cbam  # Import your CBAM-ResNet models
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
    # 1. Parse the command arguments
    args = argparse.ArgumentParser(description='Train a deep model for lane segmentation')
    args.add_argument('-d', '--dataset', default='dataset', type=str, help='Dataset folder')
    args.add_argument('-e', '--epochs', default=100, type=int, help='Number of epochs')
    args.add_argument('-b', '--batch-size', default=8, type=int, help='Batch size')
    args.add_argument('-i', '--img-size', default=480, type=int, help='Image size')
    args.add_argument('-c', '--checkpoint', default='checkpoints', type=str, help='Checkpoint folder')
    args.add_argument('-t', '--metric', default='iou', type=str, help='Metric for optimization')

    cmd_args = args.parse_args()
    device = setup_cuda()

    # 2. Load the dataset
    train_dataset = LaneDataset(dataset_dir=cmd_args.dataset, subset='train', img_size=cmd_args.img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cmd_args.batch_size, shuffle=True, num_workers=6)

    valid_dataset = LaneDataset(dataset_dir=cmd_args.dataset, subset='valid', img_size=cmd_args.img_size)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=cmd_args.batch_size, shuffle=False, num_workers=6)

    # 3. Create a segmentation model
   
    model = resnet18_cbam(pretrained=True).to(device)

    # 4. Specify loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # 5. Start training the model
    max_perf = 0
    for epoch in range(cmd_args.epochs):
        # 5.1. Train the model over a single epoch
        model.train()
        train_loss = 0.0
        train_perf = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        # Compute average loss
        train_loss = train_loss / len(train_loader.dataset)

        # 5.2. Validate the model
        model.eval()
        valid_loss = 0.0
        valid_perf = 0.0
        with torch.no_grad():
            for images, targets in valid_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, targets)
                valid_loss += loss.item() * images.size(0)

        # Compute average loss
        valid_loss = valid_loss / len(valid_loader.dataset)

        print('Epoch: {} \tTraining Loss: {:.4f} \tValidation Loss: {:.4f}'.format(epoch, train_loss, valid_loss))

        # 5.3. Save the model if the validation performance is increasing
        if valid_perf > max_perf:
            print('Validation performance increased. Model saved')
            torch.save(model.state_dict(), cmd_args.checkpoint + '/unet_resnet18_epoch_' + str(epoch) + '_perf_{0:.4f}'.format(valid_perf) + '.pt')
            max_perf = valid_perf
