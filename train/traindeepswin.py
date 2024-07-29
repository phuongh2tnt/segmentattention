import torch
from tqdm import tqdm
import numpy as np
import argparse
from torch.optim import Adam
import utils.metrics as metrics

# Import your SwinDeepLabV3 class
from deepswin import SwinDeepLabV3 

# Setup CUDA
def setup_cuda():
    # Setting seeds for reproducibility
    seed = 50
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_model():
    """
    Train the model over a single epoch
    :return: training loss and segmentation performance
    """
    model.train()
    train_loss = 0.0
    performance = 0

    for i, (img, gt) in enumerate(tqdm(train_loader, ncols=80, desc='Training')):
        optimizer.zero_grad()
        img, gt = img.to(device, dtype=torch.float), gt.to(device, dtype=torch.long)
        logits = model(img)
        # Upsample logits to match the size of gt
        logits = torch.nn.functional.interpolate(logits, size=gt.shape[1:], mode='bilinear', align_corners=False)
        loss = loss_fn(logits, gt)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        seg_maps = logits.cpu().detach().numpy().argmax(axis=1)
        gt = gt.cpu().detach().numpy()
        performance += getattr(metrics, cmd_args.metric)(seg_maps, gt)

    return train_loss / len(train_loader), performance / len(train_loader)

def validate_model():
    """
    Validate the model over a single epoch
    :return: validation loss and segmentation performance
    """
    model.eval()
    valid_loss = 0.0
    performance = 0

    with torch.no_grad():
        for i, (img, gt) in enumerate(valid_loader):
            img, gt = img.to(device, dtype=torch.float), gt.to(device, dtype=torch.long)
            logits = model(img)
            # Upsample logits to match the size of gt
            logits = torch.nn.functional.interpolate(logits, size=gt.shape[1:], mode='bilinear', align_corners=False)
            loss = loss_fn(logits, gt)
            valid_loss += loss.item()
            seg_maps = logits.cpu().detach().numpy().argmax(axis=1)
            gt = gt.cpu().detach().numpy()
            performance += getattr(metrics, cmd_args.metric)(seg_maps, gt)

    return valid_loss / len(valid_loader), performance / len(valid_loader)

if __name__ == "__main__":
    # 1. Parse the command arguments
    args = argparse.ArgumentParser(description='Train a deep model for iris segmentation')
    args.add_argument('-d', '--dataset', default='dataset', type=str, help='Dataset folder')
    args.add_argument('-e', '--epochs', default=100, type=int, help='Number of epochs')
    args.add_argument('-b', '--batch-size', default=8, type=int, help='Batch size')
    args.add_argument('-i', '--img-size', default=480, type=int, help='Image size')
    args.add_argument('-c', '--checkpoint', default='segmentattention/train/checkpoints', type=str, help='Checkpoint folder')
    args.add_argument('-t', '--metric', default='iou', type=str, help='Metric for optimization')

    cmd_args = args.parse_args()
    device = setup_cuda()

    # 2. Load the dataset
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

    # 3. Create a segmentation model using SwinDeepLabV3
    model = SwinDeepLabV3(
        H=cmd_args.img_size,
        W=cmd_args.img_size,
        ch=3,  # number of input channels (RGB)
        C=64,  # number of embedding channels (adjustable)
        num_class=2,  # number of output classes (2 for binary classification)
    ).to(device)

    # 4. Specify loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # 5. Start training the model
    max_perf = 0
    for epoch in range(cmd_args.epochs):
        # 5.1. Train the model over a single epoch
        train_loss, train_perf = train_model()

        # 5.2. Validate the model
        valid_loss, valid_perf = validate_model()

        print('Epoch: {} \tTraining {}: {:.4f} \tValid {}: {:.4f}'.format(epoch, cmd_args.metric, train_perf,
                                                                          cmd_args.metric, valid_perf))

        # 5.3. Save the model if the validation performance is increasing
        if valid_perf > max_perf:
            print('Valid {} increased ({:.4f} --> {:.4f}). Model saved'.format(cmd_args.metric, max_perf, valid_perf))
            torch.save(model.state_dict(), cmd_args.checkpoint + '/swindeeplabv3_epoch_' + str(epoch) +
                       '_' + cmd_args.metric + '_{0:.4f}'.format(valid_perf) + '.pt')
            max_perf = valid_perf
