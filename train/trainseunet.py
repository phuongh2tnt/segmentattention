import os
import argparse
import torch
import numpy as np
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from optim import CrossEntropyLoss2d, adjust_learning_rate
from utils.lane_dataset import LaneDataset  # Ensure your dataset module is correctly located
from visualization import Dashboard, gray2rgb, gray2rgb_norm
from metric import dice_tensor
pathcheckpoints='/content/drive/My Drive/segattention/unet/unetse/checkpoints'
pathchweight='/content/drive/My Drive/segattention/unet/unetse/weight'
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='data', help='Path to the dataset')
parser.add_argument('--save_dir', type=str, default=pathchweight, help='Models are saved here')
parser.add_argument('--weight_dir', type=str, default=None, help='Path to pretrained weight')
parser.add_argument('--model', type=str, default='se_unet', help='Select model, e.g., unet, se_unet, densenet, se_densenet')
parser.add_argument('--reduction_ratio', type=int, default=None, help='Number of reduction ratio in SE block')
parser.add_argument('--growth_rate', type=int, default=16, help='Number of growth_rate in Denseblock')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--gpu_ids', type=str, default='0', help='GPU ids: e.g. 0  0,1,2, 0,2')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate')
parser.add_argument('--num_class', type=int, default=2, help='Number of segmentation classes')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--step_print', type=int, default=100, help='Frequency of showing training results')
parser.add_argument('--epochs_save', type=int, default=5, help='Frequency of saving weight at the end of epochs')
parser.add_argument('--init_features', type=int, default=32, help='Initial feature map in the first conv')
parser.add_argument('--network_depth', type=int, default=4, help='Number of network depth')
parser.add_argument('--bottleneck', type=int, default=5, help='Number of bottleneck layer')
parser.add_argument('--down_blocks', type=str, default=None, help='Number of down blocks')
parser.add_argument('--up_blocks', type=str, default=None, help='Number of up blocks')
parser.add_argument('--port', type=int, default=8097, help='Visdom port of the web display')
parser.add_argument('--visdom_env_name', type=str, default='SE_segmentation', help='Name of current environment in visdom')
parser.add_argument('--img_size', type=int, default=256, help='Image size for dataset')
parser.add_argument('--checkpoint', type=str, default=pathcheckpoints, help='Directory to save model checkpoints')
parser.add_argument('--metric', type=str, default='Dice', help='Metric to track for best performance')
args = parser.parse_args()

print(args)
str_ids = args.gpu_ids.split(',')
gpu_ids = [int(id) for id in str_ids if int(id) >= 0]

if args.down_blocks is not None:
    down_blocks = [int(block) for block in args.down_blocks.split(',')]

if args.up_blocks is not None:
    up_blocks = [int(block) for block in args.up_blocks.split(',')]

torch.cuda.set_device(gpu_ids[0])
board = Dashboard(args.port, args.visdom_env_name)

# Initialize the dataset and dataloader
train_dataset = LaneDataset(dataset_dir=args.dataset, subset='test', img_size=args.img_size)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)

valid_dataset = LaneDataset(dataset_dir=args.dataset, subset='test', img_size=args.img_size)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6)

print('Create model')

if args.model == 'unet':
    from seunet import UNet
    model = torch.nn.DataParallel(
        UNet(num_classes=args.num_class, init_features=args.init_features, network_depth=args.network_depth,
             bottleneck_layers=args.bottleneck, reduction_ratio=None), device_ids=gpu_ids)

if args.model == 'se_unet':
    from seunet import UNet
    model = torch.nn.DataParallel(
        UNet(num_classes=args.num_class, init_features=args.init_features, network_depth=args.network_depth,
             bottleneck_layers=args.bottleneck, reduction_ratio=args.reduction_ratio), device_ids=gpu_ids)

if args.model == 'densenet':
    from seunet import FCDenseNet
    model = torch.nn.DataParallel(
        FCDenseNet(in_channels=3, down_blocks=down_blocks, up_blocks=up_blocks, bottleneck_layers=args.bottleneck,
                   growth_rate=args.growth_rate, out_chans_first_conv=args.init_features, n_classes=args.num_class),
        device_ids=gpu_ids)

if args.model == 'se_densenet':
    from seunet import FCDenseNet
    model = torch.nn.DataParallel(
        FCDenseNet(reduction_ratio=args.reduction_ratio, in_channels=3, down_blocks=down_blocks, up_blocks=up_blocks,
                   bottleneck_layers=args.bottleneck, growth_rate=args.growth_rate, out_chans_first_conv=args.init_features,
                   n_classes=args.num_class), device_ids=gpu_ids)

if args.weight_dir is not None:
    print(f'Load weight from {args.weight_dir}')
    model.load_state_dict(torch.load(args.weight_dir))

model.cuda(gpu_ids[0])
print(model)

weight = torch.ones(args.num_class)
criterion = CrossEntropyLoss2d(weight.cuda())
optimizer = Adam(model.parameters(), lr=args.lr)

print('Start training')

max_perf = 0
for epoch in range(1, args.num_epochs + 1):
    adjust_learning_rate(optimizer, args.lr, epoch)
    model.train()

    epoch_train_loss = []
    epoch_train_perf = []

    for step, (images, targets) in enumerate(train_loader):
        images = images.cuda(gpu_ids[0], non_blocking=True)
        targets = targets.long().cuda(gpu_ids[0], non_blocking=True)

        inputs = Variable(images, requires_grad=True)
        targets = Variable(targets)
        outputs = model(inputs)

        optimizer.zero_grad()
        loss = criterion(outputs, targets[:, 0])
        loss.backward()
        optimizer.step()

        epoch_train_loss.append(loss.item())
        performance = dice_tensor(outputs.max(1)[1], targets[:, 0])
        epoch_train_perf.append(performance.item())

        step_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
        step_train_perf = sum(epoch_train_perf) / len(epoch_train_perf)

        if step % args.step_print == 0:
            print(f'train loss: {step_train_loss:.7f}, train {args.metric}: {step_train_perf:.7f} (step: {step})')

    train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
    train_perf = sum(epoch_train_perf) / len(epoch_train_perf)
    board.plot("loss", "train", epoch, train_loss)
    board.plot(args.metric, "train", epoch, train_perf)

    model.eval()
    epoch_val_loss = []
    epoch_val_perf = []

    for step, (images, targets) in enumerate(valid_loader):
        images = images.cuda(gpu_ids[0], non_blocking=True)
        targets = targets.long().cuda(gpu_ids[0], non_blocking=True)

        with torch.no_grad():
            inputs = Variable(images)
            targets = Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets[:, 0])

            performance = dice_tensor(outputs.max(1)[1], targets[:, 0])
            epoch_val_loss.append(loss.item())
            epoch_val_perf.append(performance.item())

    val_loss = sum(epoch_val_loss) / len(epoch_val_loss)
    val_perf = sum(epoch_val_perf) / len(epoch_val_perf)

    print(f'Epoch: {epoch} \tTraining {args.metric}: {train_perf:.4f} \tValid {args.metric}: {val_perf:.4f}')
    board.plot("loss", "val", epoch, val_loss)
    board.plot(args.metric, "val", epoch, val_perf)

    if val_perf > max_perf:
        print(f'Valid {args.metric} increased ({max_perf:.4f} --> {val_perf:.4f}). Model saved')
        if not os.path.exists(args.checkpoint):
            os.makedirs(args.checkpoint)
        torch.save(model.state_dict(), f'{args.checkpoint}/unet_epoch_{epoch}_{args.metric}_{val_perf:.4f}.pt')
        max_perf = val_perf
