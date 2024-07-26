"""
UOW, 14/07/2022
"""
import torch
import argparse
import numpy as np
import os
import torchvision.transforms as T
import utils.metrics as metrics
import os
import glob
from utils.iris_dataset import visualize
from PIL import Image, ImageOps
from timeit import default_timer as timer


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


def predict(in_file, img_size=480):
    """
    :param in_file: image file
    :param img_size: image size
    """
    model.eval()

    # Pre-process input image and its ground-truth
    img = Image.open(in_file).convert('RGB')  # convert to RGB image
    W, H = img.size
  
    img_resized = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR)(img)
    img_tensor = T.ToTensor()(img_resized).to(device, dtype=torch.float).unsqueeze(dim=0)

    # Perform a forward pass
    logits = model(img_tensor)

    # Upsample the output to the original resolution for a better visualization
    logits = torch.nn.Upsample(size=(H, W), mode='bilinear', align_corners=False)(logits)

    # Measure the segmentation performance
    seg_map = logits.cpu().detach().numpy().argmax(axis=1)
    seg_map = seg_map.squeeze()  # 'squeeze' used to remove the first dimension of 1 (i.e., batch size)

    overlaid = visualize(seg_map, np.array(img))
    overlaid = Image.fromarray(overlaid)
    overlaid.save(cmd_args.output + os.sep + os.path.basename(in_file))

    print('File: ' + os.path.basename(in_file) + ' done')


if __name__ == "__main__":

    # 1. Parse the command arguments
    args = argparse.ArgumentParser(description='Test a post-mortem iris segmentation model')
    args.add_argument('-i', '--input', default=None, type=str)
    args.add_argument('-w', '--weights', default='weights/unet_resnet18_epoch_7_iou_0.9624.pt', type=str,
                      help='Trained weights')
    args.add_argument('-o', '--output', default='outputs', type=str, help='Output folder')
    cmd_args = args.parse_args()

    device = setup_cuda()

    # 2. Create a segmentation model, then load the trained weights
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name='resnet18', in_channels=3, classes=2).to(device)
	
    model.load_state_dict(torch.load(cmd_args.weights, device))
    print('The segmentation model has been loaded.')

    # 3. Perform segmentation
    predict(cmd_args.input)
