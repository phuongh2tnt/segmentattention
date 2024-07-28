import collections
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import torch
import warnings

warnings.filterwarnings('ignore')


class LaneDataset(Dataset):
    def __init__(self, dataset_dir='segmentattention/train/dataset/', subset='test', img_size=480):
        super(LaneDataset, self).__init__()
        self.img_size = img_size
        self.resize_img = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR)
        self.resize_gt = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST)
        self.subset = subset
        self.data_path = "segmentattention/train/dataset/test"
        text_file = "segmentattention/train/dataset/test/test.txt"

        with open(text_file, 'r') as f:
            self.filenames = f.read().splitlines()
        print('Loaded {} subset with {} images'.format(subset, self.__len__()))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        img = Image.open(self.data_path + '/images/' + 'DSC_' + filename + '.JPG').convert('RGB')
        gt = Image.open(self.data_path + '/groundtruth/' + 'DSC_' + filename + '.png')

        img = self.resize_img(img)
        gt = self.resize_gt(gt)

        img = T.ToTensor()(img)  # [0, 1] normalization included
        gt = torch.from_numpy(np.array(gt, dtype=np.int64))  # Ensure this is a 2D tensor with class indices

        return img, gt

