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
        """
        :param dataset_dir: directory containing the dataset
        :param subset: subset that we are working on ('train'/'test'/'valid')
        :param img_size: image size
        """
        super(LaneDataset, self).__init__()
        self.img_size = img_size
        self.resize_img = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR)
        self.resize_gt = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST)
        self.subset = subset
        #self.data_path = dataset_dir + '/' + subset
        self.data_path="segmentattention/train/dataset/test"
        #text_file = "{}/{}/{}.txt".format(dataset_dir, subset, subset)
        text_file = "segmentattention/train/dataset/test/test.txt"
        
        # Read the text file
        with open(text_file, 'r') as f:
            self.filenames = f.read().splitlines()
        print('Loaded {} subset with {} images'.format(subset, self.__len__()))

    def __len__(self):
        """
        :return: length of the dataset (i.e., number of images)
        """
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        # Read an image and its ground-truth
        img = Image.open(self.data_path + '/images/' + 'DSC_' + filename + '.JPG').convert('RGB')    
        gt = Image.open(self.data_path + '/groundtruth/' + 'DSC_' + filename + '.png')

        # Resize the image and ground-truth
        img = self.resize_img(img)
        gt = self.resize_gt(gt)

        # Convert image to tensor and normalize [0, 1]
        img = T.ToTensor()(img)  # [0, 1] normalization included

        # Convert ground-truth to a 2D tensor containing class indices
        gt = torch.from_numpy(np.array(gt, dtype=np.int64))

        return img, gt
