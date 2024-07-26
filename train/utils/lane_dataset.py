"""
UOW, 14/07/2022
"""
import collections
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import cv2
import torch
import random
import math
import os
import glob
from operator import itemgetter
from shutil import copyfile
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
        self.filenames = collections.defaultdict(list)
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

        # Read an image and its ground-truth toi khong viet hoa
        img = Image.open(self.data_path + '/images/' + 'DSC_' + filename + '.JPG').convert('RGB')    
        #bài của thầy thanh
        #gt = Image.open(self.data_path + '/groundtruth/' + filename + '_lane.png')
        
        #gt = Image.open(self.data_path + '/groundtruth/' + filename + '_lane.png')
        #bai tom
        gt = Image.open(self.data_path + '/groundtruth/' 'DSC_' + filename + '.png')
        # Resize the image and ground-truth
        img = self.resize_img(img)
        gt = self.resize_gt(gt)

        # Convert both to Pytorch tensors
        img = T.ToTensor()(img)  # [0, 1] normalization included
        gt = torch.from_numpy(np.asarray(gt))  # must use 'torch.from_numpy' otherwise all pixels become 0

        return img, gt

