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


class IrisDataset(Dataset):
    def __init__(self, dataset_dir='./dataset/RGB/session_1/', subset='train', img_size=480, modality='NIR'):
        """
        :param dataset_dir: directory containing the dataset
        :param subset: subset that we are working on ('train'/'test'/'valid')
        :param img_size: image size
        :param modality: NIR or RGB
        """
        super(IrisDataset, self).__init__()
        self.filenames = collections.defaultdict(list)
        self.img_size = img_size
        self.resize_img = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR)
        self.resize_gt = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST)
        self.subset = subset
        self.modality = modality
        self.data_path = dataset_dir + '/' + subset
        text_file = "{}/{}/{}.txt".format(dataset_dir, subset, subset)

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
        img = Image.open(self.data_path + '/img/' + filename + '.bmp')
        if self.modality == 'RGB':
            img = img.convert('RGB')
        gt = Image.open(self.data_path + '/gt/' + filename + '.png')

        # Resize the image and ground-truth
        img = self.resize_img(img)
        gt = self.resize_gt(gt)

        # Convert both to Pytorch tensors
        img = T.ToTensor()(img)  # [0, 1] normalization included
        gt = torch.from_numpy(np.asarray(gt))  # must use 'torch.from_numpy' otherwise all pixels become 0

        return img, gt


def visualize(seg_map, img):
    """
    Overlay a segmentation map with an input color image
    :param seg_map: segmentation map of size H x W as a Numpy array
    :param img: original image as a Numpy array. It can be grayscale (W x H) or RGB image (W x H x 3).
    :return: overlaid image
    """
    # Generate the segmentation map in the RGB color with the color code
    # Class 0 (background): Black
    # Class 1 (iris): Green
    if len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    COLOR_CODE = [[0, 0, 0], [0, 1, 0]]
    seg_map_rgb = np.zeros(img.shape)

    # Convert the segmentation map (with class IDs) to a RGB image
    for k in np.unique(seg_map):
        seg_map_rgb[seg_map == k] = COLOR_CODE[k]
    seg_map_rgb = (seg_map_rgb * 255).astype('uint8')

    # Super-impose the color segmentation map onto the original image
    overlaid_img = cv2.addWeighted(img, 1, seg_map_rgb, 0.5, 0)

    return overlaid_img


def split_segmentation_dataset(in_folder='./dataset/RGB/session_1', train_ratio=0.8):
    """
    Split a SEGMENTATION dataset into 3 subsets: train, test, and validation
    :param in_folder: input folder
    :param train_ratio: ratio for training
    """

    # Create empty folders an empty text files
    if not (os.path.exists(in_folder + os.sep + 'train')):
        os.mkdir(in_folder + os.sep + 'train')
        os.mkdir(in_folder + os.sep + 'train' + os.sep + 'img')
        os.mkdir(in_folder + os.sep + 'train' + os.sep + 'gt')
    if not (os.path.exists(in_folder + os.sep + 'test')):
        os.mkdir(in_folder + os.sep + 'test')
        os.mkdir(in_folder + os.sep + 'test' + os.sep + 'img')
        os.mkdir(in_folder + os.sep + 'test' + os.sep + 'gt')

    with open(in_folder + '/train/' + 'train.txt', 'w') as f:
        pass
    with open(in_folder + '/test/' + 'test.txt', 'w') as f:
        pass

    # Split images in each sub-folder into train & test subsets
    imgs = glob.glob(in_folder + os.sep + 'img' + '/*.bmp')
    num_imgs = len(imgs)
    num_train = int(train_ratio * num_imgs)

    # Build a train set and a test set
    train_idx = np.random.choice(num_imgs, num_train, replace=False)  # generate random indices
    train_set = list(itemgetter(*train_idx)(imgs))  # get the elements at the indices from the list
    test_set = list(set(imgs) - set(train_set)) + list(set(train_set) - set(
        imgs))  # take the remaining list. To check if two subset intersect each other: 1) bool(set(train_set) &
    # set(test_set)), and 2) list(set(train_set).intersection(test_set))

    # From the train & test lists, we copy the images files to the corresponding folders and write the file list to
    # the text file

    with open(in_folder + '/train/' + 'train.txt', 'a') as f:
        for img in train_set:
            new_img = img.replace('img', 'train/img')
            copyfile(img, new_img)
            gt = img.replace('img', 'masks_machine').replace('bmp', 'png')
            copyfile(gt, gt.replace('masks_machine', 'train/gt'))
            f.write(os.path.splitext(os.path.basename(img))[0] + '\n')  # get filename without extension
            print(f'Train: {img} done')

    with open(in_folder + '/test/' + 'test.txt', 'a') as f:
        for img in test_set:
            new_img = img.replace('img', 'test/img')
            copyfile(img, new_img)
            gt = img.replace('img', 'masks_machine').replace('bmp', 'png')
            copyfile(gt, gt.replace('masks_machine', 'test/gt'))
            f.write(os.path.splitext(os.path.basename(img))[0] + '\n')  # get filename without extension
            print(f'Test: {img} done')


def split_user_id(in_folder='./dataset/RGB/session_1'):
    imgs = glob.glob(in_folder + os.sep + 'img' + '/*.bmp')
    for img in imgs:
        filename = os.path.basename(img)
        _filename = filename.split('_')

        # Create a new folder for the current user if not existing
        if not (os.path.exists(in_folder + os.sep + 'recognition' + os.sep + 'all' + os.sep + _filename[0])):
            os.mkdir(in_folder + os.sep + 'recognition' + os.sep + 'all' + os.sep + _filename[0])

        copyfile(img, in_folder + os.sep + 'recognition' + os.sep + 'all' + os.sep + _filename[0] + os.sep + filename)
        print(img)


def split_recognition_dataset(in_folder='./dataset/RGB/session_1/recognition/all', train_ratio=0.8):
    # Get all sub folders in the input folder
    folders = next(os.walk(in_folder))[1]

    # Create new sub-folders in 'train'/'test' folders if not existing
    for folder in folders:
        if not (os.path.exists(in_folder.replace('all', 'train') + os.sep + folder)):
            os.mkdir(in_folder.replace('all', 'train') + os.sep + folder)
        if not (os.path.exists(in_folder.replace('all', 'test') + os.sep + folder)):
            os.mkdir(in_folder.replace('all', 'test') + os.sep + folder)

    count_train = 0
    count_test = 0

    for folder in folders:
        # Get all BMP files in the folder
        imgs = glob.glob(in_folder + os.sep + folder + '/*.bmp')

        # Build a train set and a test set
        num_imgs = len(imgs)
        num_train = int(train_ratio * num_imgs)
        count_train += num_train
        count_test += (num_imgs - num_train)
        train_idx = np.random.choice(num_imgs, num_train, replace=False)  # generate random indices
        train_set = list(itemgetter(*train_idx)(imgs))  # get the elements at the indices from the list
        test_set = list(set(imgs) - set(train_set)) + list(set(train_set) - set(imgs))  # take the remaining list

        for img in train_set:
            copyfile(img, img.replace('all', 'train'))
            print(img)

        for img in test_set:
            copyfile(img, img.replace('all', 'test'))
            print(img)

    print(f'Total number of training images: {count_train}. Total number of test images: {count_test}')


if __name__ == '__main__':
    # split_segmentation_dataset(in_folder='/data/iris_recognition/dataset/NIR/session_5')
    split_user_id(in_folder='/data/iris_recognition/dataset/RGB/session_2')
    # split_recognition_dataset(in_folder='/data/iris_recognition/dataset/NIR/session_5/recognition/all', train_ratio=0.8)
