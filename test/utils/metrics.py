import numpy as np
from sklearn.metrics import jaccard_score, f1_score


def accuracy(seg_map, gt):
    """
    Calculate pixel accuracy
    :param seg_map: segmentation map as a 2D array of size [H, W]
    :param gt: ground-truth as a 2D array of size [H, W]
    :return: pixel accuracy
    """
    return np.mean(seg_map == gt)


def iou(seg_map, gt):
    """
    Calculate mean IoU (a.k.a., Jaccard Index) of an individual segmentation map. Note that, for the whole dataset, we
    must take average the mIoUs
    :param seg_map: segmentation map as a 2D array of size [H, W]
    :param gt: ground-truth as a 2D array of size [H, W]
    :return: mIoU
    """
    # 'micro': Calculate metrics globally by counting the total TP, FN, and FP
    # 'macro': calculate metrics for each label, and find their unweighted mean.
    return jaccard_score(gt.flatten(), seg_map.flatten(), average='macro')


def f1(seg_map, gt):
    """
    Calculate F-measure (a.k.a., Dice Coefficient, F1-score)
    :param seg_map: segmentation map as a 2D array of size [H, W]
    :param gt: ground-truth as a 2D array of size [H, W]
    :return: F-measure
    """
    return f1_score(gt.flatten(), seg_map.flatten(), average='macro')
