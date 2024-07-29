import numpy as np
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score

def accuracy(seg_map, gt):
    return np.mean(seg_map == gt)

def iou(seg_map, gt):
    return jaccard_score(gt.flatten(), seg_map.flatten(), average='macro')

def f1(seg_map, gt):
    return f1_score(gt.flatten(), seg_map.flatten(), average='macro')

def precision(seg_map, gt):
    return precision_score(gt.flatten(), seg_map.flatten(), average='macro')

def recall(seg_map, gt):
    return recall_score(gt.flatten(), seg_map.flatten(), average='macro')
