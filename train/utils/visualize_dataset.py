import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

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
    overlaid_img = cv2.addWeighted(img, 1, seg_map_rgb, 0.9, 0)

    return overlaid_img
	
	
if __name__ == '__main__':
    dataset_dir = 'dataset/'
    subset = 'test'

    with open(dataset_dir + subset + '/' + subset + '.txt', 'r') as f:
        filenames = f.read().splitlines()

    for filename in tqdm(filenames):        
        img = Image.open(dataset_dir + subset + '/images/' + filename + '.jpg')
        rgb_img = img.convert('RGB')
        rgb_img = np.asarray(rgb_img)

        gt = Image.open(dataset_dir + subset + '/groundtruth/' + filename + '_lane.png')
        gt = np.asarray(gt)

        overlaid_img = visualize(gt, rgb_img)
        Image.fromarray(overlaid_img).save(dataset_dir + subset + '/visualization/' + filename + '.jpg')
