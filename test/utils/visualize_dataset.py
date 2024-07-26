from utils.iris_dataset import visualize
import numpy as np
from PIL import Image

if __name__ == '__main__':
    dataset_dir = 'dataset/'
    subset = 'test'

    with open(dataset_dir + subset + '/' + subset + '.txt', 'r') as f:
        filenames = f.read().splitlines()

    for k in range(len(filenames)):
        filename = filenames[k]
        img = Image.open(dataset_dir + subset + '/images/' + filename + '.bmp')
        rgb_img = img.convert('RGB')
        rgb_img = np.asarray(rgb_img)

        gt = Image.open(dataset_dir + subset + '/groundtruths/' + filename + '.PNG')
        gt = np.asarray(gt)

        overlaid_img = visualize(gt, rgb_img)
        Image.fromarray(overlaid_img).save(filename + '.jpg')
        print('{}: {} done'.format(filename, k))
