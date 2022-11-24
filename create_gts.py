import os
import cv2
import numpy as np
from glob import glob
from natsort import natsorted

from scipy.ndimage.filters import gaussian_filter

from hydra.utils import to_absolute_path as abs_path


def minmax(img):
    min = img.min()
    max = img.max()
    img = (img - min) / (max - min)
    return img

def edge_check(mask, result, coord, r):
    xmin, xmax, ymin, ymax = coord[0]-r, coord[0]+r+1, coord[1]-r, coord[1]+r+1
    if xmin < 0:
        mask = mask[-xmin:]
        xmin = 0
    elif xmax > result.shape[0]:
        mask = mask[:-(xmax - result.shape[0])]
        xmax = result.shape[0]
    if ymin < 0:
        mask = mask[:, -ymin:]
        ymin = 0
    elif ymax > result.shape[1]:
        mask = mask[:, :-(ymax - result.shape[1])]
        ymax = result.shape[1]
    return xmin, xmax, ymin, ymax, mask

def gaus_filter(img, kernel_size, sigma):
    pad_size = int(kernel_size - 1 / 2)
    img_t = np.pad(img, (pad_size, pad_size), 'constant')  # zero padding(これしないと正規化後、画像端付近の尤度だけ明るくなる)
    img_t  = cv2.GaussianBlur(img_t, ksize=(kernel_size, kernel_size), sigmaX=sigma)
    img_t = img_t[pad_size:-pad_size, pad_size:-pad_size]  # remove padding
    return img_t


lr = 10 
sigma = 6 
mr = 12  
lm = np.zeros((lr*2+1, lr*2+1))
lm[lr, lr] = 255
lm = gaussian_filter(lm, sigma=sigma, mode="constant")
lm = minmax(lm)

# Create validation gts heatmap 
for index, text in enumerate(natsorted(glob(abs_path('dataset/val_txt/*')))):
    m = int(text.split('video')[1].split('.txt')[0])
    save_dir = abs_path('dataset/val_gts') + "/sequence{}".format(str(m).zfill(3))
    os.makedirs(abs_path(save_dir)) if os.path.isdir(abs_path(save_dir)) is False else None
    data = np.loadtxt(text)
    print(data)
    for i in range(12):
        temp = data[data[:, 0] == i, 2:4]*2    
        img = np.zeros([1080, 1920])
        for i2 in temp:
            i2 = i2.astype(np.int)
            i2 = np.array([i2[1], i2[0]])
            xmin, xmax, ymin, ymax, lm_tmp = edge_check(lm, img, i2, lr)
            img[int(xmin):int(xmax), int(ymin):int(ymax)] = np.maximum(lm_tmp, img[int(xmin):int(xmax), int(ymin):int(ymax)])
        number = str(i + 1).zfill(3)
        save_name = save_dir + '/img' + str(m).zfill(3) + number
        cv2.imwrite(save_name + '.png', img*255)


# Create train gts heatmap 
for index, text in enumerate(natsorted(glob(abs_path('dataset/train_txt/*')))):
    m = int(text.split('video')[1].split('.txt')[0])
    save_dir = abs_path('dataset/train_gts') + "/sequence{}".format(str(m).zfill(3))
    os.makedirs(abs_path(save_dir)) if os.path.isdir(abs_path(save_dir)) is False else None
    data = np.loadtxt(text)
    print(data)
    for i in range(300):
        temp = data[data[:, 0] == i, 2:4]*2    
        img = np.zeros([1080, 1920])
        for i2 in temp:
            i2 = i2.astype(np.int)
            i2 = np.array([i2[1], i2[0]])
            xmin, xmax, ymin, ymax, lm_tmp = edge_check(lm, img, i2, lr)
            img[int(xmin):int(xmax), int(ymin):int(ymax)] = np.maximum(lm_tmp, img[int(xmin):int(xmax), int(ymin):int(ymax)])
        # img = img
        number = str(i + 1).zfill(3)
        save_name = save_dir + '/img' + str(m).zfill(3) + number
        cv2.imwrite(save_name + '.png', img*255)