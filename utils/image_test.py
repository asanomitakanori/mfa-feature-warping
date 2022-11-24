from PIL import Image
import numpy as np
import scipy.io as io
from torchvision import transforms
from os.path import splitext, basename

def load_data(img_paths, train = False):
    #DroneCrowd
    imgset = []
    gts = []
    gtset = []
    
    for i in range(len(img_paths)):
        img = Image.open(img_paths[i]).convert('RGB')
        imgset.append(img)
        mat_path = img_paths[i].replace('.jpg', '.mat').replace('val_imgs', 'ground_truth').replace('img', 'GT_img')
        mat_path2 = img_paths[i].replace('.jpg', '.mat').replace('val_imgs', 'val_map').replace('mat', 'png')
        gtset.append(Image.open(mat_path2))
        base = basename(splitext(splitext(mat_path)[0])[0])
        num = base[6:9]
        mat_path = mat_path.replace('sequence' + num + '/', '')
        mat = io.loadmat(mat_path)
        gt = mat["image_info"][0, 0][0, 0][0]
        gt = np.array(gt, dtype=np.float32)
        # gt_ = np.zeros(shape=[512 - gt.shape[0], 2])
        gt_ = np.zeros(shape=[512 - gt.shape[0], 3])
        all_gt = np.vstack((gt, gt_))
        if i == 0:
            gtnum = np.sum(gt.shape[0])
        else:
            gtnum = np.vstack((gtnum, np.sum(gt.shape[0])))
        gts.append(all_gt)

    # crop the images
    crop_factor = 0.5
    imgs1, imgs2, imgs3, imgs4 = [], [], [], []
    for i in range(len(imgset)):
        # crop the images
        crop_size = (int(imgset[0].size[1] * crop_factor), int(imgset[0].size[0] * crop_factor))
        imgs = transforms.FiveCrop(crop_size)(imgset[i])
        img1, img2, img3, img4 = imgs[0:4]
        imgs1.append(img1)
        imgs2.append(img2)
        imgs3.append(img3)
        imgs4.append(img4)
    return imgs1, imgs2, imgs3, imgs4, gtnum, gts, gtset
