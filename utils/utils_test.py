import h5py
import torch
import shutil
import os
import glob
import os
import torch
import torch.nn as nn
import numpy as np
import scipy
import glob
import cv2
from scipy import spatial
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

def load_train_data(root, train_step = 1):
    # training data
    train_path = os.path.join(root, 'train_data', 'images')
    train_list = []
    count = 0
    for img_path in glob.glob(os.path.join(train_path, '*.jpg')):
        if count % train_step == 0:
            train_list.append(img_path)
        count += 1
    train_list.sort()
    train_pair = []
    for idx in range(len(train_list)):
        img_pair = []
        pre_img = train_list[max(0, idx - 1)]
        cur_img = train_list[idx]
        seq_id1 = pre_img[-9:-7]
        seq_id2 = cur_img[-9:-7]
        if seq_id1 == seq_id2:
            img_pair.append(pre_img)
            img_pair.append(cur_img)
            train_pair.append(img_pair)
        else:
            img_pair.append(cur_img)
            img_pair.append(cur_img)
            train_pair.append(img_pair)

    return train_pair

def load_val_data(root, val_step = 10):
    # validation data
    val_path = os.path.join(root, 'val_data', 'images')
    val_list = []
    for img_path in glob.glob(os.path.join(val_path, '*.jpg')):
        val_list.append(img_path)
    val_list.sort()
    val_pair = []
    count = 0
    for idx in range(len(val_list)):
        if count % val_step == 0:
            img_pair = []
            pre_img = val_list[max(0, idx - 1)]
            cur_img = val_list[idx]
            seq_id1 = pre_img[-9:-7]
            seq_id2 = cur_img[-9:-7]
            if seq_id1 == seq_id2:
                img_pair.append(pre_img)
                img_pair.append(cur_img)
                val_pair.append(img_pair)
            else:
                img_pair.append(cur_img)
                img_pair.append(cur_img)
                val_pair.append(img_pair)
        count += 1

    return val_pair

def load_test_data(root, test_step = 5):
    test_path = os.path.join(root, 'test_data', 'images')
    test_list = []
    for img_path in glob.glob(os.path.join(test_path, '*.jpg')):
        test_list.append(img_path)
    test_list.sort()
    test_pair = []
    for idx in range(len(test_list)):
        img_pair = []
        pre_img = test_list[max(0, idx - test_step)]
        cur_img = test_list[idx]
        seq_id1 = pre_img[-9:-7]
        seq_id2 = cur_img[-9:-7]
        if seq_id1 == seq_id2:
            img_pair.append(pre_img)
            img_pair.append(cur_img)
            test_pair.append(img_pair)
        else:
            for id in range(test_step):
                pre_img = test_list[max(0, idx - (test_step - id - 1))]
                seq_id1 = pre_img[-9:-7]
                if seq_id1 == seq_id2:
                    img_pair.append(pre_img)
                    img_pair.append(cur_img)
                    test_pair.append(img_pair)
                    break
    return test_pair

def load_pretrained_model(model_name, model):
    # print("=> loading checkpoint")
    # # checkpoint = torch.load(model_name)
    # my_models = model.state_dict()
    # pre_models = checkpoint['state_dict'].items()
    # count = 0
    # for layer_name, value in my_models.items():
    #     prelayer_name, pre_weights = pre_models[count]
    #     my_models[layer_name] = pre_weights
    #     count += 1
    # model.load_state_dict(my_models)
    return model

def load_txt(fname):
    lines = fname.readlines()
    outs = []
    for i in range(len(lines)):
        predata = []
        for j in range(len(list(lines[0].split()))):
            predata.append(float(lines[i].split(' ')[j]))
        outs.append(predata)
    return outs


def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())
def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)
            
def save_checkpoint(state, is_best,task_id, filename='checkpoint.pth.tar'):
    torch.save(state, task_id+filename)
    if is_best:
        shutil.copyfile(task_id+filename, task_id+'model_best.pth.tar')            

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density


def pt_eval(gt,x,y,sc):
    gx, gy = [], []
    for k in range(len(gt[0])):
        if gt[0][k][0] > 0:
            gx.append(gt[0][k][0])
            gy.append(gt[0][k][1])

    sorted_ind = np.argsort(-sc)
    outx = x[sorted_ind]
    outy = y[sorted_ind]

    cnt = 0
    posnum, negnum, gtnum = 0, 0, 0
    for ii in range(len(x)):
        curx = outx[ii]
        cury = outy[ii]
        min_dist = np.inf
        for jj in range(len(gx)):
            curgx = gx[jj]
            curgy = gy[jj]
            dist = np.sqrt((curx - curgx) ** 2 + (cury - curgy) ** 2)
            if dist < min_dist:
                min_dist = dist
        if min_dist <= 10:
            cnt += 1
    posnum += cnt
    negnum += len(x) - cnt
    gtnum += len(gx)
    pts = np.hstack((np.expand_dims(x.T,axis=1), np.expand_dims(y.T,axis=1), np.expand_dims(sc.T,axis=1)))
    return gtnum, posnum, negnum, pts


def calc_trkpt(outpts, outscs, thre, ourmap, neighbor_thre, ratio):
    binamap = ourmap > 0
    ourmap[binamap == 0] = 0
    data_max = filters.maximum_filter(ourmap, neighbor_thre)
    data_min = filters.minimum_filter(ourmap, neighbor_thre)
    maxima = ourmap == data_max
    diffmap = ((data_max - data_min) > 0.001)
    maxima[diffmap == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y, sc = [], [], []
    # points from density
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2 * ratio
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1) / 2 * ratio
        y.append(y_center)
        sc.append(ourmap[int(y_center/ratio), int(x_center/ratio)])

    # points from tracking
    for k in range(outpts.shape[0]):
        if outscs[k]>=thre:
            x_center = outpts[k,1] * ratio
            y_center = outpts[k,0] * ratio
            x.append(x_center)
            y.append(y_center)
            sc.append(outscs[k])

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    sc = np.asarray(sc, dtype=np.float32)
    return x, y, sc

def calc_locpt(locmap, regmap, thre, ourmap, neighbor_thre, ratio):
    binamap = ourmap > 0
    ourmap[binamap == 0] = 0
    data_max = filters.maximum_filter(ourmap, neighbor_thre)
    data_min = filters.minimum_filter(ourmap, neighbor_thre)
    maxima = ourmap == data_max
    diffmap = ((data_max - data_min) > 0.001)
    maxima[diffmap == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y, sc = [], [], []
    # points from density
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2 * ratio
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1) / 2 * ratio
        y.append(y_center)
        sc.append(ourmap[int(y_center/ratio), int(x_center/ratio)])

    binamap = locmap > thre
    locmap[binamap == 0] = 0
    data_max = filters.maximum_filter(locmap, neighbor_thre)
    data_min = filters.minimum_filter(locmap, neighbor_thre)
    maxima = locmap == data_max
    diffmap = ((data_max - data_min) > 0.001)
    maxima[diffmap == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    # points from localization
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2 * ratio
        y_center = (dy.start + dy.stop - 1) / 2 * ratio
        offset = regmap[:, int(y_center / ratio), int(x_center / ratio)]
        x.append(x_center+offset[1]*5*ratio)
        y.append(y_center+offset[0]*5*ratio)
        sc.append(locmap[int(y_center/ratio), int(x_center/ratio)])

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    sc = np.asarray(sc, dtype=np.float32)
    return x, y, sc
 
def calc_denpt(ourmap, neighbor_thre, ratio):
    binamap = ourmap > 0
    ourmap[binamap == 0] = 0
    data_max = filters.maximum_filter(ourmap, neighbor_thre)
    data_min = filters.minimum_filter(ourmap, neighbor_thre)
    maxima = ourmap == data_max
    diffmap = ((data_max - data_min) > 0.001)
    maxima[diffmap == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)

    x, y, sc = [], [], []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2 * ratio
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1) / 2 * ratio
        y.append(y_center)
        sc.append(ourmap[int(y_center/ratio), int(x_center/ratio)])

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    sc = np.asarray(sc, dtype=np.float32)
    return x, y, sc


def localization(gt,x,y,sc):
    temp = sc
    Pre = []
    gx, gy = [], []
    for k in range(len(gt[0])):
        if gt[0][k][0] > 0:
            gx.append(gt[0][k][0])
            gy.append(gt[0][k][1])
    mAP = []
    thr = 0.1
    num = np.where(temp[temp > thr])
    sc_t = sc[num]
    sorted_ind = np.argsort(-sc_t)
    outx = x[num][sorted_ind]
    outy = y[num][sorted_ind]

    for localization in range(1, 25):
        cnt = 0
        precision = []
        recall = []
        AP = []
        for ii in range(len(sc_t)):
            if ii > 10000:
                break
            curx = outx[ii]
            cury = outy[ii]
            min_dist = np.inf
            for jj in range(len(gx)):
                curgx = gx[jj]
                curgy = gy[jj]
                dist = np.sqrt((curx - curgx) ** 2 + (cury - curgy) ** 2)
                if dist < min_dist:
                    min_dist = dist
            if min_dist <= localization:
                cnt += 1
            if ii >= 1:
                if recall[ii-1] == cnt / len(gx):
                    precision.append(cnt / (ii+1))
                    recall.append(cnt / len(gx))             
                else:
                    precision.append(cnt / (ii+1))
                    recall.append(cnt / len(gx))
                    AP.append((recall[ii] - recall[ii-1])*(precision[ii]+precision[ii-1])*0.5)
            else:
                precision.append(cnt / (ii+1))
                recall.append(cnt / len(gx))
        mAP.append(np.sum(AP))
        Pre.append(np.sum(precision))
    mAP = np.mean(mAP)
    Pre = np.mean(Pre)
    return mAP
