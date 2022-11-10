import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from scipy.ndimage.filters import gaussian_filter

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

def getpoints(mag, mag_max=1.0, th=0.1):
    '''
    Args:
        mpm (numpy.ndarray): MPM
        mag (numpy.ndarray): Magnitude of MPM i.e. heatmap
        mag_max: Maximum value of heatmap
    Return:
        result (numpy.ndarray): Table of peak coordinates, warped coordinates, and peak value [x, y, wx, wy, pv]
    '''
    mag[mag > mag_max] = mag_max
    map_left_top, map_top, map_right_top = np.zeros(mag.shape), np.zeros(mag.shape), np.zeros(mag.shape)
    map_left, map_right = np.zeros(mag.shape), np.zeros(mag.shape)
    map_left_bottom, map_bottom, map_right_bottom = np.zeros(mag.shape), np.zeros(mag.shape), np.zeros(mag.shape)
    map_left_top[1:, 1:], map_top[:, 1:], map_right_top[:-1, 1:] = mag[:-1, :-1], mag[:, :-1], mag[1:, :-1]
    map_left[1:, :], map_right[:-1, :] = mag[:-1, :], mag[1:, :]
    map_left_bottom[1:, :-1], map_bottom[:, :-1], map_left_bottom[1:, :-1] = mag[:-1, 1:], mag[:, 1:], mag[1:, 1:]
    peaks_binary = np.logical_and.reduce((
        mag >= map_left_top, mag >= map_top, mag >= map_right_top,
        mag >= map_left, mag > th, mag >= map_right,
        mag >= map_left_bottom, mag >= map_bottom, mag >= map_right_bottom,
    ))
    _, _, _, center = cv2.connectedComponentsWithStats((peaks_binary * 1).astype('uint8'))
    center = center[1:]
    result = []
    for center_cell in center.astype('int'):
        mag_value = mag[center_cell[1], center_cell[0]]
        result.append([center_cell[0], center_cell[1], mag_value])
    return result



def inferenceMPM(model, names, ch=None, max_v=255):
    if ch is None:
        imgs = [cv2.imread(name, -1)[None, ...] for name in names]
    else:
        imgs = [cv2.imread(name, -1).transpose(2, 0, 1) for name in names]
    img = np.concatenate(imgs, axis=0)
    img = img / max_v
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.cuda()
    output = model(img)
    mpm = output[0].cpu().detach().numpy()
    return mpm.transpose(1, 2, 0)


def getIndicatedPoints(acm, mag_max=0.9, mag_min=0.1, z_value=5, gauss=False, sigma=3):
    mag = np.sqrt(np.sum(np.square(acm), axis=-1))
    if gauss:
        mag = gaussian_filter(mag, sigma=sigma)

    mag[mag > mag_max] = mag_max
    map_left = np.zeros(mag.shape)
    map_right = np.zeros(mag.shape)
    map_top = np.zeros(mag.shape)
    map_bottom = np.zeros(mag.shape)
    map_left_top = np.zeros(mag.shape)
    map_right_top = np.zeros(mag.shape)
    map_left_bottom = np.zeros(mag.shape)
    map_right_bottom = np.zeros(mag.shape)
    map_left[1:, :] = mag[:-1, :]
    map_right[:-1, :] = mag[1:, :]
    map_top[:, 1:] = mag[:, :-1]
    map_bottom[:, :-1] = mag[:, 1:]
    map_left_top[1:, 1:] = mag[:-1, :-1]
    map_right_top[:-1, 1:] = mag[1:, :-1]
    map_left_bottom[1:, :-1] = mag[:-1, 1:]
    map_right_bottom[:-1, :-1] = mag[1:, 1:]
    peaks_binary = np.logical_and.reduce((
        mag >= map_left,
        mag >= map_right,
        mag >= map_top,
        mag >= map_bottom,
        mag >= map_left_top,
        mag >= map_left_bottom,
        mag >= map_right_top,
        mag >= map_right_bottom,
        mag > mag_min
    ))
    _, _, _, center = cv2.connectedComponentsWithStats((peaks_binary * 1).astype('uint8'))
    center = center[1:]
    result = []
    for center_cell in center.astype('int'):
        vec = acm[center_cell[1], center_cell[0]]
        mag_value = mag[center_cell[1], center_cell[0]]
        vec = vec / np.linalg.norm(vec)
        # print(vec)
        x = 0 if vec[1] == 0 else z_value * (vec[1] / vec[2])
        y = 0 if vec[0] == 0 else z_value * (vec[0] / vec[2])
        x = int(x)
        y = int(y)
        result.append([center_cell[0], center_cell[1], center_cell[0] + x, center_cell[1] + y, mag_value])

    return np.array(result)


def calculate_homography_matrix(origin, dest):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    """
    線形DLT法にて、 変換元を変換先に対応づけるホモグラフィ行列を求める。先行実装に倣う。
    :param origin: ホモグラフィ行列計算用の初期座標配列
    :param dest: ホモグラフィ行列計算用の移動先座標配列
    :return: 計算結果のホモグラフィ行列(3 x 3)
    """
    assert origin.shape == dest.shape

    origin = convert_corner_list_to_homography_param(origin.T)
    dest = convert_corner_list_to_homography_param(dest.T)

    # 点を調整する（数値計算上重要）
    origin, c1 = normalize(origin)  # 変換元
    dest, c2 = normalize(dest)      # 変換先

    # 線形法計算のための行列を作る。
    nbr_correspondences = origin.shape[1]
    a = np.zeros((2 * nbr_correspondences, 9))
    for i in range(nbr_correspondences):
        a[2 * i] = [-origin[0][i], -origin[1][i], -1, 0, 0, 0, dest[0][i] * origin[0][i], dest[0][i] * origin[1][i],
                    dest[0][i]]
        a[2 * i + 1] = [0, 0, 0, -origin[0][i], -origin[1][i], -1, dest[1][i] * origin[0][i], dest[1][i] * origin[1][i],
                        dest[1][i]]
    u, s, v = np.linalg.svd(a)
    homography_matrix = v[8].reshape((3, 3))
    homography_matrix = np.dot(np.linalg.inv(c2), np.dot(homography_matrix, c1))
    homography_matrix = homography_matrix / homography_matrix[2, 2]
    return homography_matrix

