from os.path import splitext, join, basename
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import cv2
import random
from hydra.utils import to_absolute_path as abs_path


class Heatmap_Dataset(Dataset):
    def __init__(self, cfg_type, cfg_data):
        self.imgs_dir = abs_path(cfg_type.imgs)
        self.gts_dir = abs_path(cfg_type.mpms)
        self.edge = cfg_data.edge
        self.height, self.width = cfg_data.height, cfg_data.width
        self.batch_size = cfg_type.batch_size - 1
        self.train = cfg_type.train

        seqs = sorted(glob(join(self.imgs_dir, "*")))
        self.img_paths = []
        self.ids = []
        min_itv = 1
        temp_itv = min_itv * self.batch_size

        for i, seq in enumerate(seqs):
            seq_name = basename(seq)
            self.img_paths.extend(
                [[path, seq_name] for path in sorted(glob(join(seq, "*")))][:-1]
            )
            num_seq = len(listdir(seq)) - 1
            self.ids.extend(
                [[i * temp_itv, num_seq - j] for j in range(num_seq)[:-temp_itv]]
            )

        logging.info(f"Creating dataset with {len(self.ids)} examples")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        # i = self.batch_size * i
        i = i[0]
        idx = self.img_paths[i + self.ids[i][0]]
        itv = 1

        lt = [i + self.ids[i][0]]
        temp = i + self.ids[i][0]
        for x in range(1, self.batch_size + 1):
            lt.append(temp + itv)
            temp += itv
        nxt_idx = []

        for x in lt:
            nxt_idx.append(self.img_paths[x][0])

        img_names = nxt_idx
        img_files = []
        mpm_file = []
        for i in range(len(img_names)):
            name = splitext(basename(img_names[i]))[0]
            img_files.append(splitext(basename(img_names[i]))[0])
            mpm_file.append(
                glob(join(self.gts_dir, idx[1], f"{itv:03}", name) + ".*")[0]
            )

        assert (
            len(mpm_file) >= 1
        ), f"Either no mask or multiple masks found for the ID {idx}: {mpm_file}"
        assert (
            len(img_files) > 1
        ), f"Either no image or multiple images found for the ID {idx}: {img_files}"

        img1 = []
        mag = []
        for id, x in enumerate(img_names):
            img1.append(np.expand_dims(cv2.imread(img_names[id]), axis=0))
            mag.append(
                np.expand_dims(
                    cv2.cvtColor(np.load(mpm_file[id])["arr_0"], cv2.COLOR_BGR2GRAY),
                    axis=0,
                )
            )
        img = np.concatenate(img1)
        mag = np.concatenate(mag)[:, :, :, np.newaxis]

        if img.max() > 1:
            img = img / 255

        # random crop
        if self.train:
            seed1 = np.random.randint(self.edge, img.shape[1] - self.height - self.edge)
            seed2 = np.random.randint(self.edge, img.shape[2] - self.width - self.edge)
            img = img[:, seed1 : seed1 + self.height, seed2 : seed2 + self.width]
            mag = mag[:, seed1 : seed1 + self.height, seed2 : seed2 + self.width]

        # crop : same as DroneCrowd dataset's method
        if self.train == True:
            crop_factor = 0.5
            crop_size = (
                int(img.shape[1] * crop_factor),
                int(img.shape[2] * crop_factor),
            )
            dx = int(random.randint(0, 1) * img.shape[1] * 1.0 / 2)
            dy = int(random.randint(0, 1) * img.shape[0] * 1.0 / 2)

            img = img[:, dy : crop_size[0] + dy, dx : crop_size[1] + dx]
            mag = mag[:, dy : crop_size[0] + dy, dx : crop_size[1] + dx]

        img_trans = img.transpose((0, 3, 1, 2))
        mpm_trans = img.transpose((0, 3, 1, 2))

        return {
            "img": torch.from_numpy(img_trans).type(torch.FloatTensor),
            "mag": torch.from_numpy(mpm_trans).type(torch.FloatTensor),
        }
