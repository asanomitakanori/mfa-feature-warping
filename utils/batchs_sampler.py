import numpy as np
from torch.utils import data as data
import random

def split_list(l, n):
    """
    リストをサブリストに分割する
    :param l: リスト
    :param n: サブリストの要素数
    :return: 
    """
    for idx in range(0, len(l), n):
        yield l[idx:idx + n]


class MyDataset(data.Dataset):
    def __init__(self, n):
        self.data = np.random.rand(n, 100, 100, 3)
        self.labels = np.arange(n)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class SequentialSampler(data.Sampler):
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        temp = list(split_list(range(0, len(self.dataset), self.batch_size), 1))
        if self.shuffle == True:
            return iter(random.sample(temp, len(temp)))
        if self.shuffle == False:
            return iter(temp)


