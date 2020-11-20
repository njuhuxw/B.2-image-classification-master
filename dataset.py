""" train and test dataset

author baiyu
"""
import os
import sys
import pickle

from skimage import io
import matplotlib.pyplot as plt
import numpy
import torch
from torch.utils.data import Dataset


# 构建Dataset-Train
class CIFAR100Train(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    # 初始化
    def __init__(self, path, transform=None):
        # if transform is given, we transoform data using
        # 打开文件
        with open(os.path.join(path, 'train'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    # 求长度
    def __len__(self):
        return len(self.data['fine_labels'.encode()])

    # 接收一个索引，返回一个样本
    def __getitem__(self, index):
        # 获取label
        label = self.data['fine_labels'.encode()][index]
        # 三通道
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        # 判断是否有transform，将transform对image进行操作
        if self.transform:
            image = self.transform(image)

        # 返回样本的label和image
        return label, image


# 构建Dataset-Test
class CIFAR100Test(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        # 打开文件
        with open(os.path.join(path, 'test'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    # len
    def __len__(self):
        return len(self.data['data'.encode()])

    # # 接收一个索引，返回一个样本
    def __getitem__(self, index):
        # 获取label
        label = self.data['fine_labels'.encode()][index]

        # 三通道
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        # 判断是否有transform，将transform对image进行操作
        if self.transform:
            image = self.transform(image)

        # 返回样本的label和image
        return label, image
