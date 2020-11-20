
import argparse
import glob
import os

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from torchvision import transforms
from conf import settings
from utils import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from torch.optim.lr_scheduler import _LRScheduler


# 学习率呈指数增长
class FindLR(_LRScheduler):
    """exponentially increasing learning rate

    Args:
        optimizer: optimzier(e.g. SGD)
        num_iter: totoal_iters
        max_lr: maximum  learning rate
    """
    # 初始化学习率参数、优化器、最大学习率、迭代次数
    def __init__(self, optimizer, max_lr=10, num_iter=100, last_epoch=-1):

        self.total_iters = num_iter
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch)

    # 求得新的学习率
    def get_lr(self):

        return [base_lr * (self.max_lr / base_lr) ** (self.last_epoch / (self.total_iters + 1e-32)) for base_lr in self.base_lrs]