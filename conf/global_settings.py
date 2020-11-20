""" configurations for this project

author baiyu
"""
import os
from datetime import datetime

# mean and std of cifar100 dataset
CIFAR10_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR10_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

# total training epoches
# epoches
EPOCH = 30
MILESTONES = [10, 15, 20]
# MILESTONES = [60, 120, 160]

# initial learning rate
# INIT_LR = 0.1

# time of we run the script
TIME_NOW = datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')

# tensorboard log dir
LOG_DIR = 'runs'

# save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10
