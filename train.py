# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
# 参数
import argparse
# 时间
import time
# 用于获取当前时间
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 导入conf文件夹下的settings
from conf import settings
# 导入utuls文件夹下的相关文件
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR

# 定义训练函数
def train(epoch):

    # 计时开始时间
    start = time.time()

    # 网络训练模式
    net.train()

    # 将data数据集进行枚举，取一小批量batch
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        # 判断epoch是否满足参数设置
        if epoch <= args.warm:
            # warmup_steps是lr调整的耐心系数
            warmup_scheduler.step()

        # 判断是不是用gpu，如果用gpu，则调用cuda
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        # 设置梯度为0
        optimizer.zero_grad()
        # 网络输出
        outputs = net(images)
        # 损失函数
        loss = loss_function(outputs, labels)
        # 反向传播
        loss.backward()
        # 权值更新
        optimizer.step()

        # 打印训练过程中的参数：loss，lr
        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))

    # 计时结束时间
    finish = time.time()
    # 输出运行的总时间
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


# 评测模式
@torch.no_grad()
def eval_training(epoch):

    # 计时开始时间
    start = time.time()
    # 评测模式
    net.eval()
    # 初始化loss
    test_loss = 0.0 # cost function error
    # 初始化correct
    correct = 0.0

    # 将data数据集进行枚举，取一小批量batch
    for (images, labels) in cifar100_test_loader:

        # 判断是不是GPU
        if args.gpu:
            # 转换为cuda张量
            images = images.cuda()
            # 转换为cuda张量
            labels = labels.cuda()

        # 网络预测输出
        outputs = net(images)
        # 求loss
        loss = loss_function(outputs, labels)
        # loss求和
        test_loss += loss.item()
        # 求正确率
        _, preds = outputs.max(1)
        # 预测正确样本的累加
        correct += preds.eq(labels).sum()

    # 计时结束时间
    finish = time.time()

    # 打印网络loss、correct的信息
    if args.gpu:
        print('Use GPU')
    print('Evaluating Network.....')
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    # 返回正确率
    return correct.float() / len(cifar100_test_loader.dataset)

# 主函数
if __name__ == '__main__':

    # 设置模型参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()

    # 选择net
    net = get_network(args)

    # data preprocessing:
    # 构建数据装载器：train
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    # 构建数据装载器：test
    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    # 损失函数为交叉熵损失函数
    loss_function = nn.CrossEntropyLoss()

    # 优化器为SGD
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # 设置优化器调整学习率的方法
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay

    # 设置epoch
    iter_per_epoch = len(cifar100_training_loader)

    # 预热学习率warmup_scheduler
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    # 路径
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #create checkpoint folder to save model
    #  判断checkpoint_path是否存在
    if not os.path.exists(checkpoint_path):
        # 创建
        os.makedirs(checkpoint_path)
    # 更新
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    # 初始化准确率
    best_acc = 0.0

    # 迭代
    for epoch in range(1, settings.EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        # 训练模式
        train(epoch)
        # 准确率=验证模式的返回
        acc = eval_training(epoch)

        # 保存最好的模型参数
        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            # 保存模型
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            # 保存最好的结果
            best_acc = acc
            # 结束循环
            continue

        # 每10个epoch保存一次模型参数
        if not epoch % settings.SAVE_EPOCH:
            # 保存模型
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

