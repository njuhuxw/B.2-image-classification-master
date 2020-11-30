#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""
# 参数
import argparse


import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# 导入conf的包
from conf import settings
# 导入utils的包
from utils import get_network, get_test_dataloader

# 主函数
if __name__ == '__main__':
    # 设置参数：net/ weights/ gpu/ batch size
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()

    # 网络框架
    net = get_network(args)
    # 构建数据装载器：test
    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        # 4进程装载
        num_workers=4,
        batch_size=args.b,
    )
    # 加载模型训练好的参数
    net.load_state_dict(torch.load(args.weights))
    # 打印网络参数
    print(net)
    # 验证模式
    net.eval()

    # 初始化参数correct
    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        # 枚举
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))
            # 判断是否用gpu
            if args.gpu:
                # 转换为cuda张量
                image = image.cuda()
                # 转换为cuda张量
                label = label.cuda()
            # 网络输出
            output = net(image)
            # 输出预测的前五
            # _, pred = output.topk(5, 1, largest=True, sorted=True)
            _, pred = output.topk(1, 1, largest=True, sorted=True)
            # 得到label和correct
            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()

    # 打印输出
    print()
    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))