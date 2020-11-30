# 阶段2-练习1
1. 介绍：本次练习为图像分类（image classification）
2. 要求：
	- 适配CIFAR100数据集，任选3种不同的网络结构（参考下方Implemented network），使用默认超参数训练、测试模型：
		- 关于CIFAR100数据集，直接使用torchvision.datasets.CIFAR100加载数据，用法与torchvision.datasets.CIFAR10一样，具体参考代码；
		- **结果1**：将测试结果添加到下方的Results表格中；
		- **结果2**：将你实验中效果最好的模型在CIFAR100测试集上跑一遍，将预测结果保存为一个文本文件，命名为results_xxx.txt（xxx，为网络结构名称，如vgg16），每一行保存一张图片的预测类别，即0-99的整数，**样本顺序与测试集中样本的原始顺序一致**；
		- **结果3**：通读本项目代码，给每一行添加注释（不含models目录）。
3. 提交结果：
results_xxx.txt放在本项目根目录下，将本项目打包为zip文件提交。
注意，把data目录删掉。


# Pytorch-cifar10

practice on cifar10 using pytorch

## Requirements

Experiment environment
- python3.6
- pytorch1.6.0
- cuda10 (optional)


## Usage (example)
### 1. train the model
```bash
# use gpu to train vgg16
$ python train.py -net vgg16
```
sometimes, you might want to use warmup training by set ```-warm``` to 1 or 2, to prevent network
diverge during early training phase.


The supported net args are:
```
squeezenet
mobilenet
mobilenetv2
shufflenet
shufflenetv2
vgg11
vgg13
vgg16
vgg19
densenet121
densenet161
densenet201
googlenet
inceptionv3
inceptionv4
inceptionresnetv2
xception
resnet18
resnet34
resnet50
resnet101
resnet152
preactresnet18
preactresnet34
preactresnet50
preactresnet101
preactresnet152
resnext50
resnext101
resnext152
attention56
attention92
seresnet18
seresnet34
seresnet50
seresnet101
seresnet152
nasnet
```
Normally, the weights file with the best accuracy would be written to the disk with name suffix 'best'(default in checkpoint folder).


### 2. test the model
Test the model using test.py
```bash
$ python test.py -net vgg16 -weights path_to_vgg16_weights_file
```

## Implementated NetWork

- vgg [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556v6)
- googlenet [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842v1)
- inceptionv3 [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567v3)
- inceptionv4, inception_resnet_v2 [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)
- xception [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
- resnet [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385v1)
- resnext [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431v2)
- resnet in resnet [Resnet in Resnet: Generalizing Residual Architectures](https://arxiv.org/abs/1603.08029v1)
- densenet [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993v5)
- shufflenet [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083v2)
- shufflenetv2 [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164v1)
- mobilenet [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- mobilenetv2 [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- residual attention network [Residual Attention Network for Image Classification](https://arxiv.org/abs/1704.06904)
- senet [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- squeezenet [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360v4)
- nasnet [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012v4)


## Results

|dataset|network|params|top1 err|top5 err|memory|
|:---:|:---:|:---:|:---:|:---:|:---:|
|cifar100|vgg16_bn |34.0M|27.07|8.84|2.03GB|
|cifar100|vgg13_bn |27.37M|33.12|10.94|109.49MB|
|cifar100|vgg11_bn |27.18M|36.14|12.92|108.79MB|
|cifar100|mobilenet|3.16M|36.81|11.97|12.64MB|




