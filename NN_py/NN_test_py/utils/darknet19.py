# -*- coding: utf-8 -*-
# darknet19.py

from torch import nn
import os
import torch

# 参数配置,标准的darknet19参数
cfg = [32, 'M', 64, 'M', 128, 64, 128, 'M', 256, 128, 256, 'M',
       512, 256, 512, 256, 512, 'M', 1024, 512, 1024, 512, 1024]

def make_layers(cfg, in_channels=3, batch_norm=True):
    """
    从配置参数中构建网络
    :param cfg:  参数配置
    :param in_channels: 输入通道数,RGB彩图为3, 灰度图为1
    :param batch_norm:  是否使用批正则化
    :return:
    """
    layers = []
    flag = True             # 用于变换卷积核大小
    in_channels = in_channels
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels=in_channels,
                                   out_channels=v,
                                   kernel_size=(1, 3)[flag],
                                   stride=1,
                                   padding=(0,1)[flag],
                                   bias=False))
            if batch_norm:
                layers.append(nn.BatchNorm2d(v))
            in_channels = v
            
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            
        flag = not flag
        
    return nn.Sequential(*layers)

class Darknet19(nn.Module):
    """
    Darknet19 模型
    """
    def __init__(self, num_classes=1000, in_channels=3, batch_norm=True, pretrained=False):
        """
        模型结构初始化
        :param num_classes: 最终分类数
        :param in_channels: 输入数据的通道数
        :param batch_norm:  是否使用正则化
        :param pretrained:  是否导入预训练参数
        """
        super(Darknet19, self).__init__()
        # 特征提取部分
        self.features = make_layers(cfg, in_channels=in_channels, batch_norm=batch_norm)
        # 分类层
        self.classifier = nn.Sequential(
            nn.Conv2d(1024, num_classes, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d(output_size=(1)),
            nn.Softmax(dim=0)
        )
        
        if pretrained:
            self.load_weight()
        else:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_weight(self, weight_file='weights/darknet19-deepBakSu-e1b3ec1e.pth'):
        """
        加载预训练权重
        :param weight_file: 权重文件路径
        """
        if not os.path.exists(weight_file):
            raise FileNotFoundError(f"weight file {weight_file} not found")
            
        pretrained_dict = torch.load(weight_file)
        model_dict = self.state_dict()
        # 转换权重文件中的keys
        dic = {}
        for now_keys, values in zip(self.state_dict().keys(), pretrained_dict.values()):
            dic[now_keys] = values
        self.load_state_dict(dic)