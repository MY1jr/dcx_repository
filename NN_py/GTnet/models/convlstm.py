import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# 输入图像为（batch,time_step,100,100,3）
class ConvLSTM(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_step=5):
        super().__init__()
        self.time_step = time_step
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 卷积块 I: 输入(batch,time_step,100,100,3)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        
        # 特征编码全连接块
        self.fc1 = nn.Sequential(
            nn.Linear(23*23*32, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Dropout(0.1)
        )
        
        # LSTM网络层: 输入序列长度为time_step，每个时间步64维特征
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        
        self.fc2 = nn.Sequential(nn.Linear(64, 64), nn.Tanh())
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN特征提取 [batch, time_step, channel, height, width]
        cnn_out = []
        for t in range(self.time_step):
            # 处理每个时间步的图像
            out = self.conv1(x[:,t,:,:,:])
            out = self.pool1(out)
            out = self.conv2(out)
            out = self.pool2(out)
            
            # 展平并进行特征编码
            out = torch.flatten(out, start_dim=1)
            out = self.fc1(out)  # 得到64维特征向量
            cnn_out.append(out)
        
        # 堆叠时间步的特征 [batch, time_step, feature_size]
        cnn_out = torch.stack(cnn_out, dim=1)
        
        # LSTM处理序列
        lstm_out, _ = self.lstm(cnn_out)
        
        # 取最后一个时间步的输出
        out = self.fc2(lstm_out[:,-1,:])
        out = self.fc3(out)
        return out