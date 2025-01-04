import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#### 检测头
### 重叠区间的多bin损失函数
class MultiBinLoss(nn.Module):
    def __init__(self, num_bins=4):  # 默认分为4个bin,每个bin 90度,相邻bin之间有重叠
        super(MultiBinLoss, self).__init__()
        self.num_bins = num_bins
        self.bin_size = 2 * np.pi / num_bins  # 每个bin的大小
        self.overlap = np.pi / 32  # 重叠区域大小,5.62*2度
        
        # 生成每个bin的中心角度
        self.bin_centers = torch.arange(num_bins).float() * self.bin_size
        
        # 调整权重
        self.cls_weight = 1.0
        self.reg_weight = 2.0  # 增加回归权重
        self.unit_weight = 0.1  # 单位向量约束权重
        self.smooth_weight = 0.05  # 平滑性约束权重
        
    def forward(self, pred, target):
        """
        Args:
            pred: [batch, num_bins*2] 预测值
            target: [batch, 2]  即(cos, sin)形式的角度
        Returns:
            loss: 总损失 = 分类损失 + 回归损失
        """
        batch_size = pred.shape[0]
        
        # 1. 分离预测值的cos和sin
        pred_cos = pred[:, :self.num_bins]  # [batch, num_bins]
        pred_sin = pred[:, self.num_bins:]  # [batch, num_bins]
        
        # 2. 获取目标角度（从cos和sin计算）
        target_angle = torch.atan2(target[:, 1], target[:, 0])  # [batch]
        target_angle = (target_angle + 2 * np.pi) % (2 * np.pi)  # 转换到[0, 2π]
        
        # 计算bin权重
        diff = torch.abs(target_angle.unsqueeze(1) - self.bin_centers.to(target.device).unsqueeze(0))
        diff = torch.min(diff, 2*np.pi - diff)
        
        weights = torch.zeros_like(diff)
        weights[diff <= self.bin_size/2 - self.overlap] = 1.0  # bin中心区域权重为1
        overlap_mask = (diff > self.bin_size/2 - self.overlap) & (diff < self.bin_size/2)
        weights[overlap_mask] = 1 - (diff[overlap_mask] - (self.bin_size/2 - self.overlap)) / self.overlap
        
        # 分类损失
        cls_loss = F.binary_cross_entropy_with_logits(pred_cos, weights)
        
        # 回归损失
        target_cos = target[:, 0].unsqueeze(1).expand(-1, self.num_bins)  # [batch, num_bins]
        target_sin = target[:, 1].unsqueeze(1).expand(-1, self.num_bins)  # [batch, num_bins]
        
        cos_loss = (weights * F.mse_loss(pred_cos, target_cos, reduction='none')).sum(dim=1).mean()
        sin_loss = (weights * F.mse_loss(pred_sin, target_sin, reduction='none')).sum(dim=1).mean()
        reg_loss = cos_loss + sin_loss
        
        # 单位向量约束 #########
        pred_norm = torch.sqrt(pred_cos**2 + pred_sin**2)
        unit_loss = F.mse_loss(pred_norm, torch.ones_like(pred_norm))
        
        # 平滑性约束（相邻bin的预测应该平滑）
        cos_diff = pred_cos[:, 1:] - pred_cos[:, :-1]
        sin_diff = pred_sin[:, 1:] - pred_sin[:, :-1]
        smooth_loss = torch.mean(cos_diff**2 + sin_diff**2)
        
        # 总损失
        total_loss = (self.cls_weight * cls_loss + 
                     self.reg_weight * reg_loss + 
                     self.unit_weight * unit_loss + 
                     self.smooth_weight * smooth_loss)
        
        return total_loss # 标量


#### 主干网络 with Muti-bin 接口
class CNN_LSTM_MB(nn.Module):
    def __init__(self, in_channels=3, num_bins=24, height=64, width=64, time_step=5):
        super(CNN_LSTM_MB, self).__init__()
        self.in_channels = in_channels
        self.time_step = time_step
        self.num_bins = num_bins
        self.out_channels = num_bins * 2 + 1  # muti-bin angle(cos, sin) + speed
        
        # CNN部分 
        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),  # 添加2D Dropout
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),  # 添加2D Dropout
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),  # 添加2D Dropout
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1)  # 添加2D Dropout
        )
        
        # 计算CNN输出特征维度
        h = height // 16  # 经过4次池化层
        w = width // 16
        self.feature_size = 128 * h * w
        
        # LSTM部分
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=256,
            num_layers=3,
            batch_first=True,
            dropout=0.1  # 添加LSTM层间的dropout
        )
        
        # Dropout层
        self.dropout = nn.Dropout(0.2)  # 添加全连接层前的dropout
        
        # 分别预测角度和速度
        self.fc_angle = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.num_bins * 2)  # 角度预测
        )
        
        self.fc_speed = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)  # 速度预测
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN特征提取 [batch, time_step, channel, height, width]
        cnn_out = []
        for t in range(self.time_step):
            out = self.cnn(x[:,t,:,:,:])  # 处理每个时间步的图像
            out = out.view(batch_size, -1)  # 展平
            cnn_out.append(out)
            
        # 堆叠时间步的特征
        cnn_out = torch.stack(cnn_out, dim=1)  # [batch, time_step, feature_size]
        
        # LSTM处理序列
        lstm_out, _ = self.lstm(cnn_out)
        
        # 在全连接层前应用dropout
        lstm_out = self.dropout(lstm_out[:,-1,:])
        
        # 分别预测角度和速度
        angle_pred = self.fc_angle(lstm_out)    # [batch, num_bins*2]
        speed_pred = self.fc_speed(lstm_out)    # [batch, 1]
        
        
        angle_pred = angle_pred.view(batch_size, self.num_bins, 2)
        # 使用tanh激活函数确保cos和sin的值在[-1,1]范围内
        angle_pred = torch.tanh(angle_pred)
        
        # 归一化以严格满足cos^2 + sin^2 = 1 #########
        angle_norms = torch.norm(angle_pred, dim=2, keepdim=True)
        angle_pred = angle_pred / (angle_norms + 1e-7)  # 添加一个小的epsilon防止除零
        angle_pred = angle_pred.view(batch_size, -1)
        
        # 合并输出
        output = torch.cat([angle_pred, speed_pred], dim=1)
        
        return output


# 输入图像为（batch,time_step,100,100,3）
class ConvLSTM_MB(nn.Module):
    def __init__(self, in_channels=3, num_bins=4, time_step=5):
        super(ConvLSTM_MB, self).__init__()
        self.time_step = time_step
        self.in_channels = in_channels
        self.num_bins = num_bins
        self.out_channels = num_bins * 2 + 1  # num_bins对cos,sin + 1个速度
        
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
        self.fc3 = nn.Linear(64, self.out_channels)

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