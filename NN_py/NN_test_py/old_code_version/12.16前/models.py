import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from attentions import MultiHeadAttention
#######################################################################################
# 输入图像为（batch,time_step,100,100,3）
class ConvLSTM(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_step=5):
        super(ConvLSTM, self).__init__()
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

#######################################################################################
class GTNet(nn.Module):
    """
    输入: [batch, time_step, channel, height, width]
    输出: [batch, 3] - out[:,0]为速度，out[:,1:]为角度(cos,sin)
    """
    def __init__(self, in_channels=3, out_channels=3, height=64, width=64, time_step=20):
        super(GTNet, self).__init__()
        
        # 共享的特征提取层
        self.time_step = time_step
        self.input_layer = InputLayer(in_channels)  # 输入[batch,channel,h,w] -> 输出[batch,32,h,w]
        self.backbone = Backbone(input_dim=32)  # 输入[batch,32,h,w] -> 输出[batch,128,h/8,w/8]
        self.out_channels = out_channels
        # 共享的特征融合层
        self.neck = Neck(
            in_channels=128,
            feature_size=128 * (height//8) * (width//8),  # 展平后的特征维度
            hidden_size=256,
            time_step=time_step
        )  # 输入[batch,time_step,feature_size] -> 输出[batch,512]
        
        # 分离的任务头
        self.speed_head = SpeedHead(
            in_channels=512,
            out_channels=1  # 速度预测
        )  # 输入[batch,512] -> 输出[batch,1]
        
        self.angle_head = AngleHead(
            in_channels=512,
            out_channels=out_channels-1  # cos和sin预测
        )  # 输入[batch,512] -> 输出[batch,2]
        
    def forward(self, x):  # x: [batch, time_step, channel, height, width]
        batch_size = x.size(0)
        
        # 处理每个时间步
        features = []
        for t in range(self.time_step):
            # Input处理 [batch, channel, h, w] -> [batch, 32, h, w]
            x_t = self.input_layer(x[:,t,:,:,:])
            
            # Backbone特征提取 [batch, 32, h, w] -> [batch, 128, h/8, w/8]
            feat = self.backbone(x_t)
            feat = feat.view(batch_size, -1)  # [batch, feature_size]
            features.append(feat)
            
        # 转换为时序特征 [batch, time_step, feature_size]
        features = torch.stack(features, dim=1)
        
        # Neck处理 [batch, time_step, feature_size] -> [batch, 512]
        feat_enhanced = self.neck(features)
        
        # 分别通过不同的任务头
        speed = self.speed_head(feat_enhanced)  # [batch, 1]
        angle = self.angle_head(feat_enhanced)  # [batch, 2]
        
        # 合并输出 [batch, 3]
        out = torch.cat([speed, angle], dim=1)
        
        return out

class InputLayer(nn.Module):
    """
    输入预处理模块 - 使用卷积实现Focus
    输入: [batch, channel, height, width]
    输出: [batch, 32, height, width]
    """
    def __init__(self, in_channels):
        super(InputLayer, self).__init__()
        
        # 使用卷积模拟Focus效果
        self.focus_conv = nn.Sequential(
            # 步长为2的卷积实现Focus: [b,c,h,w] -> [b,64,h/2,w/2]
            nn.Conv2d(in_channels, 64, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            
            # 特征增强
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        
        # 上采样恢复分辨率并降低通道数
        self.up_conv = nn.Sequential(
            # 上采样: [b,64,h/2,w/2] -> [b,32,h,w]
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        
    def forward(self, x):
        # Focus效果的卷积
        x = self.focus_conv(x)  # [b,64,h/2,w/2]
        
        # 恢复分辨率
        x = self.up_conv(x)     # [b,32,h,w]
        
        return x

class Backbone(nn.Module):
    """
    主干特征提取网络
    输入: [batch, input_dim, height, width]
    输出: [batch, 128, height/8, width/8]
    """
    def __init__(self, input_dim):
        super(Backbone, self).__init__()
        self.features = nn.Sequential(
            # Stage 1: [batch, input_dim, h, w] -> [batch, 64, h/2, w/2]
            nn.Conv2d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Stage 2: [batch, 64, h/2, w/2] -> [batch, 96, h/4, w/4]
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Stage 3: [batch, 96, h/4, w/4] -> [batch, 128, h/8, w/8]
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
    def forward(self, x):
        return self.features(x)

class EncoderLayer(nn.Module):
    """
    输入: [batch, time_step, hidden_size]
    输出: [batch, time_step, hidden_size]
    """
    def __init__(self, hidden_size, num_heads, feedforward_size, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, feedforward_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_size, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.self_attn(
            query=x,
            key=x,
            value=x
        )
        x = x + self.dropout1(attn_output)  # Residual connection
        x = self.norm1(x)

        # Feedforward network
        ff_output = self.feedforward(x)
        x = x + self.dropout2(ff_output)  # Residual connection
        x = self.norm2(x)

        return x

class DecoderLayer(nn.Module):
    """
    输入：
    x: [batch, 1, hidden_size]
    encoder_output: [batch, time_step, hidden_size]
    输出：
    x: [batch, time_step, hidden_size]
    """
    def __init__(self, hidden_size, num_heads, feedforward_size, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn = MultiHeadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, feedforward_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_size, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output):
        # mask 生成一个上三角矩阵，用于遮蔽未来的信息 [time_step, time_step]
        mask = self.generate_square_subsequent_mask(x.size(1)).to(device=x.device, dtype=torch.bool) # 需要把mask发送至GPU
         
        # Self-attention
        attn_output, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=mask
        )
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Cross-attention
        cross_output, _ = self.cross_attn(
            query=x,
            key=encoder_output,
            value=encoder_output
        )
        x = x + self.dropout2(cross_output)
        x = self.norm2(x)
        
        # Feedforward network
        ff_output = self.feedforward(x)
        x = x + self.dropout3(ff_output)  # Residual connection
        x = self.norm3(x)

        return x
    
    def generate_square_subsequent_mask(self, sz):
        """生成一个bool类型的上三角mask矩阵"""
        # 直接生成bool类型的mask ## true表示遮挡
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

class Neck(nn.Module):
    """
    特征融合与增强模块 - 使用Transformer结构
    输入: [batch, time_step, feature_size]
    输出: [batch, hidden_size*2] (默认为[batch, 512])
    """
    def __init__(self, in_channels, feature_size, hidden_size, time_step, num_coder_layer=6, num_heads=8):
        super(Neck, self).__init__()
        # 特征降维 [batch, time_step, feature_size] -> [batch, time_step, hidden_size]
        self.feature_proj = nn.Linear(feature_size, hidden_size)
        
        # 位置编码 [1, time_step, hidden_size] 训练可学习
        self.pos_encoding = nn.Parameter(torch.randn(1, time_step, hidden_size))
        
        # 编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(hidden_size, num_heads=num_heads, feedforward_size=hidden_size * 4)
            for _ in range(num_coder_layer)
        ])
        
        # 解码器层
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(hidden_size, num_heads=num_heads, feedforward_size=hidden_size * 4)
            for _ in range(num_coder_layer)
        ])
        
        # 输出投影 [batch, hidden_size] -> [batch, hidden_size*2]
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):  # x: [batch, time_step, feature_size]
        # 特征降维
        x = self.feature_proj(x)  # [batch, time_step, hidden_size]
        
        # 添加位置编码
        x = x + self.pos_encoding
        
        # 编码器层处理
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x) # [batch, time_step, hidden_size]
        
        #######################################################################
        # 取编码器输出的最后一个时间步的特征，作为解码器的输入之一
        # decoder_input = x[:, -1:, :]  # [batch, 1, hidden_size]，保持最后一维
        
        # 解码器输入是编码器输出的所有时间步长
        decoder_input = x  # [batch, time_step, hidden_size]
        #######################################################################
        
        # 解码器层处理
        for decoder_layer in self.decoder_layers:
            decoder_input = decoder_layer(decoder_input, x) # [batch, time_step, hidden_size]
        
        # 取最后一个时间步的特征并扩展维度
        final_feat = decoder_input[:, -1, :]  # [batch, hidden_size]
        enhanced_feat = self.output_proj(final_feat)  # [batch, hidden_size*2]
        
        return enhanced_feat


class SpeedHead(nn.Module):
   """
   速度预测头 - 标量回归任务
   输入: [batch, in_channels]
   输出: [batch, out_channels]
   """
   def __init__(self, in_channels, out_channels=1):
       super(SpeedHead, self).__init__()
       
       # 特征适应层 ## 调整特征的维度，使其与任务头相匹配 ### 后面可以改成特征自适应模块
       self.adapter = nn.Sequential(
           nn.Linear(in_channels, in_channels // 2),
           nn.BatchNorm1d(in_channels // 2),
           nn.SiLU()
       )
       
       # C2f风格模块 - 更窄的设计，因为是标量回归
       self.c2f = nn.ModuleList([
           # 分支1 - 特征处理
           nn.Sequential(
               nn.Linear(in_channels // 4, in_channels // 8),
               nn.BatchNorm1d(in_channels // 8),
               nn.SiLU(),
               nn.Linear(in_channels // 8, in_channels // 4),
               nn.BatchNorm1d(in_channels // 4),
               nn.SiLU()
           ),
           # 分支2 - 恒等映射
           nn.Identity()
       ])
       
       # 速度预测头 - 使用较小的网络 
       self.task_head = nn.Sequential(
           nn.Linear(in_channels // 2, in_channels // 4),
           nn.BatchNorm1d(in_channels // 4),
           nn.SiLU(),
           nn.Linear(in_channels // 4, out_channels)
       )
   def forward(self, x):
       # 特征适应
       x = self.adapter(x)
       
       # C2f风格处理
       x1, x2 = x.chunk(2, dim=1)
       x1 = self.c2f[0](x1)
       x2 = self.c2f[1](x2)
       x = torch.cat([x1, x2], dim=1)
       
       # 速度预测
       out = self.task_head(x)
       return out

class AngleHead(nn.Module):
   """
   角度预测头 - cos和sin预测任务
   输入: [batch, in_channels]
   输出: [batch, out_channels] (默认2，用于cos和sin)
   """
   def __init__(self, in_channels, out_channels=2):
       super(AngleHead, self).__init__()
       
       # 特征适应层 ## 调整特征的维度，使其与任务头相匹配 ### 后面可以改成特征自适应模块
       self.adapter = nn.Sequential(
           nn.Linear(in_channels, in_channels // 2),
           nn.BatchNorm1d(in_channels // 2),
           nn.SiLU()
       )
       
       # C2f风格模块 - 更宽的设计，因为需要预测两个相关值
       self.c2f = nn.ModuleList([
           # 分支1 - 特征处理
           nn.Sequential(
               nn.Linear(in_channels // 4, in_channels // 2),  # 更宽的中间层
               nn.BatchNorm1d(in_channels // 2),
               nn.SiLU(),
               nn.Linear(in_channels // 2, in_channels // 4),
               nn.BatchNorm1d(in_channels // 4),
               nn.SiLU()
           ),
           # 分支2 - 恒等映射
           nn.Identity()
       ])
       
       # 角度预测头 - 使用较大的网络以捕获cos和sin的关系
       self.task_head = nn.Sequential(
           nn.Linear(in_channels // 2, in_channels // 2),
           nn.BatchNorm1d(in_channels // 2),
           nn.SiLU(),
           nn.Linear(in_channels // 2, out_channels),
           nn.Tanh()  # 确保输出在[-1,1]范围内 ## 预测的是cos和sin
       )
   
   def forward(self, x):
       # 特征自适应
       x = self.adapter(x)
       
       # C2f风格处理
       x1, x2 = x.chunk(2, dim=1)
       x1 = self.c2f[0](x1)
       x2 = self.c2f[1](x2)
       x = torch.cat([x1, x2], dim=1)
       
       # 角度预测
       out = self.task_head(x)
       return out
