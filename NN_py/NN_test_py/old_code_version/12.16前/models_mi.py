import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from attentions import MultiHeadAttention, CBAM, AdaptiveDiffSimAttention
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
class MI_GTNet(nn.Module):
    """
    输入: [batch, time_step, channel, height, width]
    输出: [batch, 3] - out[:,0]为速度，out[:,1:]为角度(cos,sin)
    """
    # orig: 原图 argu: 增强图（聚类二值圆） dot: 一维数据（偏转长度）
    def __init__(self, in_channels_orig=3, in_channels_argu=None, in_channels_dot=None, out_channels=3, height=64, width=64, time_step=20):
        super().__init__()
        
        self.out_channels = out_channels
        self.time_step = time_step
        
        # 计算feature sizes
        self.orig_feature_size = 512 * (height//8) * (width//8)  # 32768 for 64x64
        self.argu_feature_size = 512 * (height//8) * (width//8) if in_channels_argu else 0
        self.dot_feature_size = 32768 if in_channels_dot else 0
        self.total_feature_size = self.orig_feature_size  # 改为只使用原始特征大小

        # 输入层
        self.input_layer_orig = InputLayer_orig(in_channels=in_channels_orig, out_dim=32)
        self.input_layer_argu = InputLayer_argu(in_channels=in_channels_argu, out_dim=32)
        self.input_layer_dot = InputLayer_dot(in_channels=in_channels_dot, out_dim=32)

        # 共享的特征提取层
        self.backbone_img = Backbone_img(input_dim=32) 
        self.backbone_dot = Backbone_dot(input_dim=32) 
        
        # 共享的特征融合层
        self.neck = Neck(
            in_channels=128,
            feature_size=self.total_feature_size,
            hidden_size=256,
            time_step=time_step
        )  # 输入[batch,time_step,feature_size] -> 输出[batch,512]
        
        self.head_shared_feature = HeadSharedFeature( # head共享权重
            in_channels=512,
            out_channels=256
        )
        # 分离的任务头
        self.speed_head = SpeedHead(
            in_channels=256,
            out_channels=1  # 速度预测
        )  # 输入[batch,512] -> 输出[batch,1]
        
        self.angle_head = AngleHead(
            in_channels=256,
            out_channels=out_channels-1  # cos和sin预测
        )  # 输入[batch,512] -> 输出[batch,2]
        
    def forward(self, x_orig, x_argu=None, x_dot=None):  # x: [batch, time_step, channel, height, width]

        batch_size = x_orig.size(0)
        
        # 1. 处理原始输入
        features_orig = []
        for t in range(self.time_step):
            x_t_orig = self.input_layer_orig(x_orig[:,t,:,:,:])
            feat_orig = self.backbone_img(x_t_orig)
            feat_orig = feat_orig.view(batch_size, -1)
            features_orig.append(feat_orig)
        features = torch.stack(features_orig, dim=1)
        
        # 2. 如果有辅助输入，处理并相加
        if x_argu is not None:
            features_argu = []
            for t in range(self.time_step):
                x_t_argu = self.input_layer_argu(x_argu[:,t,:,:,:])
                feat_argu = self.backbone_img(x_t_argu)
                feat_argu = feat_argu.view(batch_size, -1)
                features_argu.append(feat_argu)
            features_argu = torch.stack(features_argu, dim=1)
            features = features + features_argu
            
        if x_dot is not None:
            features_dot = []
            for t in range(self.time_step):
                x_t_dot = self.input_layer_dot(x_dot[:,t,:])
                feat_dot = self.backbone_dot(x_t_dot)
                feat_dot = feat_dot.view(batch_size, -1)
                features_dot.append(feat_dot)
            features_dot = torch.stack(features_dot, dim=1)
            features = features + features_dot

        # Neck处理 [batch, time_step, feature_size] -> [batch, 512]
        feat_enhanced = self.neck(features)
        feat_shared = self.head_shared_feature(feat_enhanced)
                
        # 分别通过不同的任务头
        speed = self.speed_head(feat_shared)  # [batch, 1]
        angle = self.angle_head(feat_shared)  # [batch, 2]
        
        # 合并输出 [batch, 3]
        out = torch.cat([speed, angle], dim=1)
        
        return out


class InputLayer_orig(nn.Module):
    def __init__(self, in_channels=3, out_dim=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        self.adaptive = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.out_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.adaptive(x)


class InputLayer_argu(nn.Module):
    def __init__(self, in_channels=1, out_dim=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        self.adaptive = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.out_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.adaptive(x)


class InputLayer_dot(nn.Module):
    def __init__(self, in_channels=1, out_dim=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        self.adaptive = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.out_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.adaptive(x)



class Backbone_img(nn.Module):
    """
    主干特征提取网络
    输入: [batch, input_dim, height, width]
    输出: [batch, 512, height/8, width/8]
    """
    def __init__(self, input_dim):
        super().__init__()
        
        # Stage 1: [batch, input_dim, h, w] -> [batch, 64, h/2, w/2]
        self.stage1 = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
          
        # Stage 2: [batch, 64, h/2, w/2] -> [batch, 128, h/4, w/4]
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Stage 3: [batch, 128, h/4, w/4] -> [batch, 512, h/8, w/8]
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(), 
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 全局上下文编码
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 128, 1),
            nn.SiLU(),
            nn.Conv2d(128, 512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Stage 1
        x = self.stage1(x)
        
        # Stage 2 
        x = self.stage2(x)
        
        # Stage 3
        x = self.stage3(x)
        
        # 全局上下文注意力
        context = self.global_context(x)
        x = x * context

        return x  # [batch, 512, h/8, w/8] flatten后 [batch, 512*h/8*w/8]
    


class Backbone_dot(nn.Module):
    """
    输入: [batch, input_dim, 1] # l = 1
    输出: [batch, 512, 1]
    """
    def __init__(self, input_dim):
        super().__init__()
        
        self.stage1 = nn.Sequential(
            nn.Conv1d(input_dim, 32768, kernel_size=1, padding=0),
            nn.BatchNorm1d(32768),
            nn.ReLU(),
        )
          
    def forward(self, x):
        # Stage 1
        x = self.stage1(x)

        return x # [batch, 64, 1] flatten后 [batch, 64*1]

        
class EncoderLayer(nn.Module):
    """
    输入: [batch, time_step, hidden_size]
    输出: [batch, time_step, hidden_size]
    """
    def __init__(self, hidden_size, num_heads, feedforward_size, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = AdaptiveDiffSimAttention(
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
        self.self_attn = AdaptiveDiffSimAttention(
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
        # Self-attention
        attn_output, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            # attn_mask=mask
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
    

class Neck(nn.Module):
    """
    特征融合与增强模块 - 使用Transformer结构
    输入: [batch, time_step, feature_size]
    输出: [batch, hidden_size*2] (默认为[batch, 512])
    """
    def __init__(self, in_channels, feature_size, hidden_size, time_step, num_coder_layer=3, num_heads=8):
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
            nn.Linear(time_step*hidden_size, hidden_size * 2),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):  # x: [batch, time_step, feature_size]
        batch_size = x.size(0)
        # 特征降维
        x = self.feature_proj(x)  # [batch, time_step, hidden_size]
        
        # 添加位置编码
        x = x + self.pos_encoding
        
        # 编码器层
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x) # [batch, time_step, hidden_size]
        
        # 解码器输入是编码器输出的所有时间步长
        decoder_input = x  # [batch, time_step, hidden_size]
        
        # 解码器层处理
        for decoder_layer in self.decoder_layers:
            decoder_input = decoder_layer(decoder_input, x) # [batch, time_step, hidden_size]
        
        # # 平均池化时序 ### 需改输出投影连接层的参数
        # final_feat = torch.mean(decoder_input, dim=1)  # [batch, hidden_size]
        # reshape保留全部时序
        final_feat = decoder_input.reshape(batch_size, -1) # [batch, time_step*hidden_size]
        enhanced_feat = self.output_proj(final_feat)  # [batch, hidden_size*2]
        
        return enhanced_feat

class HeadSharedFeature(nn.Module):
    """
    检测头共享权重
    """
    def __init__(self, in_channels, out_channels):
        super(HeadSharedFeature, self).__init__()
        self.shared_connect = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.SiLU()
        )
    def forward(self, x):
        x = self.shared_connect(x)
        return x

class SpeedHead(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(SpeedHead, self).__init__()
        
        # skip connection
        self.skip_connect = nn.ModuleList([
            # 分支1 - 特征处理
            nn.Sequential(
                nn.Linear(in_channels // 2, in_channels // 4),  # 128->64
                nn.BatchNorm1d(in_channels // 4),
                nn.SiLU(),
                nn.Linear(in_channels // 4, in_channels // 2),  # 64->128
                nn.BatchNorm1d(in_channels // 2),
                nn.SiLU()
            ),
            # 分支2 - 恒等映射
            nn.Identity()
        ])
        
        # 速度预测头
        self.task_head = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),  # 256->128
            nn.BatchNorm1d(in_channels // 2),
            nn.SiLU(),
            nn.Linear(in_channels // 2, out_channels)  # 128->1
        )

    def forward(self, x):
        # C2f风格处理
        x1, x2 = x.chunk(2, dim=1)  # 每个分支512/2=256维
        x1 = self.skip_connect[0](x1)  # 256维
        x2 = self.skip_connect[1](x2)  # 256维
        x = torch.cat([x1, x2], dim=1)  # 512维
        
        # 速度预测
        out = self.task_head(x)
        return out

class AngleHead(nn.Module):
    def __init__(self, in_channels, out_channels=2):
        super(AngleHead, self).__init__()
        
        # skip connection
        self.skip_connect = nn.ModuleList([
            # 分支1 - 特征处理
            nn.Sequential(
                nn.Linear(in_channels // 2, in_channels // 4),  # 128->64
                nn.BatchNorm1d(in_channels // 4),
                nn.SiLU(),
                nn.Linear(in_channels // 4, in_channels // 2),  # 64->128
                nn.BatchNorm1d(in_channels // 2),
                nn.SiLU()
            ),
            # 分支2 - 恒等映射
            nn.Identity()
        ])
        
        # 角度预测头
        self.task_head = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),  # 256->128
            nn.BatchNorm1d(in_channels // 2),
            nn.SiLU(),
            nn.Linear(in_channels // 2, out_channels),  # 128->2
            nn.Tanh()  # 确保cos和sin输出在[-1,1]范围内
        )
    
    def forward(self, x):
        # C2f风格处理
        x1, x2 = x.chunk(2, dim=1)  # 每个分支512/2=256维
        x1 = self.skip_connect[0](x1)  # 256维
        x2 = self.skip_connect[1](x2)  # 256维
        x = torch.cat([x1, x2], dim=1)  # 512维
        
        # 角度预测
        out = self.task_head(x)
        return out



