import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.darknet19 import make_layers, cfg  # 只导入make_layers函数和配置参数
from attentions import MultiHeadAttention,TemporalDiffAndCorrelation
#######################################################################################

#######################################################################################
class GTNet(nn.Module):
    """
    输入: [batch, time_step, channel, height, width]
    输出: [batch, 3] - out[:,0]为速度，out[:,1:]为角度(cos,sin)
    """
    def __init__(self, in_channels=3, out_channels=3, height=100, width=100, time_step=20):
        super(GTNet, self).__init__()
        
        self.out_channels = out_channels
        self.time_step = time_step

        # 输入层
        self.input_layer = InputLayer(in_channels,out_dim=32)  # 输入[batch,channel,h,w] -> 输出[batch,64,h,w]
        
        # 共享的特征提取层
        self.backbone = Backbone(input_dim=32)  # 输入[batch,64,h,w] -> 输出[batch,128,h/32,w/32]

        # 共享的特征融合层
        self.neck = Neck(
            in_channels=128,
            feature_size=128 * (height//32) * (width//32),  # 展平后的特征维度
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
        
    def forward(self, x):  # x: [batch, time_step, channel, height, width]

        batch_size = x.size(0)
        # 处理每个时间步
        features = []
        for t in range(self.time_step):
            # Input处理 [batch, channel, h, w] -> [batch, 32, h, w]
            x_t = self.input_layer(x[:,t,:,:,:])
            
            # Backbone特征提取 [batch, 32, h, w] -> [batch, 128, h/8, w/8]
            feat = self.backbone(x_t)
            # 展平成特征向量
            feat = feat.view(batch_size, -1)  # [batch, feature_size]
            features.append(feat)
            
        # 转换为时序特征 [batch, time_step, feature_size]
        features = torch.stack(features, dim=1)
        
        # Neck处理 [batch, time_step, feature_size] -> [batch, 512]
        feat_enhanced = self.neck(features)
        
        # head共享权重
        feat_shared = self.head_shared_feature(feat_enhanced)
                
        # 分别通过不同的任务头
        speed = self.speed_head(feat_shared)  # [batch, 1]
        angle = self.angle_head(feat_shared)  # [batch, 2]
        
        # 合并输出 [batch, 3]
        out = torch.cat([speed, angle], dim=1)
        
        return out


class InputLayer(nn.Module):
    """
    输入预处理模块 - 使用卷积实现Focus：将图像空间重组为通道
    输入: [batch, channel, height, width]
    输出: [batch, 32, height, width]
    """
    def __init__(self, in_channels, out_dim=32):
        super(InputLayer, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.SiLU(),
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x


class Backbone(nn.Module):
    """
    主干特征提取网络 - 使用Darknet19的特征提取部分
    输入: [batch, input_dim, height, width]
    输出: [batch, 128, height/32, width/32]
    """
    def __init__(self, input_dim):
        super(Backbone, self).__init__()
        
        # 直接构建Darknet19的特征提取部分
        self.features = make_layers(cfg, in_channels=input_dim, batch_norm=True)
        
        # 特征调整层(1024->128)
        self.adjust = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )
        
        # 全局上下文编码
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 32, 1),
            nn.SiLU(),
            nn.Conv2d(32, 128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Darknet特征提取
        x = self.features(x)  # [batch, 1024, h/32, w/32]
        
        # 调整特征通道
        x = self.adjust(x)    # [batch, 128, h/32, w/32]
        
        # 全局上下文注意力
        context = self.global_context(x)
        x = x * context
        
        return x

        
class EncoderLayer(nn.Module):
    """
    输入: [batch, time_step, hidden_size]
    输出: [batch, time_step, hidden_size]
    """
    def __init__(self, hidden_size, num_heads, feedforward_size, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = TemporalDiffAndCorrelation(
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
        self.self_attn = TemporalDiffAndCorrelation(
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
            # 分支2 - 恒等映��
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



