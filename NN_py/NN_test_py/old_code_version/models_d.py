import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from attentions import MultiHeadAttention, CBAM, FrameDiffAttention, BADSAttention
#######################################################################################
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

#######################################################################################
class GTNet(nn.Module):
    """
    输入: [batch, time_step, channel, height, width]
    输出: [batch, 3] - out[:,0]为速度，out[:,1:]为角度(cos,sin)
    """
    def __init__(self, in_channels=3, out_channels=3, height=64, width=64, time_step=20):
        super().__init__()
        
        self.out_channels = out_channels
        self.time_step = time_step

        # 输入层
        self.input_layer = InputLayer(in_channels,out_dim=32)  # 输入[batch,channel,h,w] -> 输出[batch,32,h,w]
        
        # 共享的特征提取层
        self.backbone_1 = ConvGroup1(input_dim=32)  # 输入[batch,32,h,w] -> 输出[batch, 256, height/4, width/4]

        self.FDattention = FrameDiffAttention( # 输入[batch,256,h/4,w/4] -> 输出[batch,256,h/4,w/4]
            embed_dim=256,
            window_size=3,
            dropout=0.1,
            batch_first=True
        )  
        self.backbone_2 = ConvGroup2(input_dim=256) # 输入[batch,256,h/4,w/4] -> 输出[batch,128,h/8,w/8]
        
        # 共享的特征融合层
        self.neck = EncoderDecoderGroup( # 输入[batch,time_step,feature_size] -> 输出[batch,512]
            feature_size=128 * (height//8) * (width//8),  # 展平后的特征维度
            hidden_size=256,
            time_step=time_step
        )  
        
        self.head_shared_proj = HeadSharedFeature( # head共享权重
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
        # 输入处理 [batch, time_step, channel, height, width] -> [batch, time_step, 32, height, width]
        x = self.input_layer(x)
        
        x_t_state = []
        features = []
        # stage1 处理每个时间步
        for t in range(self.time_step):
            x_t = x[:,t,:,:,:]  # [batch, 32, h, w]
            # Backbone特征提取 [batch, 32, h, w] -> [batch, 128, h/8, w/8]
            x_t = self.backbone_1(x_t)  # [batch, 128, h/8, w/8]
            x_t_state.append(x_t)
        
        # 重组回时序 [batch, time_step, 128, h/8, w/8]
        x = torch.stack(x_t_state, dim=1)
        
        # 帧间注意力 [batch, time_step, 128, h/8, w/8] -> [batch, time_step, 128, h/8, w/8]
        x = self.FDattention(x)
        # identity = x
        # x = self.FDattention(x) + identity  # 残差连接
        
        # stage2 处理每个时间步
        for t in range(self.time_step):
            x_t = x[:,t,:,:,:]  # [batch, 128, h/8, w/8]
            # Backbone特征提取 [batch, 128, h/8, w/8] -> [batch, 512, h/8, w/8]
            x_t = self.backbone_2(x_t)  # [batch, 512, h/8, w/8]
            
            # 展平成特征向量
            feat = x_t.view(batch_size, -1)  # [batch, feature_size]
            features.append(feat)
            
        # 转换为时序特征 [batch, time_step, feature_size]
        features = torch.stack(features, dim=1)
        
        # Neck处理 [batch, time_step, feature_size] -> [batch, 512]
        feat_enhanced = self.neck(features)
        
        # head共享权重
        feat_shared = self.head_shared_proj(feat_enhanced)
                
        # 分别通过不同的任务头
        speed = self.speed_head(feat_shared)  # [batch, 1]
        angle = self.angle_head(feat_shared)  # [batch, 2]
        
        # 合并输出 [batch, 3]
        out = torch.cat([speed, angle], dim=1)
        
        return out

class InputLayer(nn.Module):
    """
    输入预处理模块 - 3D卷积
    输入: [batch, time_step, channel, height, width]
    输出: [batch, time_step, 32, height, width]
    """
    def __init__(self, in_channels=3, out_dim=32):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_dim, 
                     kernel_size=(1,1,1),  # 时间维度kernel_size=1
                     stride=1, 
                     padding=0, 
                     bias=False),
            nn.BatchNorm3d(out_dim),
            nn.SiLU(),
        )
        
    def forward(self, x):
        # 输入x需要调整维度顺序：[batch, time_step, channel, h, w] -> [batch, channel, time_step, h, w]
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv(x)
        # 输出调整回原来的维度顺序：[batch, channel, time_step, h, w] -> [batch, time_step, channel, h, w]
        x = x.permute(0, 2, 1, 3, 4)
        return x
        

class ConvGroup1(nn.Module):
    """
    输入: [batch, input_dim, height, width]
    输出: [batch, 256, height/4, width/4]
    """
    def __init__(self, input_dim):
        super().__init__()
        
        # Stage 1: [batch, input_dim, h, w] -> [batch, 64, h/2, w/2]
        self.stage1_main = nn.Sequential(
            nn.Conv2d(input_dim//2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.stage1_direct = nn.Sequential(
            nn.Conv2d(input_dim - input_dim//2, 32, kernel_size=1),
            nn.MaxPool2d(2)
        )
          
        # Stage 2: [batch, 64, h/2, w/2] -> [batch, 128, h/2, w/2]
        self.stage2_main = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.stage2_direct = nn.Conv2d(32, 64, kernel_size=1)
        
        # Stage 3: [batch, 128, h/2, w/2] -> [batch, 256, h/4, w/4]
        self.stage3_main = nn.Sequential(
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
        self.stage3_direct = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.MaxPool2d(2)
        )
        
        # 转换层
        self.transition = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, x):
        # Stage 1 - CSP style
        x1, x2 = torch.split(x, [x.size(1)//2, x.size(1)-x.size(1)//2], dim=1)
        stage1_main = self.stage1_main(x1)
        stage1_direct = self.stage1_direct(x2)
        stage1_out = torch.cat([stage1_main, stage1_direct], dim=1)  # 64通道
        
        # Stage 2 - CSP style 
        x1, x2 = torch.split(stage1_out, [stage1_out.size(1)//2, stage1_out.size(1)-stage1_out.size(1)//2], dim=1)
        stage2_main = self.stage2_main(x1)
        stage2_direct = self.stage2_direct(x2)
        stage2_out = torch.cat([stage2_main, stage2_direct], dim=1)  # 128通道
        
        # Stage 3 - CSP style
        x1, x2 = torch.split(stage2_out, [stage2_out.size(1)//2, stage2_out.size(1)-stage2_out.size(1)//2], dim=1)
        stage3_main = self.stage3_main(x1)
        stage3_direct = self.stage3_direct(x2)
        stage3_out = torch.cat([stage3_main, stage3_direct], dim=1)  # 256通道
        
        # 特征转换
        out = self.transition(stage3_out)
        
        return out

class ConvGroup2(nn.Module):
    """
    输入: [batch, input_dim, height, width]
    输出: [batch, 128, height/8, width/8]
    """
    def __init__(self, input_dim):
        super().__init__()
        
        # Stage 1: [batch, 256, h/4, w/4] -> [batch, 512, h/4, w/4]
        self.stage1_main = nn.Sequential(
            nn.Conv2d(input_dim//2, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.stage1_direct = nn.Conv2d(input_dim - input_dim//2, 256, kernel_size=1)
        
        # Stage 2: [batch, 512, h/4, w/4] -> [batch, 256, h/4, w/4]
        self.stage2_main = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.stage2_direct = nn.Conv2d(256, 128, kernel_size=1)
        
        # Stage 3: [batch, 256, h/4, w/4] -> [batch, 128, h/8, w/8]
        self.stage3_main = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.stage3_direct = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.MaxPool2d(2)
        )
                
        # 全局上下文编码
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 64, 1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Stage 1 - CSP style
        x1, x2 = torch.split(x, [x.size(1)//2, x.size(1)-x.size(1)//2], dim=1)
        stage1_main = self.stage1_main(x1)
        stage1_direct = self.stage1_direct(x2)
        stage1_out = torch.cat([stage1_main, stage1_direct], dim=1)  # 512通道
        
        # Stage 2 - CSP style
        x1, x2 = torch.split(stage1_out, [stage1_out.size(1)//2, stage1_out.size(1)-stage1_out.size(1)//2], dim=1)
        stage2_main = self.stage2_main(x1)
        stage2_direct = self.stage2_direct(x2)
        stage2_out = torch.cat([stage2_main, stage2_direct], dim=1)  # 256通道
        
        # Stage 3 - CSP style
        x1, x2 = torch.split(stage2_out, [stage2_out.size(1)//2, stage2_out.size(1)-stage2_out.size(1)//2], dim=1)
        stage3_main = self.stage3_main(x1)
        stage3_direct = self.stage3_direct(x2)
        stage3_out = torch.cat([stage3_main, stage3_direct], dim=1)  # 128通道
        
        # 全局上下文注意力
        context = self.global_context(stage3_out)
        out = stage3_out * context

        return out
        
class EncoderLayer(nn.Module):
    """
    输入: [batch, time_step, embed_dim]
    输出: [batch, time_step, embed_dim]
    """
    def __init__(self, embed_dim, num_heads, feedforward_size, dropout=0.1):
        super().__init__()
        self.self_attn = BADSAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_size, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
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
    x: [batch, 1, embed_dim]
    encoder_output: [batch, time_step, embed_dim]
    输出：
    x: [batch, time_step, embed_dim]
    """
    def __init__(self, embed_dim, num_heads, feedforward_size, dropout=0.1):
        super().__init__()
        self.self_attn = BADSAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_size, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
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
    

class EncoderDecoderGroup(nn.Module):
    """
    特征融合与增强模块 - 使用Transformer结构
    输入: [batch, time_step, feature_size]
    输出: [batch, hidden_size*2] (默认为[batch, 512])
    """
    def __init__(self, feature_size, hidden_size, time_step, num_coder_layer=3, num_heads=8):
        super().__init__()
        # 特征降维 [batch, time_step, feature_size] -> [batch, time_step, hidden_size]
        self.feature_proj = nn.Linear(feature_size, hidden_size)
        
        # 位置编码 [1, time_step, hidden_size] 训练可学习
        self.pos_encoding = nn.Parameter(torch.randn(1, time_step, hidden_size))
        
        # 编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(embed_dim=hidden_size, num_heads=num_heads, feedforward_size=hidden_size * 4)
            for _ in range(num_coder_layer)
        ])
        
        # 解码器层
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embed_dim=hidden_size, num_heads=num_heads, feedforward_size=hidden_size * 4)
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
        super().__init__()
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
        super().__init__()
        
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
        super().__init__()
        
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



