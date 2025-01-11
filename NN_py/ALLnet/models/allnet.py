import torch
import torch.nn as nn
from models.waveletblock import WaveletBlock
from models.priorblock import PriorGuidedBlock, FlowGuidedBranch, ClusterGuidedBranch
from models.head import HeadSharedFeature, SpeedHead, AngleHead

class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, ffn_expansion_factor=0.5, dropout=0.1):
        super().__init__()
        
        hidden_dim = int(in_channels * ffn_expansion_factor)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.ffn(x)

class ConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        # 原始尺度特征提取 (1x) - 使用深度可分离卷积
        self.conv_1x = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        
        # 1/2尺度特征提取 (2x) - 保留可学习的下采样
        self.down_2x = nn.Sequential(
            nn.Conv2d(channels, channels*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels*2),
            nn.GELU()
        )
        # 1/2尺度特征处理 - 使用深度可分离卷积
        self.conv_2x = nn.Sequential(
            nn.Conv2d(channels*2, channels*2, kernel_size=3, padding=1, groups=channels*2),
            nn.Conv2d(channels*2, channels*2, kernel_size=1),
            nn.BatchNorm2d(channels*2),
            nn.GELU()
        )
        
        # 1/4尺度特征提取 (4x) - 保留可学习的下采样
        self.down_4x = nn.Sequential(
            nn.Conv2d(channels*2, channels*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels*4),
            nn.GELU()
        )
        # 1/4尺度特征处理 - 使用深度可分离卷积
        self.conv_4x = nn.Sequential(
            nn.Conv2d(channels*4, channels*4, kernel_size=3, padding=1, groups=channels*4),
            nn.Conv2d(channels*4, channels*4, kernel_size=1),
            nn.BatchNorm2d(channels*4),
            nn.GELU()
        )
        
        # 上采样模块 - 保留可学习的上采样
        self.up_2x = nn.Sequential(
            nn.ConvTranspose2d(channels*4, channels*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels*2),
            nn.GELU()
        )
        self.up_1x = nn.Sequential(
            nn.ConvTranspose2d(channels*2, channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        
        # 最终特征融合 - 使用深度可分离卷积
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
    
    def forward(self, x):
        identity = x
        
        # 原始尺度 (1x)
        feat_1x = self.conv_1x(x)
        
        # 1/2尺度 (2x)
        feat_2x = self.down_2x(x)
        feat_2x = self.conv_2x(feat_2x)
        
        # 1/4尺度 (4x)
        feat_4x = self.down_4x(feat_2x)
        feat_4x = self.conv_4x(feat_4x)
        
        # 特征融合：自底向上
        out = self.up_2x(feat_4x)
        out = out + feat_2x  # 2x尺度跳跃连接
        
        out = self.up_1x(out)
        out = out + feat_1x  # 1x尺度跳跃连接
        
        out = self.fusion(out)
        out = out + identity  # 全局残差连接
        
        return out

class ALLBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ffn_expansion_factor=0.25, dropout=0.1):
        super().__init__()
        
        self.convblock = ConvBlock(in_channels)
        self.wave_block = WaveletBlock(dims=in_channels, num_bottlenecks=3)
        self.prior_block = PriorGuidedBlock(dims=in_channels)
        self.ffn = FFN(in_channels, out_channels, ffn_expansion_factor=ffn_expansion_factor, dropout=dropout)
    
    def forward(self, x, flow_attn, cluster_attn):
        
        x = self.convblock(x)
        wave_out = self.wave_block(x)
        prior_out = self.prior_block(wave_out, flow_attn, cluster_attn)
        out = self.ffn(prior_out)
        
        return out

class ALLNet(nn.Module):
    def __init__(self, in_channels=3, max_dims=48, num_blocks=3, channels=None, dropout=0.1):
        super().__init__()
        self.out_channels = 3
        if channels is not None:
            self.num_blocks = len(channels)
            self.channels = channels
        else:
            self.num_blocks = num_blocks
            assert max_dims % 3 == 0, "max_dims必须是3的倍数"
            # 计算每个block的通道数，确保是3的倍数
            self.channels = []
            current_dims = max_dims
            for _ in range(self.num_blocks):
                self.channels.append(current_dims)
                next_dims = int(current_dims * 0.8) // 3 * 3  # 确保是3的倍数
                current_dims = max(next_dims, 6)  # 最小保持6个通道
        
        # 网络结构
        # 输入卷积
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, max_dims, kernel_size=1),
            nn.BatchNorm2d(max_dims),
            nn.GELU()
        )
        
        # ALLBlocks和对应的先验分支
        self.all_blocks = nn.ModuleList([
            ALLBlock(
                in_channels=self.channels[i],
                out_channels=self.channels[i+1] if i < num_blocks-1 else self.channels[-1],
                ffn_expansion_factor=0.25,
                dropout=dropout
            ) for i in range(num_blocks)
        ])
        
        # 每个stage后的shortcut连接
        self.shortcuts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.channels[i+1], self.channels[i+1], kernel_size=1),
                nn.BatchNorm2d(self.channels[i+1])
            ) for i in range(num_blocks-1)
        ])
        
        # 光流先验分支
        self.flow_pbs = nn.ModuleList([
            FlowGuidedBranch(dims=self.channels[i]) 
            for i in range(num_blocks)
        ])
        
        # 聚类先验分支
        self.cluster_pbs = nn.ModuleList([
            ClusterGuidedBranch(dims=self.channels[i]) 
            for i in range(num_blocks)
        ])
        
        head_dims = self.channels[-1]
        self.shared_feat = HeadSharedFeature(head_dims, head_dims*2)
        self.speed_head = SpeedHead(head_dims*2, 1)
        self.angle_head = AngleHead(head_dims*2, 2)
    
    def forward(self, x, flow, cluster):
        # 主干特征提取
        feat = self.dwconv(x)
        
        # 初始化先验特征
        flow_feat = flow      # flow应该是[B, 3, H, W]格式的HSV光流
        cluster_feat = cluster  # cluster应该是[B, 1, H, W]格式的聚类图
        
        # 主干前向传播
        for i in range(self.num_blocks):
            # 获取并更新先验分支特征
            flow_attn, flow_feat = self.flow_pbs[i](flow_feat)  # flow_pbs期望输入是3通道HSV格式
            cluster_attn, cluster_feat = self.cluster_pbs[i](cluster_feat)  # cluster_pbs期望输入是1通道
            
            # 主干block处理
            feat = self.all_blocks[i](feat, flow_attn, cluster_attn)
            
            # shortcut连接
            if i < self.num_blocks - 1:
                shortcut = self.shortcuts[i](feat)
                feat = feat + shortcut
        
        # 全局平均池化
        feat = torch.mean(feat, dim=(2, 3))
        
        # 共享特征提取
        shared_feat = self.shared_feat(feat)
        
        # 预测速度和角度
        speed = self.speed_head(shared_feat)
        angle = self.angle_head(shared_feat)
        
        # 合并输出 [batch, 3]
        out = torch.cat([speed, angle], dim=1)
        
        return out

###########################################################

# def test_allnet():
#     # 初始化参数
#     batch_size = 2
#     in_channels = 3
#     height = 64
#     width = 64
#     dims = 48
#     num_blocks = 6
    
#     # 初始化模型
#     model = ALLNet(
#         in_channels=in_channels,
#         max_dims=dims,
#         num_blocks=num_blocks
#     )
    
#     # 创建测试输入
#     x = torch.randn(batch_size, in_channels, height, width)
#     flow = torch.randn(batch_size, 3, height, width)
#     cluster = torch.randn(batch_size, 1, height, width)
    
#     # 前向传播
#     out = model(x, flow, cluster)
#     speed = out[:, 0]
#     angle = out[:, 1:]
        
#     # 打印输入输出信息
#     print("=== 输入数据信息 ===")
#     print(f"输入图像尺寸: {x.shape}")
#     print(f"光流特征尺寸: {flow.shape}")
#     print(f"聚类特征尺寸: {cluster.shape}")
    
#     print("\n=== 输出数据信息 ===")
#     print(f"速度预测尺寸: {speed.shape}")
#     print(f"速度预测值:\n{speed.detach().numpy()}")
#     print(f"\n角度预测尺寸: {angle.shape}")
#     print(f"角度预测值:\n{angle.detach().numpy()}")
    
#     # 打印模型参数信息
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"\n=== 模型信息 ===")
#     print(f"总参数量: {total_params:,}")
    
#     # 测试不同输入尺寸
#     test_sizes = [(32, 32), (128, 128)]
#     print("\n=== 测试不同输入尺寸 ===")
#     for h, w in test_sizes:
#         x = torch.randn(batch_size, in_channels, h, w)
#         flow = torch.randn(batch_size, 3, h, w)
#         cluster = torch.randn(batch_size, 1, h, w)
#         speed, angle = model(x, flow, cluster)
#         print(f"\n输入尺寸 {h}x{w}:")
#         print(f"速度输出: {speed.shape}")
#         print(f"角度输出: {angle.shape}")

# if __name__ == '__main__':
#     test_allnet()