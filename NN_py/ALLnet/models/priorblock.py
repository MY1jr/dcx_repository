import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cspblock import CSPBlock

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect'),
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.BatchNorm2d(out_chans * patch_size ** 2),
            nn.Conv2d(out_chans, out_chans, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect')
        )

    def forward(self, x):
        x = self.proj(x)
        return x
    
class HSVChannelEmbed(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, embed_dim=96, kernel_size=None):
        super().__init__()
        if kernel_size is None:
            kernel_size = 1
            
        # 为HSV每个通道创建独立的处理路径
        self.h_proj = nn.Sequential(
            nn.Conv2d(in_channels//3, embed_dim//3, kernel_size=kernel_size, stride=patch_size,
                               padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect'),
            nn.BatchNorm2d(embed_dim//3)
        )
        self.s_proj = nn.Sequential(
            nn.Conv2d(in_channels//3, embed_dim//3, kernel_size=kernel_size, stride=patch_size,
                               padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect'),
            nn.BatchNorm2d(embed_dim//3)
        )
        self.v_proj = nn.Sequential(
            nn.Conv2d(in_channels//3, embed_dim//3, kernel_size=kernel_size, stride=patch_size,
                               padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect'),
            nn.BatchNorm2d(embed_dim//3)
        )

    def forward(self, x):
        # 分离HSV通道
        h, s, v = torch.split(x, x.size(1)//3, dim=1)
        # 分别处理并保持通道语义
        return torch.cat([
            self.h_proj(h),
            self.s_proj(s),
            self.v_proj(v)
        ], dim=1)

class HSVChannelUnEmbed(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, embed_dim=96, kernel_size=None):
        super().__init__()
        if kernel_size is None:
            kernel_size = 1
            
        # 为HSV每个通道创建独立的还原路径
        self.h_proj = nn.Sequential(
            nn.Conv2d(embed_dim//3, patch_size**2, kernel_size=kernel_size,
                     padding=kernel_size//2, padding_mode='reflect'),
            nn.BatchNorm2d(patch_size**2),
            nn.Conv2d(patch_size**2, in_channels//3, kernel_size=kernel_size,
                     padding=kernel_size//2, padding_mode='reflect')
        )
        self.s_proj = nn.Sequential(
            nn.Conv2d(embed_dim//3, patch_size**2, kernel_size=kernel_size,
                     padding=kernel_size//2, padding_mode='reflect'),
            nn.BatchNorm2d(patch_size**2),
            nn.Conv2d(patch_size**2, in_channels//3, kernel_size=kernel_size,
                     padding=kernel_size//2, padding_mode='reflect')
        )
        self.v_proj = nn.Sequential(
            nn.Conv2d(embed_dim//3, patch_size**2, kernel_size=kernel_size,
                     padding=kernel_size//2, padding_mode='reflect'),
            nn.BatchNorm2d(patch_size**2),
            nn.Conv2d(patch_size**2, in_channels//3, kernel_size=kernel_size,
                     padding=kernel_size//2, padding_mode='reflect')
        )

    def forward(self, x):
        # 分离嵌入特征
        h_feat, s_feat, v_feat = torch.split(x, x.size(1)//3, dim=1)
        # 分别还原各通道
        h = self.h_proj(h_feat)
        s = self.s_proj(s_feat)
        v = self.v_proj(v_feat)
        # 合并通道
        return torch.cat([h, s, v], dim=1)
    
class ClusterAttention(nn.Module):
    def __init__(self, dims=64, kernel_sizes=[3,5,7]):
        super().__init__()
        
        # 多尺度空间注意力
        self.spatial_attns = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dims, dims//4, k, padding=k//2),
                nn.BatchNorm2d(dims//4),
                nn.GELU(),
                nn.Conv2d(dims//4, 1, k, padding=k//2)
            ) for k in kernel_sizes
        ])
        
        # 注意力图融合
        self.attn_fusion = nn.Sequential(
            nn.Conv2d(len(kernel_sizes), 1, 1),
            nn.Sigmoid()
        )
        
        # 特征增强
        self.feat_enhance = nn.Sequential(
            nn.Conv2d(dims, dims, 3, padding=1, groups=dims),
            nn.BatchNorm2d(dims),
            nn.GELU(),
            nn.Conv2d(dims, dims, 1),
            nn.BatchNorm2d(dims)
        )
        
    def forward(self, cluster):
        # 2. 生成多尺度空间注意力图
        cluster_weights = []
        for attn in self.spatial_attns:
            weight = attn(cluster)
            cluster_weights.append(weight)
        
        # 3. 融合多尺度注意力图
        cluster_attn = self.attn_fusion(torch.cat(cluster_weights, dim=1))
        
        return cluster_attn


class FlowAttention(nn.Module):
    def __init__(self, dims=96):
        super().__init__()
        
        # H通道(方向)注意力
        self.direction_attn = nn.Sequential(
            # 使用较大卷积核捕获方向信息
            nn.Conv2d(dims//3, dims//6, 5, padding=2),
            nn.BatchNorm2d(dims//6),
            nn.GELU(),
            nn.Conv2d(dims//6, dims//3, 1),
            nn.Sigmoid()
        )
        
        # S通道(饱和度)注意力
        self.saturation_attn = nn.Sequential(
            nn.Conv2d(dims//3, dims//6, 3, padding=1),
            nn.BatchNorm2d(dims//6),
            nn.GELU(),
            nn.Conv2d(dims//6, dims//3, 1),
            nn.Sigmoid()
        )
        
        # V通道(运动幅度)注意力
        self.magnitude_attn = nn.Sequential(
            # 使用空洞卷积增大感受野
            nn.Conv2d(dims//3, dims//6, 3, padding=2, dilation=2),
            nn.BatchNorm2d(dims//6),
            nn.GELU(),
            nn.Conv2d(dims//6, dims//3, 1),
            nn.Sigmoid()
        )
        
        
    def forward(self, hsv_flow):
        # 分离HSV通道
        h, s, v = torch.split(hsv_flow, hsv_flow.size(1)//3, dim=1)
        # 计算各通道注意力权重
        dir_weight = self.direction_attn(h)     # 方向注意力
        sat_weight = self.saturation_attn(s)    # 饱和度注意力
        mag_weight = self.magnitude_attn(v)     # 幅度注意力
        # 组合注意力
        attention = torch.cat([dir_weight, sat_weight, mag_weight], dim=1)
            
        return attention

class FlowGuidedBranch(nn.Module):
    def __init__(self, dims):
        super().__init__()
        
        # 光流卷积嵌入
        self.flow_conv = HSVChannelEmbed(
            patch_size=1,
            embed_dim=dims,
            kernel_size=3
        )
    
        # 光流路径空间注意力
        self.flow_attn = FlowAttention(dims)
        
        # 光流反卷积输出
        self.flow_iconv = HSVChannelUnEmbed(
            patch_size=1,
            embed_dim=dims,
            kernel_size=3
        )
        
    def forward(self, flow):
        # 光流处理分支
        flow_map = self.flow_conv(flow)
        attn = self.flow_attn(flow_map)
        
        # 光流路径输出
        flow_out = self.flow_iconv(flow_map)
        
        return attn, flow_out

class ClusterGuidedBranch(nn.Module):
    def __init__(self, dims):
        super().__init__()
        
        # 聚类卷积嵌入
        self.cluster_conv = PatchEmbed(
            patch_size=1,
            in_chans=3,
            embed_dim=dims,
            kernel_size=3
        )
        
        # 聚类路径空间注意力
        self.cluster_attn = ClusterAttention(dims)
        
        # 聚类反卷积输出
        self.cluster_iconv = PatchUnEmbed(
            patch_size=1,
            out_chans=3,
            embed_dim=dims,
            kernel_size=3
        )
        
    def forward(self, cluster):
        # 聚类处理分支
        cluster_map = self.cluster_conv(cluster)
        attn = self.cluster_attn(cluster_map)
        
        # 聚类路径输出
        cluster_out = self.cluster_iconv(cluster_map)
        
        return attn, cluster_out

class PriorGuidedBlock(nn.Module):
    def __init__(self, dims, dropout=0.1):
        super().__init__()
        
        # 主路径CSP
        self.main_csp = CSPBlock(dims, dims, num_bottlenecks=3, expansion_factor=0.5)
        
        # 先验分支
        self.flow_branch = FlowGuidedBranch(dims)
        self.cluster_branch = ClusterGuidedBranch(dims)
        
        # 主路径特征融合
        self.main_fusion = nn.Sequential(
            nn.Conv2d(dims*3, dims, 1),
            nn.BatchNorm2d(dims),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 输出归一化
        self.norm = nn.BatchNorm2d(dims)
        
    def forward(self, x, flow_attn, cluster_attn):
        identity = x
        
        # 主路径CSP特征
        main_feat = self.main_csp(x)
        # 先验路径shortcut
        flow_feat = x * flow_attn + identity
        cluster_feat = x * cluster_attn + identity
        
        # 主路径特征融合
        main_out = self.main_fusion(torch.cat([main_feat, flow_feat, cluster_feat], dim=1))
        out = self.norm(main_out+identity)
        
        return out

#####################################

def test_prior_guided_block():
    # 设置随机种子以保证结果可复现
    torch.manual_seed(42)
    
    # 定义测试参数
    batch_size = 2
    channels = 36  # 特征维度
    height = 32
    width = 32
    
    # 初始化模块
    flow_branch = FlowGuidedBranch(dims=channels)
    cluster_branch = ClusterGuidedBranch(dims=channels)
    block = PriorGuidedBlock(dims=channels)
    
    # 创建模拟输入并启用梯度
    x = torch.randn(batch_size, channels, height, width, requires_grad=True)
    flow = torch.randn(batch_size, 3, height, width, requires_grad=True)  # HSV格式的光流
    cluster = torch.randn(batch_size, 3, height, width, requires_grad=True)  # 聚类结果
    
    # 运行前向传播
    flow_attn, flow_out = flow_branch(flow)
    cluster_attn, cluster_out = cluster_branch(cluster)
    main_out = block(x, flow_attn, cluster_attn)

    # 输出维度
    print("主路径输出维度:", main_out.shape)
    print("期望主路径维度:", (batch_size, channels, height, width))
    
    print("光流路径输出维度:", flow_out.shape) 
    print("期望光流路径维度:", (batch_size, 3, height, width))
    
    print("聚类路径输出维度:", cluster_out.shape)
    print("期望聚类路径维度:", (batch_size, 1, height, width))
    
    # 计算梯度
    loss = main_out.sum() + flow_out.sum() + cluster_out.sum()
    loss.backward()
    
    # 输出梯度值
    print("输入特征x的梯度形状:", x.grad.shape)
    print("光流输入的梯度形状:", flow.grad.shape) 
    print("聚类输入的梯度形状:", cluster.grad.shape)
    
    print("输入特征x的梯度均值:", x.grad.mean().item())
    print("光流输入的梯度均值:", flow.grad.mean().item())
    print("聚类输入的梯度均值:", cluster.grad.mean().item())
    
    print("所有测试通过！")

if __name__ == "__main__":
    test_prior_guided_block()