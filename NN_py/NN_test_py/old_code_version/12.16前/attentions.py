import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttention(nn.Module): # [batch_size, seq_len, embed_dim] (B,T,C)
    """
    自定义多头注意力模块
    输入: [batch_size, seq_len, embed_dim]
    输出: [batch_size, seq_len, embed_dim]
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1, batch_first=True):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embed_dim must be divisible by num_heads!"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5 # 根下dk
        
        # Q、K、V的线性变换层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.batch_first = batch_first
        
    def forward(self, query, key, value, attn_mask=None):
        """
        参数:
            query: [batch_size, seq_len_q, embed_dim]
            key: [batch_size, seq_len_k, embed_dim]
            value: [batch_size, seq_len_v, embed_dim]
            attn_mask: [seq_len_q, seq_len_k] 或 [batch_size, seq_len_q, seq_len_k]
        """
        
        # 如果batch_first=False，转换输入维度
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        batch_size = query.size(0)
        
        # 线性变换并分头
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 处理mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
        
        # 注意力softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        
        # 注意力加权
        attn_output = torch.matmul(attn_weights, v)
        
        # 重排维度并合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim)
        
        # 输出投影
        attn_output = self.out_proj(attn_output)
        attn_output = self.proj_drop(attn_output)
        return attn_output, attn_weights
    
#######################################################################################
#############################################
#######################################################################################
# CBAM注意力模块（通道注意力与空间注意力）
class ChannelAttention(nn.Module):
    def __init__(self, embed_dim, reduction_ratio=16, batch_first=True):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        
        # 全连接层mlp
        self.shared_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // reduction_ratio),
            nn.ReLU(),
            nn.Linear(embed_dim // reduction_ratio, embed_dim)
        )
        self.sigmoid = nn.Sigmoid()
        self.batch_first = batch_first

    def forward(self, x):
        if not self.batch_first:
            x = x.transpose(0, 1)
        B, T, C, H, W = x.size()
        
        # 重塑以处理每个时间步
        x = x.view(B * T, C, H, W)

        # 计算池化特征
        avg_pool = self.avg_pool(x).view(B * T, C)
        max_pool = self.max_pool(x).view(B * T, C)
        
        # 共享MLP
        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)
        
        # 注意力权重
        attention = self.sigmoid(avg_out + max_out).view(B * T, C, 1, 1)
        
        # 应用注意力
        x = x * attention
        
        # 恢复原始维度
        x = x.view(B, T, C, H, W)
        return x, attention

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, batch_first=True):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2 )
        self.sigmoid = nn.Sigmoid()
        self.batch_first = batch_first

    def forward(self, x):
        if not self.batch_first:
            x = x.transpose(0, 1)
        B, T, C, H, W = x.size()
        
        # 重塑以处理每个时间步
        x = x.view(B * T, C, H, W)
        
        # 计算空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        
        # 应用注意力
        x = x * attention
        
        # 恢复原始维度
        x = x.view(B, T, C, H, W)
        
        return x, attention

class CBAM(nn.Module):
    def __init__(self, embed_dim, reduction_ratio=16, kernel_size=7, batch_first=True):
        super().__init__()
        self.channel_attention = ChannelAttention(embed_dim, reduction_ratio, batch_first)
        self.spatial_attention = SpatialAttention(kernel_size, batch_first)
        self.batch_first = batch_first
        self.embed_dim = embed_dim

    def forward(self, x):  # [B,T,C,H,W]
        if not self.batch_first:
            x = x.transpose(0, 1)

        # 通道注意力
        x, channel_attention = self.channel_attention(x)
        
        # 空间注意力 
        x, spatial_attention = self.spatial_attention(x)
        
        attn_weights = channel_attention * spatial_attention
        return x, attn_weights



class SpatialMutiAttention(nn.Module):
    """多头空间注意力模块 - 支持自注意力和交叉注意力"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, batch_first=True):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.scale = (embed_dim // num_heads) ** -0.5
        self.batch_first = batch_first
        
        # QKV投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 输出投影
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, query, key=None, value=None):  # query: [B,T,C,H,W]
        """
        参数:
            query: [B,T,C,H,W] - 查询张量
            key: [B,T,C,H,W] - 键张量(可选，默认None表示自注意力)
            value: [B,T,C,H,W] - 值张量(可选，默认None表示自注意力)
        返回:
            output: 与query相同shape
            attention: 注意力权重
        """
        # 自注意力模式
        if key is None:
            key = query
        if value is None:
            value = query
            
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            
        B, T, C, H, W = query.shape
        
        # 重塑输入，将空间维度展平
        q = query.view(B*T, C, H*W).transpose(1, 2)  # [B*T,H*W,C]
        k = key.view(B*T, C, H*W).transpose(1, 2)    # [B*T,H*W,C]
        v = value.view(B*T, C, H*W).transpose(1, 2)  # [B*T,H*W,C]
        
        # QKV投影
        q = self.q_proj(q)  # [B*T,H*W,C]
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # 重塑为多头形式
        q = q.reshape(B*T, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B*T, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B*T, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B*T,num_heads,H*W,H*W]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 注意力加权
        x = (attn @ v).transpose(1, 2).reshape(B*T, H*W, C)
        
        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # 恢复空间维度
        x = x.transpose(1, 2).view(B, T, C, H, W)
        
        return x, attn

class TemporalDiffMutiAttention(nn.Module): 
    """
    多头时序差分注意力模块 - 支持自注意力和交叉注意力
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, batch_first=True):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.scale = (embed_dim // num_heads) ** -0.5
        self.batch_first = batch_first
        
        # 时序特征提取
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        
        # 注意力层
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
    def _extract_temporal_features(self, x):
        """提取时序特征方法"""
        B, T, C, H, W = x.shape
        
        # 创建当前帧和前一帧：错位帧组
        current = x[:, 1:, ...]     # [B,T-1,C,H,W]
        previous = x[:, :-1, ...]   # [B,T-1,C,H,W]
        
        # 重塑维度以便并行处理所有时间步
        current = current.reshape(B*(T-1), C, H, W)
        previous = previous.reshape(B*(T-1), C, H, W)
        
        # 连接特征并并行处理
        concat_feat = torch.cat([current, previous], dim=1)  # [B*(T-1),2C,H,W]
        temporal_feat = self.temporal_conv(concat_feat)          # [B*(T-1),C,H,W]
        
        # 重塑回原始维度
        temporal_feat = temporal_feat.reshape(B, T-1, C, H, W)
        
        # 处理第一帧
        first_feat = torch.zeros_like(x[:,:1,...]) if T == 1 else temporal_feat[:,:1,...]
        
        # 拼接所有特征
        return torch.cat([first_feat, temporal_feat], dim=1)    # [B,T,C,H,W]
        
    def forward(self, query, key=None, value=None):
        """
        参数:
            query: [B,T,C,H,W] - 查询张量
            key: [B,T,C,H,W] - 键张量(可选，默认None表示自注意力)
            value: [B,T,C,H,W] - 值张量(可选，默认None表示自注意力)
        返回:
            output: 与query相同shape
            attention: 注意力权重
        """
        # 自注意力模式
        if key is None:
            key = query
        if value is None:
            value = query
            
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            
        B, T, C, H, W = query.shape
        
        # 1. 并行提取时序特征
        q_feats = self._extract_temporal_features(query)
        k_feats = self._extract_temporal_features(key)
        v_feats = self._extract_temporal_features(value)
        
        # 2. 并行重塑为注意力输入格式
        q_feats = q_feats.view(B, T, C, -1).permute(0,3,1,2)  # [B,H*W,T,C]
        k_feats = k_feats.view(B, T, C, -1).permute(0,3,1,2)
        v_feats = v_feats.view(B, T, C, -1).permute(0,3,1,2)
        
        HW = q_feats.shape[1]
        q_feats = q_feats.reshape(-1, T, C)  # [B*HW,T,C]
        k_feats = k_feats.reshape(-1, T, C)
        v_feats = v_feats.reshape(-1, T, C)
        
        # 3. 注意力计算
        attn_feats, attn_weights = self.attn(q_feats, k_feats, v_feats)
        
        # 4. 恢复原始维度
        out = attn_feats.reshape(B, HW, T, C).permute(0,2,3,1).reshape(B, T, C, H, W)
        
        return out, attn_weights

class SpatioTemporalAttentionGate(nn.Module): # [B,T,C,H,W]
    """时空注意力门控""" ##并联注意力
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, batch_first=True):
        super().__init__() 
        self.batch_first = batch_first
        
        # 特征投影
        self.norm = nn.LayerNorm(embed_dim)
        
        # 注意力分支
        self.spatial_attn = SpatialMutiAttention(embed_dim, num_heads, dropout, batch_first)
        self.temporal_attn = TemporalDiffMutiAttention(embed_dim, num_heads, dropout)
        
        # 特征融合 使用1x1卷积实现通道降维
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(embed_dim * 2, embed_dim, 1),
            nn.Dropout(dropout)
        )
        
        # 注意力门控单元
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.Sigmoid()
        )
        
        self.temporal_gate = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.Sigmoid()
        )

        
    def forward(self, query,key,value):  ### 自注意力 ### 
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        B, T, C, H, W = query.shape
        
        # 1. 空间注意力
        spatial_out, spatial_attn = self.spatial_attn(query,key,value)
        spatial_out = spatial_out.reshape(B*T, C, H, W)
        
        # 2. 时序差分注意力
        temporal_out, temporal_attn = self.temporal_attn(query,key,value)
        temporal_out = temporal_out.reshape(B*T, C, H, W) # PyTorch的卷积层和批归一化层等操作都是针对[batch_size, channels, height, width]这种标准格式优化

        # 3. 自适应门控
        spatial_weight = self.spatial_gate(spatial_out)
        temporal_weight = self.temporal_gate(temporal_out)
        
        # 4. 加权融合
        attn_out = spatial_weight * spatial_out + temporal_weight * temporal_out
        
        # 5. MLP
        out = self.mlp(attn_out)
        
        # 恢复时序维度
        out = out.view(B, T, C, H, W)
        attn = spatial_attn + temporal_attn
        return out, attn
    
######################
#############################################
#######################################################################################
# class SpaceTimeAttentionSediment(nn.Module):
#     """
#     注意力降采样并残差连接
#     输入: [batch_size, time_step, channel, height, width]
#     输出: [batch_size, time_step, channel, height, width]
#     """
#     def __init__(self, embed_dim, num_heads=8, dropout=0.1, batch_first=True):
#         super().__init__()
#         self.batch_first = batch_first
#         # 空间和时序注意力
#         self.spatial_attn = CBAM( # 单头注意力
#             embed_dim=embed_dim,
#             reduction_ratio=4,
#             kernel_size=3,
#             batch_first=self.batch_first
#         )    
#         self.temporal_attn = TemporalDiffMutiAttention(
#             embed_dim=embed_dim,
#             num_heads=num_heads,  # 多头注意力
#             dropout=dropout,
#             batch_first=self.batch_first
#         )
        
#         # 沉淀token 
#         self.spatial_sediment_token = nn.Parameter(torch.zeros(1, embed_dim, 1, 1))
#         self.temporal_sediment_token = nn.Parameter(torch.zeros(1, embed_dim, 1, 1))
        
#         # 特征归一化
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
        
#         # 沉淀特征融合
#         self.sediment_fusion = nn.Sequential(
#             nn.Linear(embed_dim * 2, embed_dim),
#             nn.GELU(),
#             nn.Dropout(dropout)
#         )
        
#         # 初始化
#         nn.init.trunc_normal_(self.spatial_sediment_token, std=.02)
#         nn.init.trunc_normal_(self.temporal_sediment_token, std=.02)
        
#     def forward(self, x):
#         if not self.batch_first:
#             x = x.transpose(0, 1)
#         B, T, C, H, W = x.shape
#         identity = x.view(B*T, C, H, W)
        
#         # 1. 空间注意力分支
#         spatial_out, _ = self.spatial_attn(x)
#         spatial_out = spatial_out.reshape(B*T, C, H, W)
        
#         # 2. 空间沉淀 
#         spatial_weight = F.adaptive_avg_pool2d(spatial_out, 1)  # [B*T, C, 1, 1]
#         spatial_sediment = spatial_weight * self.spatial_sediment_token  # [B*T, C, 1, 1]
#         spatial_sediment = spatial_sediment.expand(-1, -1, H, W)  # [B*T, C, H, W]
        
#         # 3. 时序注意力分支
#         temporal_out, _ = self.temporal_attn(x,x,x)
#         temporal_out = temporal_out.reshape(B*T, C, H, W)
        
#         # 4. 时序沉淀 
#         temporal_weight = F.adaptive_avg_pool2d(temporal_out, 1)  # [B*T, C, 1, 1]
#         temporal_sediment = temporal_weight * self.temporal_sediment_token  # [B*T, C, 1, 1]
#         temporal_sediment = temporal_sediment.expand(-1, -1, H, W)  # [B*T, C, H, W]
        
#         # 5. 沉淀特征融合
#         sediment_feat = torch.cat([
#             spatial_sediment,
#             temporal_sediment
#         ], dim=1)  # [B*T, 2C, H, W]
        
#         sediment_feat = F.adaptive_avg_pool2d(sediment_feat, 1)  # [B*T, 2C, 1, 1]
#         sediment_feat = sediment_feat.squeeze(-1).squeeze(-1)  # [B*T, 2C]
#         sediment_feat = self.sediment_fusion(sediment_feat)  # [B*T, C]
#         sediment_feat = sediment_feat.view(B*T, C, 1, 1).expand(-1, -1, H, W)  # [B*T, C, H, W]
        
#         # 6. 注意力沉淀残差连接
#         out = identity + spatial_out + temporal_out + sediment_feat
        
#         # 恢复时序维度
#         out = out.view(B, T, C, H, W)
        
#         return out

#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################

def test_model():
    """测试代码"""
    print("开始测试模型...")
    
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 初始化模型并移至目标设备
    model = SpatioTemporalAttentionGate(
        embed_dim=64,
        num_heads=8,
        dropout=0.1
    ).to(device)
    
    # 测试用例1-4: 基本功能测试
    test_cases = [
        (torch.randn(2, 4, 64, 32, 32), "标准输入尺寸"),
        (torch.randn(2, 2, 64, 32, 32), "最小时间步长(T=2)"),
        (torch.randn(2, 4, 64, 16, 16), "不同空间尺寸"),
        (torch.randn(4, 4, 64, 32, 32), "较大批次")
    ]
    
    for i, (x, desc) in enumerate(test_cases, 1):
        print(f"\n测试{i}：{desc}")
        x = x.to(device)
        with torch.no_grad():
            out = model(x)   ### 融合后 单输出
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")
    
        
    # 测试用例5：梯度检查
    print("\n测试5：梯度检查")
    
    # 创建输入张量并确保它是叶子节点
    x = torch.randn(2, 4, 64, 32, 32).to(device)
    x.requires_grad_(True)  # 这样设置可以确保它是叶子节点
    
    # 清除现有梯度
    model.zero_grad()
    if x.grad is not None:
        x.grad.zero_()
    
    # 前向传播和反向传播
    out, _ = model(x)  # 修改为处理注意力权重返回值
    target = torch.zeros_like(out)
    loss = F.mse_loss(out, target)
    loss.backward()
    
    # 检查输入梯度
    grad_info = {}
    if x.grad is not None:  # 先检查梯度是否存在
        grad_info = {
            "存在": True,
            "平均值": x.grad.abs().mean().item(),
            "最大值": x.grad.abs().max().item(),
            "标准差": x.grad.std().item(),
            "损失值": loss.item()
        }
    else:
        grad_info = {
            "存在": False,
            "平均值": 0.0,
            "最大值": 0.0,
            "标准差": 0.0,
            "损失值": loss.item() if loss is not None else 0.0
        }
    
    print("\n输入梯度信息:")
    for k, v in grad_info.items():
        print(f"- {k}: {v:.6f}" if isinstance(v, float) else f"- {k}: {v}")
    
    # 检查模型参数梯度
    param_grads = []
    print("\n模型参数梯度:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_grads.append(grad_norm)
            print(f"- {name}: {grad_norm:.6f}")
    
    # 梯度统计
    if param_grads:
        param_grads = torch.tensor(param_grads)
        print("\n梯度统计:")
        print(f"- 参数总数: {len(list(model.parameters()))}")
        print(f"- 有梯度参数数: {len(param_grads)}")
        print(f"- 平均梯度范数: {param_grads.mean():.6f}")
        print(f"- 最大梯度范数: {param_grads.max():.6f}")
        print(f"- 最小梯度范数: {param_grads.min():.6f}")
    
    # 内存清理
    print("\n内存清理")
    del model, x, out, loss
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print("\n所有测试完成!")
    return True

if __name__ == "__main__":
    test_model()
    
# 3/5 self/cross