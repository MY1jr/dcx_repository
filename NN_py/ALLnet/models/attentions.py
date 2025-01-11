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
        
    def forward(self, query, key=None, value=None, attn_mask=None):
        """
        参数:
            query: [batch_size, seq_len_q, embed_dim]
            key: [batch_size, seq_len_k, embed_dim]
            value: [batch_size, seq_len_v, embed_dim]
            attn_mask: [seq_len_q, seq_len_k] 或 [batch_size, seq_len_q, seq_len_k]
        """
        # 自注意力模式
        if key is None:
            key = query
        if value is None:
            value = query
        
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
#######################################################################################
# CBAM注意力模块（通道注意力与空间注意力） backbone中
class ChannelAttention(nn.Module):
    def __init__(self, embed_dim, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 全连接层mlp
        self.shared_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // reduction_ratio),
            nn.ReLU(),
            nn.Linear(embed_dim // reduction_ratio, embed_dim)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # [B,C,H,W] 
        B, C, H, W = x.size()
        
        # 计算池化特征
        avg_pool = self.avg_pool(x).view(B, C)
        max_pool = self.max_pool(x).view(B, C)
        
        # 共享MLP
        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)
        
        # 注意力权重
        attention = self.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        
        # 应用注意力
        x = x * attention

        return x, attention

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # [B,C,H,W]
        B, C, H, W = x.size()
        
        # 计算空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        
        # 应用注意力
        x = x * attention
    
        return x, attention

class CBAM(nn.Module):
    def __init__(self, embed_dim, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(embed_dim, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.embed_dim = embed_dim

    def forward(self, x):  # [B,C,H,W]
        # 通道注意力
        x, channel_attention = self.channel_attention(x)
        
        # 空间注意力 
        x, spatial_attention = self.spatial_attention(x)
        
        # attn_weights = channel_attention * spatial_attention
        return x

class MaskCSAttention(nn.Module):
    def __init__(self, embed_dim, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(embed_dim, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.embed_dim = embed_dim

    def forward(self, x_orig, x_argu):  # [B,C,H,W]
        # 通道注意力
        _, channel_attention = self.channel_attention(x_argu) # argu的channel注意力
        # 空间注意力 
        _, spatial_attention = self.spatial_attention(x_argu) # argu的spatial注意力
        # 融合注意力
        x = x_orig * channel_attention * spatial_attention
        
        return x
#######################################################################################
#######################################################################################

class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class FrameDiffAttention(nn.Module):
    """
    帧间差分注意力模块
    - 使用滑动窗口计算相邻帧差分
    - 差分特征经过降采样后与当前帧特征融合
    """
    def __init__(self, embed_dim, window_size=3, dropout=0.1, batch_first=True):
        super().__init__()
        assert window_size % 2 == 1, "Window size must be odd!"
        self.batch_first = batch_first
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.half_w = window_size // 2
        
        # 特征变换层
        self.conv1 = nn.Conv2d(embed_dim, embed_dim, 1)  # 当前帧的特征变换
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, 1)  # 差分特征的变换
        
        # 残差块
        self.res = ResBlock(embed_dim)
        
        # 上采样和下采样
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        
        # dropout
        self.dropout = nn.Dropout(dropout)
        
    def _compute_temporal_diffs(self, x, t):
        """
        优化的时序差分特征计算
        x: [B,T,C,H,W]
        t: 当前帧索引
        """
        B, T, C, H, W = x.shape
        
        # 1. 计算窗口范围
        start_idx = max(0, t - self.half_w)
        end_idx = min(T, t + self.half_w + 1)
        
        # 2. 提取窗口帧(排除当前帧)
        if t > start_idx and t < end_idx - 1:
            # 当前帧在窗口中间
            window_frames = torch.cat([
                x[:, start_idx:t],
                x[:, t+1:end_idx]
            ], dim=1)  # [B,window_size-1,C,H,W]
        elif t <= start_idx:
            # 当前帧在窗口开始
            window_frames = x[:, start_idx+1:end_idx]
        else:
            # 当前帧在窗口结束
            window_frames = x[:, start_idx:end_idx-1]
            
        if window_frames.size(1) < 2:
            # 如果窗口帧数不足以计算差分,返回零张量
            return torch.zeros_like(x[:,:1,:,::2,::2])
            
        # 3. 计算相邻帧差分
        diffs = window_frames[:, 1:] - window_frames[:, :-1]  # [B,window_size-2,C,H,W]
        
        # 4. 降采样
        diffs = self.downsample(diffs.flatten(0,1))  # [B*(window_size-2),C,H/2,W/2]
        diffs = diffs.view(B, -1, C, H//2, W//2)    # [B,window_size-2,C,H/2,W/2]
        
        return diffs
    
    def forward(self, x):
        """
        输入: x [B,T,C,H,W] - 视频特征序列
        输出: out [B,T,C,H,W] - 增强后的特征
        """
        B, T, C, H, W = x.shape
        enhanced_frames = []
        
        for t in range(T):
            # 1. 获取当前帧
            current = x[:, t]  # [B,C,H,W]
            current_feat = self.dropout(self.conv1(current))
            
            # 2. 计算时序差分特征
            diff_features = self._compute_temporal_diffs(x, t)  # [B,window_size-2,C,H/2,W/2]
            
            # 3. 平均差分特征
            diff_feat = torch.mean(diff_features, dim=1)  # [B,C,H/2,W/2]
            
            # 4. 差分特征变换
            diff_feat = self.dropout(self.conv2(diff_feat))  # [B,C,H/2,W/2]
            
            # 5. 上采样差分特征
            diff_feat = self.upsample(diff_feat)  # [B,C,H,W]
            
            # 6. 特征融合
            fused = current_feat + diff_feat
            
            # 7. 残差处理
            out = self.res(fused)
            out = self.dropout(out + current)
            
            enhanced_frames.append(out)
            
        # 拼接所有增强帧
        output = torch.stack(enhanced_frames, dim=1)
        
        return output

class AdaptiveDiffSimAttention(nn.Module):
    """
    DIM_QV = Linear((α+γ)*V + (β-γ)*CM_QV)
    (α，β，γ)∈[0,1]
    (α+γ)∈[0,2]: V的系数
    (β-γ)∈[-1,1]: CM_QV的系数
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.scale = (embed_dim // num_heads) ** -0.5
        
        self.layer_norm = nn.LayerNorm(embed_dim) # 层归一化
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 初始化两个参数
        self.alpha_gamma = nn.Parameter(torch.ones(embed_dim))  # α+γ 初始化为1
        self.beta_gamma = nn.Parameter(torch.ones(embed_dim))  # β-γ 初始化为1
        self.linear = nn.Linear(embed_dim, embed_dim)     # 最终的线性变换
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key=None, value=None):
        
        query = self.layer_norm(query)
        # 自注意力模式
        if key is None:
            key = query
        if value is None:
            value = query
        # 如果batch_first=False，转换输入维度
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        B, T, E = query.shape
        
        # 1. LayerNorm和QKV投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 2. 多头注意力
        q = q.reshape(B, T, self.num_heads, E // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, E // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, E // self.num_heads).permute(0, 2, 1, 3)
        
        # 3. 计算CM_QV
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        cm_qv = (attn @ v).transpose(1, 2).reshape(B, T, E)
        
        # 4. (α+γ)*V + (β-γ)*CM_QV
        v = v.transpose(1, 2).reshape(B, T, E)  # 原始特征V
        alpha_gamma = 2 * torch.sigmoid(self.alpha_gamma)  # 将(α+γ)限制在[0,2]
        beta_gamma = torch.tanh(self.beta_gamma)  # 将(β-γ)限制在[-1,1]
        out = alpha_gamma * v + beta_gamma * cm_qv  # 核心公式
        out = self.linear(out)  # 最后的线性变换
        attn_weights = {
            'attention': attn,
            'cm_qv': cm_qv,
            'alpha_gamma': alpha_gamma,
            'beta_gamma': beta_gamma
        }
        return out, attn_weights
    
class BADSAttention(nn.Module):
    """Batch Attention 并行聚合 自注意力模块"""
    def __init__(self, embed_dim, num_heads=8, num_stages=4, dropout=0.1, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.attentions = nn.ModuleList([
            AdaptiveDiffSimAttention(
                embed_dim, 
                num_heads=num_heads, 
                dropout=dropout, 
                batch_first=batch_first
            )
            for _ in range(num_stages)
        ])
        
        # 改用一维池化
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 平均池化
        self.max_pool = nn.AdaptiveMaxPool1d(1)  # 最大池化
        
        # 权重生成网络 - 修改输入维度
        self.weight_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),  # 每个stage输出一个权重
            nn.Softmax(dim=1)  # 在stage维度上做softmax
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.num_stages = num_stages
        
    def forward(self, query, key=None, value=None):
        """
        输入 x: [B,T,E]
        B: batch_size
        T: sequence_length
        E: embed_dim
        N: num_stages
        group_size: 每组数据的大小
        """
        # 如果batch_first=False，转换输入维度
        if not self.batch_first:
            query = query.transpose(0, 1)
        
        ### 只能聚合自注意力机制 ### 
        x = query
        B, T, E = x.shape
        
        # 1. 检查batch_size是否可以被num_stages整除
        if B % self.num_stages != 0:
            return x, None # 如果不能整除,直接返回原始输入和None权重
            
        group_size = B // self.num_stages
        
        # 2. 使用torch.chunk分组并并行处理
        x_chunks = torch.chunk(x, self.num_stages, dim=0)  # 将输入分成num_stages份
        stage_outputs = [attn(chunk)[0] for attn, chunk in zip(self.attentions, x_chunks)]
            
        # 3. 拼接所有组的输出
        stacked_outputs = torch.cat(stage_outputs, dim=0)  # [B,T,E]
        
        # 4. 特征池化
        pooling_feat = stacked_outputs.transpose(1, 2)  # [B,E,T]
        
        # 应用多种池化
        avg_feat = self.avg_pool(pooling_feat).squeeze(-1)  # [B,E]
        max_feat = self.max_pool(pooling_feat).squeeze(-1)  # [B,E]
        
        # 重塑为组
        avg_feat = avg_feat.view(self.num_stages, group_size, E)  # [N,group_size,E]
        max_feat = max_feat.view(self.num_stages, group_size, E)  # [N,group_size,E]
        
        # 5. 合并池化特征
        pooled_feat = torch.cat([avg_feat, max_feat], dim=-1)  # [N,group_size,2E]
        
        # 6. 生成融合权重
        stage_weights = []
        for i in range(self.num_stages):
            weight = self.weight_net(pooled_feat[i])  # [group_size,1]
            stage_weights.append(weight)
        stage_weights = torch.cat(stage_weights, dim=0)  # [B,1]
        stage_weights = F.softmax(stage_weights, dim=0)  # 确保权重和为1
        
        # 7. 加权融合
        x_out = stacked_outputs * stage_weights.unsqueeze(-1)  # [B,T,E]
        
        # 8. Dropout和LayerNorm
        x_out = self.dropout(x_out)
        x_out = self.norm(x_out)
        
        return x_out, stage_weights
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
### 测试
def test_model():
    """测试代码"""
    print("开始测试模型...")
    
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 初始化模型并移至目标设备
    model = FrameDiffAttention(
        embed_dim=64,
        num_heads=8,
        dropout=0.1,
        batch_first=True
    ).to(device)
    
    # 测试用例1-4: 基本功能测试
    test_cases = [
        (torch.randn(2, 4, 64, 64, 64), "标准输入尺寸"),
        (torch.randn(2, 2, 64, 64, 64), "最小时间步长(T=2)"),
        (torch.randn(2, 4, 64, 64, 64), "不同空间尺寸"),
        (torch.randn(4, 4, 64, 64, 64), "较大批次")
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
    x = torch.randn(2, 4, 64, 64, 64).to(device)
    x.requires_grad_(True)  # 这样设置可以确保它是叶子节点
    
    # 清除现有梯度
    model.zero_grad()
    if x.grad is not None:
        x.grad.zero_()
    
    # 前向传播和反向传播
    out = model(x)  # 修改为处理注意力权重返回值
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

def test_trans_model():
    """测试代码"""
    print("开始测试模型...")
    
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 初始化模型并移至目标设备
    model = BADSAttention(
        embed_dim=64,
        num_heads=8,
        num_stages=4,
        dropout=0.1,
        batch_first=True
    ).to(device)
    
    # 测试用例1-4: 基本功能测试
    test_cases = [
        (torch.randn(5, 4, 64), "标准输入尺寸"),
        (torch.randn(8, 2, 64), "最小时间步长(T=2)"),
        (torch.randn(8, 4, 64), "不同空间尺寸"),
        (torch.randn(8, 4, 64), "较大批次")
    ]
    
    for i, (x, desc) in enumerate(test_cases, 1):
        print(f"\n测试{i}：{desc}")
        x = x.to(device)
        with torch.no_grad():
            out, _ = model(x)   ### 融合后 单输出
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")
    
        
    # 测试用例5：梯度检查
    print("\n测试5：梯度检查")
    
    # 创建输入张量并确保它是叶子节点
    x = torch.randn(8, 4, 64).to(device)
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
    # test_model()
    test_trans_model()
    
# 3/5 self/cross