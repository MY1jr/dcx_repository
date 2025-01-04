import torch
import numpy as np

# 适用于muti-bin角度预测
def mb_decode_angle(pred, num_bins):
    """
    解码预测值为角度
    Args:
        pred: [batch, num_bins*2] 预测值
    Returns:
        angles: [batch] 预测的角度值（弧度制）
    """
    pred_cos = pred[:, :num_bins]  # [batch, num_bins]
    pred_sin = pred[:, num_bins:]  # [batch, num_bins]
    
    # 找到置信度最高的bin
    _, bin_idx = torch.max(pred_cos, dim=1)  # [batch]
    
    # 获取对应bin的cos和sin值
    batch_size = pred.shape[0]
    cos_val = pred_cos[torch.arange(batch_size), bin_idx]  # [batch]
    sin_val = pred_sin[torch.arange(batch_size), bin_idx]  # [batch]
    
    # 归一化
    norm = torch.sqrt(cos_val**2 + sin_val**2)
    cos_val = cos_val / (norm + 1e-8)
    sin_val = sin_val / (norm + 1e-8)

    # 计算角度
    angles = torch.atan2(sin_val, cos_val)
    # 确保角度在[0, 2π]范围内
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    return angles # rad


def mb_robust_decode_angle(pred, num_bins, threshold=0.01):
    """
    更稳健的多bin角度解码函数，特别处理四个象限边界情况
    Args:
        pred: [batch, num_bins*2] 预测值,包含每个bin的cos和sin
        num_bins: bin的数量
        threshold: 判断是否在象限边界的阈值
    Returns:
        angles: [batch] 预测的角度值（弧度制）
    """
    pred_cos = pred[:, :num_bins]  # [batch, num_bins]
    pred_sin = pred[:, num_bins:]  # [batch, num_bins]
    
    # 找到置信度最高的bin
    _, bin_idx = torch.max(pred_cos, dim=1)  # [batch]
    
    # 获取对应bin的cos和sin值
    batch_size = pred.shape[0]
    cos_val = pred_cos[torch.arange(batch_size), bin_idx]  # [batch]
    sin_val = pred_sin[torch.arange(batch_size), bin_idx]  # [batch]
    
    # 归一化
    norm = torch.sqrt(cos_val**2 + sin_val**2)
    cos_val = cos_val / (norm + 1e-6)
    sin_val = sin_val / (norm + 1e-6)
    
    # 创建输出张量
    angles = torch.zeros_like(cos_val)
    
    # 处理每个象限边界的情况
    for i in range(cos_val.shape[0]):
        c, s = cos_val[i], sin_val[i]
        
        # 0°/360° 边界 (第一/四象限)
        if abs(s) < threshold and c > 0:
            angles[i] = 0 if s >= 0 else 2 * np.pi
            
        # 90° 边界 (第一/二象限)
        elif abs(c) < threshold and s > 0:
            angles[i] = np.pi / 2
            
        # 180° 边界 (第二/三象限)
        elif abs(s) < threshold and c < 0:
            angles[i] = np.pi
            
        # 270° 边界 (第三/四象限)
        elif abs(c) < threshold and s < 0:
            angles[i] = 3 * np.pi / 2
            
        # 非边界情况，使用常规atan2
        else:
            angle = torch.atan2(s, c)
            angles[i] = (angle + 2 * np.pi) % (2 * np.pi)
    
    return angles