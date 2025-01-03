import torch
import numpy as np

# 仅适用于标准角度预测
def std_decode_angle(pred):
    """
    解码预测值为角度
    Args:
        pred: [batch, 2] 预测值,包含cos和sin
        Returns:
        angles: [batch] 预测的角度值（弧度制）
    """
    # 获取cos和sin值
    cos_val = pred[:, 0]  # [batch]
    sin_val = pred[:, 1]  # [batch]
    # 归一化
    norm = torch.sqrt(cos_val**2 + sin_val**2)
    cos_val = cos_val / (norm + 1e-8)
    sin_val = sin_val / (norm + 1e-8)
    # 计算角度
    angles = torch.atan2(sin_val, cos_val) # torch.atan2() 考虑象限，返回值范围为[-π, π]
    # 确保角度在[0, 2π]范围内
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    return angles


def robust_decode_angle(pred, threshold=0.01):
    """
    更稳健的角度解码函数，特别处理四个象限边界情况
    Args:
        pred: [batch, 2] 预测值,包含cos和sin
        threshold: 判断是否在象限边界的阈值
    Returns:
        angles: [batch] 预测的角度值（弧度制）
    """
    cos_val = pred[:, 0]  # [batch]
    sin_val = pred[:, 1]  # [batch]
    
    # 归一化
    norm = torch.sqrt(cos_val**2 + sin_val**2)
    cos_val = cos_val / (norm + 1e-8)
    sin_val = sin_val / (norm + 1e-8)
    
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

def psc3_decode_angle(pred):
    """
    使用三步相移法解码预测值为角度，使用振幅正则化
    Args:
        pred: [batch, 3] 预测值,包含三步相移的强度值
    Returns:
        angles: [batch] 预测的角度值（弧度制）
    """
    # 振幅正则化
    I_max = torch.max(pred, dim=1)[0]  # [batch]
    I_min = torch.min(pred, dim=1)[0]  # [batch]
    I_amp = (I_max - I_min) / 2
    pred_norm = pred / (I_amp.unsqueeze(1) + 1e-8)
    
    # 获取归一化后的强度值
    m1 = pred_norm[:, 0]  # [batch]
    m2 = pred_norm[:, 1]  # [batch]
    m3 = pred_norm[:, 2]  # [batch]
    
    # 计算角度
    numerator = np.sqrt(3) * (m3 - m2)
    denominator = 2 * m1 - m2 - m3
    angles = torch.atan2(numerator, denominator)
    
    # 确保角度在[0, 2π]范围内
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    return angles


# def test_angle_decode():
#     """测试角度解码函数"""
#     # 创建测试数据
#     test_cases = [
#         # 标准情况（无噪声）
#         torch.tensor([[1.0, 0.0],    # 0度
#                      [0.0, 1.0],     # 90度
#                      [-1.0, 0.0],    # 180度
#                      [0.0, -1.0]]),  # 270度
        
#         # 边界情况（带小噪声）
#         torch.tensor([[0.999, 0.001],    # 接近0度
#                      [0.001, 0.999],     # 接近90度
#                      [-0.999, -0.001],   # 接近180度
#                      [-0.001, -0.999]]), # 接近270度
        
#         # 模拟网络输出（未归一化 + 噪声）
#         torch.tensor([[1.2, 0.05],      # 接近0度
#                      [0.08, 1.5],       # 接近90度
#                      [-0.95, -0.02],    # 接近180度
#                      [-0.03, -1.1]]),   # 接近270度
        
#         # 边界情况（更细致的测试）
#         torch.tensor([
#             # 0度附近（左右）
#             [0.9999, -0.0001],  # 略小于0度
#             [0.9999, 0.0001],   # 略大于0度
            
#             # 90度附近（左右）
#             [0.0001, 0.9999],   # 略小于90度
#             [-0.0001, 0.9999],  # 略大于90度
            
#             # 180度附近（左右）
#             [-0.9999, 0.0001],  # 略小于180度
#             [-0.9999, -0.0001], # 略大于180度
            
#             # 270度附近（左右）
#             [-0.0001, -0.9999], # 略小于270度
#             [0.0001, -0.9999],  # 略大于270度
#         ]),
        
#         # 带噪声的边界情况
#         torch.tensor([
#             # 接近0/360度的情况
#             [1.2, -0.02],    # 接近但略小于0度
#             [1.1, 0.02],     # 接近但略大于0度
            
#             # 接近180度的情况
#             [-1.1, 0.03],    # 接近但略小于180度
#             [-1.2, -0.03],   # 接近但略大于180度
            
#             # 非标准化的边界值
#             [0.5, -0.001],   # 接近0度
#             [-0.6, 0.001],   # 接近180度
#         ])
#     ]
    
#     print("测试标准解码和稳健解码的结果对比：")
#     for i, test_data in enumerate(test_cases):
#         print(f"\n测试用例 {i+1}:")
#         print("输入数据:")
#         print(test_data.numpy())
        
#         # 使用两种方法解码
#         std_angles = std_decode_angle(test_data)
#         # robust_angles = robust_decode_angle(test_data)
        
#         # 转换为角度制并打印结果
#         print("\n标准解码结果 (度):")
#         print(std_angles.numpy() / np.pi * 180)
#         # print("稳健解码结果 (度):")
#         # print(robust_angles.numpy() / np.pi * 180)

# if __name__ == '__main__':
#     test_angle_decode()

def test_angle_decode():
    """测试角度解码函数"""
    # 创建测试数据
    test_cases = [
        # 标准情况（理想强度值）
        torch.tensor([
            [1.0, -0.5, -0.5],      # 0度
            [0.0, -0.866, 0.866],   # 90度
            [-1.0, -0.5, -0.5],     # 180度
            [0.0, 0.866, -0.866]    # 270度
        ]),
        
        # 边界情况（带小噪声）
        torch.tensor([
            [0.999, -0.501, -0.499],    # 接近0度
            [0.001, -0.865, 0.867],     # 接近90度
            [-0.999, -0.501, -0.501],   # 接近180度
            [0.001, 0.865, -0.867]      # 接近270度
        ]),
        
        # 模拟网络输出（未归一化 + 噪声）
        torch.tensor([
            [1.2, -0.6, -0.55],     # 接近0度
            [0.05, -1.0, 1.1],      # 接近90度
            [-1.1, -0.45, -0.48],   # 接近180度
            [0.02, 0.95, -0.92]     # 接近270度
        ]),
        
        # 边界情况（更细致的测试）
        torch.tensor([
            # 0度附近
            [1.0, -0.501, -0.499],   # 略偏离0度
            [1.0, -0.499, -0.501],   # 略偏离0度
            
            # 90度附近
            [0.01, -0.867, 0.865],   # 略偏离90度
            [-0.01, -0.865, 0.867],  # 略偏离90度
            
            # 180度附近
            [-1.0, -0.499, -0.501],  # 略偏离180度
            [-1.0, -0.501, -0.499],  # 略偏离180度
            
            # 270度附近
            [0.01, 0.867, -0.865],   # 略偏离270度
            [-0.01, 0.865, -0.867]   # 略偏离270度
        ])
    ]
    
    print("测试三步相移解码结果：")
    for i, test_data in enumerate(test_cases):
        print(f"\n测试用例 {i+1}:")
        print("输入数据（三步相移强度值）:")
        print(test_data.numpy())
        
        # 解码
        angles = psc3_decode_angle(test_data)
        
        # 转换为角度制并打印结果
        print("\n解码结果 (度):")
        print(angles.numpy() / np.pi * 180)

if __name__ == '__main__':
    test_angle_decode()