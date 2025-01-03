import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append('G:/NN_py/NN_test_py/')
from utils.misc import AverageMeter
from dcx_dataset import dcxDataset
from mutibin_version.mb_models import ConvLSTM_MB
from mb_utils.mb_decode_angle import mb_decode_angle
import numpy as np
import cv2

# 图像预处理函数
def img_transform(img):
    img = cv2.resize(img, (100, 100))
    img = img / 255.0  # 图像归一化
    return img

def predict(model, dataloader, device):
    """
    使用训练好的模型进行预测
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 计算设备
    Returns:
        pred_speeds: 预测的速度值列表
        pred_angles: 预测的角度值列表（弧度制）
        true_speeds: 真实的速度值列表  
        true_angles: 真实的角度值列表（弧度制）
    """
    # 初始化存储列表
    pred_speeds = []
    pred_angles = []
    true_speeds = []
    true_angles = []
    
    # 设置为评估模式
    model.eval()
    
    # 初始化计量器
    speed_rel_error_meter = AverageMeter()
    angle_rel_error_meter = AverageMeter()
    speed_mae_meter = AverageMeter()
    angle_mae_meter = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='predicting')
        for data, target in pbar:
            # 数据移至目标设备
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            
            # 分离预测值
            speed_pred = output[:, 0]  # [batch]
            angle_pred = output[:, 1:model.out_channels]  # [batch, 2]
            
            # 解码角度预测值 ####################################################
            angle_pred_value = mb_decode_angle(angle_pred, model.num_bins)
            #####################################################################
            
            # 还原速度值（预测值已归一化，需乘以600）
            speed_pred_value = speed_pred * 600
            
            # 获取真实值（标签未归一化）
            true_speed = target[:, 0]  # 速度在第一列
            true_angle = target[:, 1]  # 角度在第二列
            
            # 计算相对误差
            speed_rel_error = torch.abs(speed_pred_value - true_speed) / (torch.abs(true_speed) + 1e-8)
            
            # 计算角度差值时考虑2pi周期性
            angle_diff = torch.abs(angle_pred_value - true_angle)
            angle_diff = torch.minimum(angle_diff, 2 * np.pi - angle_diff)  # 取最小角度差
            angle_rel_error = angle_diff / (torch.abs(true_angle) + 1e-8)
            
            # 计算MAE
            speed_mae = torch.abs(speed_pred_value - true_speed)
            angle_mae = angle_diff  # 已经计算过最小角度差
            
            # 更新计量器
            speed_rel_error_meter.update(speed_rel_error.mean().item(), data.size(0))
            angle_rel_error_meter.update(angle_rel_error.mean().item(), data.size(0))
            speed_mae_meter.update(speed_mae.mean().item(), data.size(0))
            angle_mae_meter.update(angle_mae.mean().item(), data.size(0))
            
            # 存储预测值和真实值
            pred_speeds.extend(speed_pred_value.cpu().numpy())
            pred_angles.extend(angle_pred_value.cpu().numpy())
            true_speeds.extend(true_speed.cpu().numpy())
            true_angles.extend(true_angle.cpu().numpy())
            
            # 更新进度条信息
            pbar.set_postfix({})
    
    print(f'\nSpeed MRE: {speed_rel_error_meter.avg:.2%}')
    print(f'Angle MRE: {angle_rel_error_meter.avg:.2%}')
    print()
    print(f'Speed MAE: {speed_mae_meter.avg:.2f}m/s')
    print(f'Angle MAE: {angle_mae_meter.avg:.4f}rad = {angle_mae_meter.avg/np.pi*180:.2f}°')
            
    return pred_speeds, pred_angles, true_speeds, true_angles

if __name__ == '__main__':
    # 训练超参数 ########################################################
    batch_size = 32
    time_step = 5
    pth_load_path = 'G:/NN_py/NN_test_py/mutibin_version/predict.pth' # predict.pth
    save_path = 'G:/NN_py/NN_test_py/mutibin_version/predict_results.csv'   
    root_dir = "G:/NN_py/dcx_mini"
    label_txt = "G:/NN_py/dcx_mini/dcx_mini.txt"
    #####################################################################

    # 设置随机种子以确保可重复性
    torch.manual_seed(seed=42)
    torch.cuda.manual_seed_all(seed=42)
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 加载测试数据集
    test_dataset = dcxDataset(
        root_dir=root_dir,
        label_txt=label_txt,
        time_step=time_step,
        img_transform=img_transform
    )

    # 数据加载器
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 定义模型 ############################################################
    model = ConvLSTM_MB(
        in_channels=3,
        num_bins=4,
        time_step=time_step
    ).to(device)
    #####################################################################


    # 加载模型权重
    try:
        checkpoint = torch.load(pth_load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Load model weights successfully")
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        exit()

    # 切换到评估模式
    model.eval()

    # 预测
    pred_speeds, pred_angles, true_speeds, true_angles = predict(
        model,
        test_dataloader,
        device
    )

    # 后处理
    # 将列表转换为numpy数组
    pred_speeds = np.array(pred_speeds)
    pred_angles = np.array(pred_angles)
    true_speeds = np.array(true_speeds)
    true_angles = np.array(true_angles)

    # 计算单个样本的相对误差和绝对误差
    speed_rel_error = np.abs((pred_speeds - true_speeds) / (np.abs(true_speeds) + 1e-8))  # 添加1e-8避免除零
    speed_abs_error = np.abs(pred_speeds - true_speeds)
    
    # 计算角度差值时考虑2pi周期性
    angle_diff = np.abs(pred_angles - true_angles)
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)  # 取最小角度差
    angle_rel_error = angle_diff / (np.abs(true_angles) + 1e-8)  # 添加小量避免除零
    angle_abs_error = angle_diff  # 使用考虑周期性的角度差
    
    # 保存预测结果和误差
    results = np.column_stack((
        pred_speeds, pred_angles, true_speeds, true_angles,
        speed_rel_error, angle_rel_error, speed_abs_error, angle_abs_error
    ))
    np.savetxt(
        save_path,
        results,
        delimiter=',',
        header='pred_speed,pred_angle,true_speed,true_angle,speed_rel_error,angle_rel_error,speed_abs_error,angle_abs_error',
        comments=''
    )
    print(f"预测结果已保存到 {save_path}")
    
