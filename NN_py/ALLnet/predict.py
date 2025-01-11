import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.decode_angle import  std_decode_angle
from utils.misc import AverageMeter
from dcx_dataset import mi_dcxDataset
from pathlib import Path
from models.allnet import ALLNet
import numpy as np
import cv2
from optflowaugment import get_optical_flow
from circleaugment import get_cluster_mask
# 图像预处理函数
def img_transform(img):
    img = cv2.resize(img, (64, 64))
    img = img / 255.0  # 图像归一化
    return img 

def predict_dataset(model, dataloader, device):
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
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='predicting')
        for batch_idx, (img_orig, img_flow, img_clus, target) in enumerate(pbar):
            # 数据移至目标设备
            img_orig = img_orig.to(device)
            img_flow = img_flow.to(device)
            img_clus = img_clus.to(device)
            target = target.to(device)
            
            # 前向传播
            output = model(img_orig, img_flow, img_clus)
            
            # 分离预测值
            speed_pred = output[:, 0]  # [batch]
            angle_pred = output[:, 1:model.out_channels]  # [batch, 2]
            
            # 解码角度预测值
            angle_pred_value = std_decode_angle(angle_pred)
            
            # 还原速度值（预测值已归一化，需乘以600）
            speed_pred_value = speed_pred * 600
            
            # 获取真实值（标签未归一化）
            true_speed = target[:, 0]  # 速度在第一列
            true_angle = target[:, 1]  # 角度在第二列
            
            # 存储预测值和真实值
            pred_speeds.extend(speed_pred_value.cpu().numpy())
            pred_angles.extend(angle_pred_value.cpu().numpy())
            true_speeds.extend(true_speed.cpu().numpy())
            true_angles.extend(true_angle.cpu().numpy())
            
            # 更新进度条信息
            pbar.set_postfix({})
    
    return pred_speeds, pred_angles, true_speeds, true_angles

def predict_image(model, image_path, device):
    """
    预测单张图片
    """
    # 读取并预处理图像
    img_orig = cv2.imread(str(image_path))
    display_img = img_orig.copy()
    
    # 预处理三种输入图像
    img_orig = img_transform(img_orig)
    img_flow = img_transform(img_orig)  # 这里需要替换为实际的光流图像
    img_clus = img_transform(img_orig)  # 这里需要替换为实际的聚类图像
    
    # 转换为[C, H, W]格式
    img_orig = torch.tensor(img_orig.transpose(2, 0, 1)).float()
    img_flow = torch.tensor(img_flow.transpose(2, 0, 1)).float()
    img_clus = torch.tensor(img_clus.transpose(2, 0, 1)).float()
    
    # 扩展batch维度
    img_orig = img_orig.unsqueeze(0)  # [1, C, H, W]
    img_flow = img_flow.unsqueeze(0)
    img_clus = img_clus.unsqueeze(0)
    
    # 移动到设备
    img_orig = img_orig.to(device)
    img_flow = img_flow.to(device)
    img_clus = img_clus.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(img_orig, img_flow, img_clus)
        speed_pred = output[:, 0] * 600
        angle_pred = std_decode_angle(output[:, 1:model.out_channels])
        
    # 获取预测结果
    speed_res = speed_pred.item()   
    angle_res = angle_pred.item()
    
    # 显示原始图像和预测结果
    display_img = cv2.resize(display_img, (480, 480))
    cv2.putText(display_img, f"Speed: {speed_res*0.13:.2f}cm/s", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(display_img, f"Angle: {angle_res/np.pi*180:.2f}deg", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Image', display_img)
    cv2.waitKey(0)
    
    return speed_res, angle_res

def predict_video(model, video_path, device, output_path=None):
    """
    预测视频，使用实时光流和聚类增强
    """
    cap = cv2.VideoCapture(str(video_path))
    model.eval()
    
    # 初始化光流计算的窗口队列
    window_size = 2
    frame_queue = []
    
    # 如果指定了输出路径，创建视频写入器
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 更新帧队列
            frame_queue.append(frame)
            if len(frame_queue) > window_size:
                frame_queue.pop(0)
            
            # 计算光流
            if len(frame_queue) >= 1:
                flow_img = get_optical_flow(frame_queue)
            else:
                flow_img = np.zeros_like(frame)
            
            # 计算聚类图像
            clus_img = get_cluster_mask(frame)
            
            # 预处理三种输入图像
            img_orig = img_transform(frame)
            img_flow = img_transform(flow_img)
            img_clus = img_transform(clus_img)
            
            # 转换格式并添加batch维度
            img_orig = torch.tensor(img_orig.transpose(2, 0, 1)).float().unsqueeze(0)
            img_flow = torch.tensor(img_flow.transpose(2, 0, 1)).float().unsqueeze(0)
            img_clus = torch.tensor(img_clus.transpose(2, 0, 1)).float().unsqueeze(0)
            
            # 移动到设备
            img_orig = img_orig.to(device)
            img_flow = img_flow.to(device)
            img_clus = img_clus.to(device)
            
            # 预测
            output = model(img_orig, img_flow, img_clus)
            speed_pred = output[:, 0] * 600
            angle_pred = std_decode_angle(output[:, 1:model.out_channels])
            
            # 显示预测结果
            display_frame = cv2.resize(frame, (240, 240))
            cv2.putText(display_frame, f"Speed: {speed_pred.item()*0.13:.2f}cm/s", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Angle: {angle_pred.item()/np.pi*180:.2f}deg", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 显示处理后的图像
            cv2.imshow('Original', display_frame)
            cv2.imshow('Optical Flow', cv2.resize(flow_img, (240, 240)))
            cv2.imshow('Clustering', cv2.resize(clus_img, (240, 240)))
            
            if writer:
                writer.write(display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    


if __name__ == '__main__':
    ##################################################mode#########################################
    
    mode = 'dataset' # dataset, image, video
    
    ########################################################

    # 获取当前文件所在目录
    current_dir = Path(__file__).parent # NN_test_py
    project_root = current_dir.parent # NN_py
    
    pth_load_path = current_dir / 'predict.pth'
    ###################################################################

    # 设置随机种子以确保可重复性
    torch.manual_seed(seed=42)
    torch.cuda.manual_seed_all(seed=42)
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 定义模型 ###########################################################
    model = ALLNet(
        in_channels=3,
        max_dims=48,
        num_blocks=6
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
    

    #### mode ####
    if mode == 'dataset':
        ####################################################
        batch_size = 32
        pred_save_path = current_dir / 'predict_results.csv'
        index_save_path = current_dir / 'predict_indexs.txt'
        label_txt = project_root / "dcx_mini" / "dcx_mini_merge.txt"
        ####################################################
        # 定义传感器量程常量
        MAX_SPEED_FULLSCALE = 600  # 速度传感器满量程
        MAX_ANGLE_FULLSCALE = 2 * np.pi  # 角度传感器满量程
        
        # 加载测试数据集
        test_dataset = mi_dcxDataset(
            label_txt=label_txt,
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

        # 预测
        pred_speeds, pred_angles, true_speeds, true_angles = predict_dataset(
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

        # 计算速度误差
        speed_abs_error = np.abs(pred_speeds - true_speeds)
        speed_rel_error = speed_abs_error / MAX_SPEED_FULLSCALE

        # 计算角度差值时考虑2pi周期性
        angle_diff = np.abs(pred_angles - true_angles)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)  # 取最小角度差
        angle_abs_error = angle_diff  # 使用考虑周期性的角度差
        angle_rel_error = angle_diff / MAX_ANGLE_FULLSCALE  # 角度相对误差相对满量程
        
        # 计算误差平均统计
        speed_mre_avg = np.mean(speed_rel_error)
        angle_mre_avg = np.mean(angle_rel_error) 
        speed_mae_avg = np.mean(speed_abs_error)
        angle_mae_avg = np.mean(angle_abs_error)
        
        # 保存预测结果和误差至csv文件
        results = np.column_stack((
            pred_speeds, pred_angles, true_speeds, true_angles,
            speed_rel_error, angle_rel_error, speed_abs_error, angle_abs_error
        ))
        np.savetxt(
            pred_save_path,
            results,
            delimiter=',',
            header='pred_speed,pred_angle,true_speed,true_angle,speed_rel_error,angle_rel_error,speed_abs_error,angle_abs_error',
            comments=''
        )
        print(f"Dataset predictions have been saved to {pred_save_path}")
        
        # 保存平均误差统计到txt文件
        with open(index_save_path, 'w') as f:
            f.write("平均误差统计:\n")
            f.write(f"速度平均绝对误差(MAE): {speed_mae_avg:.2f}rpm = {speed_mae_avg*0.13:.2f}cm/s\n")
            f.write(f"角度平均绝对误差(MAE): {angle_mae_avg:.4f}rad = {angle_mae_avg/np.pi*180:.2f}°\n")
            f.write(f"速度平均相对误差(MRE): {speed_mre_avg*100:.2f}%\n")
            f.write(f"角度平均相对误差(MRE): {angle_mre_avg*100:.2f}%\n")
        
        print(f"Average errors have been saved to {index_save_path}")
        
        # 打印平均误差统计
        print("\nAverage Error Statistics:")
        print(f"Speed MAE: {speed_mae_avg:.2f}rpm = {speed_mae_avg*0.13:.2f}cm/s")
        print(f"Angle MAE: {angle_mae_avg:.4f}rad = {angle_mae_avg/np.pi*180:.2f}°")
        print(f"Speed MRE: {speed_mre_avg*100:.2f}%")
        print(f"Angle MRE: {angle_mre_avg*100:.2f}%")


    elif mode == 'image':
        image_path = project_root / 'dcx_mini/600/320/15.png'
        speed, angle = predict_image(model, image_path, device)
        print(f"Image prediction: speed={speed:.4f}rpm={speed*0.13:.2f}cm/s, angle={angle:.4f}rad={angle/np.pi*180:.2f}°")
    
    
    elif mode == 'video': # 按q退出
        video_path = current_dir / 'video/dcx_video.mp4'
        video_output_path = current_dir / 'video/dcx_video_pred.mp4'
        predict_video(model, video_path, device, output_path=video_output_path)
        print(f"Video prediction completed and saved to {video_output_path}")

