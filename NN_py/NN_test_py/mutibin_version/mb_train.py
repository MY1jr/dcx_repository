import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.append('G:/NN_py/NN_test_py')
from dcx_dataset import dcxDataset

from utils.data_postprocess import plot_losscurve, save_lossdata
from mb_models import ConvLSTM_MB, MultiBinLoss
from utils.misc import AverageMeter

# 图像预处理函数
def img_transform(img):
    # img = cv2.resize(img, (64, 64))
    img = cv2.resize(img, (100, 100))
    img = img / 255.0  # 图像归一化
    
    return img

# 数据集标签是[speed，angle（rad）], 而angle在网络中是使用（cos，sin表示的）
if __name__ == "__main__":
    # 训练超参数
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    time_step = 5
    # out_channels = 3  # 输出通道数：1个速度 + (cos,sin)
    pth_save_dir = 'G:/NN_py/NN_test_py/mutibin_version/pth'
    pth_load_path = 'G:/NN_py/NN_test_py/mutibin_version/best.pth'
    losscurve_save_path = 'G:/NN_py/NN_test_py/mutibin_version/loss_curves.png'
    lossdata_save_path = 'G:/NN_py/NN_test_py/mutibin_version/loss_data.csv'

    # # 数据集定义 ### 分开定义训练集和测试集
    # train_dataset = dcxDataset(
    #     root_dir="G:/NN_py/NN_test_py/dcx_train_mini", 
    #     label_txt="G:/NN_py/NN_test_py/dcx_train_mini.txt",
    #     time_step=time_step, 
    #     img_transform=img_transform
    # )
    # test_dataset = dcxDataset(
    #     root_dir="G:/NN_py/NN_test_py/dcx_test_mini", 
    #     label_txt="G:/NN_py/NN_test_py/dcx_test_mini.txt",
    #     time_step=time_step, 
    #     img_transform=img_transform
    # )
    
    # # 设置随机种子以确保可重复性
    # torch.manual_seed(seed=42)
    # torch.cuda.manual_seed_all(seed=42)

    # 数据集定义
    source_dataset = dcxDataset(
        root_dir="G:/NN_py/dcx",
        label_txt="G:/NN_py/dcx/dcx.txt", 
        time_step=time_step,
        img_transform=img_transform
    )
    
    # 划分训练集和测试集
    train_size = int(0.8 * len(source_dataset))  # 80%用于训练
    test_size = len(source_dataset) - train_size
    # 随机划分
    train_dataset, test_dataset = torch.utils.data.random_split(
        source_dataset, 
        [train_size, test_size]
    )

    # 数据加载器
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # 随机打乱训练数据
        num_workers=4,  # 多进程加载
        pin_memory=True  # GPU训练时更快
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    

    ##### 模型接口守则：#####
    # （1）必须定义有self.out_channels的类公有参数 
    # （2）输出[0]是速度，其余输出是角度（cos,sin），一般为2个 
    ##### end ##### 
    
    model = ConvLSTM_MB(
        in_channels=3,
        num_bins=4,
        time_step=time_step
    ).to(device)  
    print("\nModel:")
    print(model)
    
    # 定义损失函数
    angle_criterion = MultiBinLoss()  
    speed_criterion = nn.MSELoss() 
    
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 定义余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,  # 余弦周期的一半
        eta_min=1e-4  # 最小学习率
    )

    # 加载最佳模型权重，默认加载最新保存的模型
    try:    
        checkpoint = torch.load(pth_load_path) # 断点checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])  # 只加载模型参数
        print(f"加载权重文件完成, 其断点Epoch为{checkpoint['epoch']+1}")
        
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # 加载优化器状态
            print("优化器状态加载成功")
        except:
            print("优化器状态加载失败，将使用初始化的优化器状态")   
    except Exception as e:
        print(f"加载权重文件失败,初始化后重新训练") # 模型加载失败则不会加载优化器
    
    
    # 模型初始化，用pytorch默认
    
    ### 训练
    # 权重系数初始化
    angle_weight = 0.01
    speed_weight = 1
    # 每5轮保存一次中间权重
    SAVE_EPOCH_NUM = 5
    # best_Loss初始化
    best_loss = float('inf')
    # 记录训练和测试损失
    train_losses = []
    test_losses = []
    
    ## 训练循环
    for epoch in range(num_epochs):
        print(f"\n第 {epoch+1}/{num_epochs} 轮训练开始...")
        print(f"当前学习率: {scheduler.get_last_lr()[0]:.6f}")
        
        ## 训练模式
        model.train()
        # 训练损失累加器初始化
        total_train_loss = 0
        # 单次训练
        train_pbar = tqdm(train_dataloader, desc='training')
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            
            # 分离预测值
            speed_pred = output[:, 0]      # [batch]
            angle_pred = output[:, 1:model.out_channels]      # [batch, 2]
            
            ##### 标签转换 #####
            # 将角度转换为cos和sin (不用再归一化)
            angle_target_rad = target[:, 1]  # 角度在第二列
            angle_target = torch.stack([
                torch.cos(angle_target_rad), 
                torch.sin(angle_target_rad)    
            ], dim=1).to(device)            # [batch, 2]
            # 将速度归一化
            speed_target = target[:, 0] / 600     # 速度在第一列
            ##### end #####
            
            # 计算损失
            angle_loss = angle_criterion(angle_pred, angle_target)
            speed_loss = speed_criterion(speed_pred, speed_target)
            
            # 权重系数 平衡位置损失和速度损失
            loss = angle_weight * angle_loss + speed_weight * speed_loss
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪，设置更大的阈值以允许更多的梯度信息传递
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
            
            # 优化器更新
            optimizer.step()
            # 累加损失
            total_train_loss += loss.item()
            
            # 更新进度条信息
            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
        train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(train_loss)
        print(f'训练集平均损失: {train_loss:.6f}')
        
        ## 测试模式
        model.eval()
        # 初始化损失计量器
        test_loss_meter = AverageMeter()
        angle_loss_meter = AverageMeter()
        speed_loss_meter = AverageMeter()
        
        test_pbar = tqdm(test_dataloader, desc='testing')
        with torch.no_grad():
            for data, target in test_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                # 分离预测值
                speed_pred = output[:, 0]      # [batch]
                angle_pred = output[:, 1:model.out_channels]      # [batch, 2]
                
                # 将角度转换为cos和sin
                angle_target_rad = target[:, 1] # rad
                angle_target = torch.stack([ # cos,sin
                    torch.cos(angle_target_rad), # 注意：角度在第二列
                    torch.sin(angle_target_rad)  
                ], dim=1).to(device)         # [batch, 2]
                # 将速度归一化
                speed_target = target[:, 0] / 600  # 速度在第一列
                
                # 计算损失
                angle_loss = angle_criterion(angle_pred, angle_target)
                speed_loss = speed_criterion(speed_pred, speed_target)
                # 权重系数 平衡位置损失和速度损失
                loss = angle_weight * angle_loss + speed_weight * speed_loss
                
                # 更新计量器
                test_loss_meter.update(loss.item(), data.size(0))
                angle_loss_meter.update(angle_loss.item(), data.size(0))
                speed_loss_meter.update(speed_loss.item(), data.size(0))
                
                # 更新进度条信息
                test_pbar.set_postfix({
                    'loss': f'{test_loss_meter.avg:.6f}'
                })
                
        test_loss = test_loss_meter.avg
        test_losses.append(test_loss)
        
        print(f'测试集平均损失: {test_loss:.6f}')
        print(f'测试集角度损失: {angle_loss_meter.avg:.6f}')
        print(f'测试集速度损失: {speed_loss_meter.avg:.6f}')
        
        # 更新学习率
        scheduler.step()
        
        # 存最佳模型
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }
        
        # 每5轮保存一次中间权重
        if (epoch + 1) % SAVE_EPOCH_NUM == 0:
            # 保存当前epoch的模型权重
            current_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
            }
            # 以epoch命名保存中间权重
            torch.save(current_model_state, f"{pth_save_dir}/epoch_{epoch//SAVE_EPOCH_NUM*SAVE_EPOCH_NUM}_{(epoch//SAVE_EPOCH_NUM+1)*SAVE_EPOCH_NUM}.pth")
            print(f"保存第{epoch//SAVE_EPOCH_NUM*SAVE_EPOCH_NUM}轮到第{(epoch//SAVE_EPOCH_NUM+1)*SAVE_EPOCH_NUM}轮模型权重，测试损失: {test_loss:.6f}")
            # 每5轮同时更新一次最优模型
            torch.save(best_model_state, f"{pth_save_dir}/best.pth")
            print(f"保存最优模型, Epoch为{best_model_state['epoch']+1}，测试损失: {best_loss:.6f}")

    print("\n训练完成!")

    ### 后处理
    ## 调用绘图函数
    plot_losscurve(train_losses, test_losses, save_path=losscurve_save_path)
    ## 保存训练损失数据到csv
    save_lossdata(train_losses, test_losses, save_path=lossdata_save_path)

