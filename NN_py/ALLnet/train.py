import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR, ConstantLR
import cv2
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as transforms
from utils.data_postprocess import plot_losscurve, save_errordata, save_detail_lossdata
from models.allnet import ALLNet
from utils.misc import AverageMeter
from utils.decode_angle import std_decode_angle
from circleaugment import CircleAugmentation
from dcx_dataset import mi_dcxDataset
from pathlib import Path
###############################################################################
# def seq_augmentation(img, p_aug):
#     if 0<= p_aug <= 0.5:
#         transform = CircleAugmentation()
#         return transform(img)
#     else:
#         return img  # 直接返回原图,不转换
###############################################################################
def img_transform(img):
    img = cv2.resize(img, (64, 64))
    img = img / 255.0  # 图像归一化
    return img
###############################################################################
# 数据集标签是[speed，angle（rad）], 而angle在网络中是使用（cos，sin表示的）
if __name__ == "__main__":
    # 训练超参数
    batch_size = 16
    num_epochs = 50
    angle_learning_rate = 0.00001 # 0.0001
    speed_learning_rate = 0.00001 # 0.0001
    # 断点重训
    resume_training = False # True
    keep_optimizer = False # 保持优化器 # 仅断点重训时生效
    keep_scheduler = False # 保持调度器 # 仅断点重训时生效
    # 获取当前文件所在目录和项目根目录
    current_dir = Path(__file__).parent # NN_test_py
    project_root = current_dir.parent # NN_py
    # 路径
    pth_save_dir = current_dir / 'pth'
    pth_load_path = current_dir / 'best.pth'
    losscurve_save_path = current_dir  # / 'anglelosscurve' and 'speedlosscurve'
    lossdata_save_path = current_dir / 'loss_data.csv'
    errordata_save_path = current_dir / 'error_data.csv'

    # 确保 pth 目录存在
    pth_save_dir.mkdir(exist_ok=True)
###############################################################################
    # # 设置随机种子以确保可重复性
    # torch.manual_seed(seed=42)
    # torch.cuda.manual_seed_all(seed=42)

    # 数据集定义
    source_dataset = mi_dcxDataset(
        label_txt=str(project_root / 'dcx_mini' / 'dcx_mini_merge.txt'), 
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
    
    # 训练集数据增强
    # train_dataset.dataset._set_seq_augmentation(seq_augmentation)
    
    # 数据加载器
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # 随机打乱训练数据 
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

###############################################################################
    ##### 模型接口守则：#####
    # （1）必须定义有self.out_channels的类公有参数 
    # （2）输出[0]是速度，其余输出是角度（cos,sin），一般为2个 
    ##### end ##### 
    model = ALLNet(
        in_channels=3,
        max_dims=48,
        num_blocks=6
    ).to(device)
    # print("\nModel:") 
    # print(model) # 打印模型结构
    
    # 计算模型总参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
###############################################################################

###############################################################################
    # # 角度损失函数
    # def angle_loss_fn(pred, target):
    #     # MSE损失
    #     mse_loss = nn.MSELoss()(pred, target)
    #     # 正则化损失
    #     cos_pred, sin_pred = pred[:, 0], pred[:, 1]
    #     norm_loss = F.l1_loss(cos_pred.pow(2) + sin_pred.pow(2), torch.ones_like(cos_pred))
    #     # 总损失 = MSE + 0.1*正则化
    #     return mse_loss + 0.1 * norm_loss
    # angle_criterion = angle_loss_fn
    angle_criterion = nn.MSELoss()
    speed_criterion = nn.MSELoss() 
###############################################################################
    # 定义两个优化器
    angle_optimizer = torch.optim.Adam(model.parameters(), lr=angle_learning_rate)
    speed_optimizer = torch.optim.Adam(model.parameters(), lr=speed_learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # 学习率调度器
    ## 超参数 ##
    # num_warmup_epochs = int(num_epochs * 0.05)  # warmup轮数5% = 5轮
    # warmup_factor = 0.1    # warmup起始学习率因子
    # cosine_T_max = (num_epochs - num_warmup_epochs) // 2  # 余弦退火半周期
    # min_lr = learning_rate * 0.01  # 最小学习率为初始学习率的1%
    # ## 调度器
    # scheduler = ChainedScheduler([
    #     # warmup：从 warmup_factor*lr 线性增加到 lr
    #     LinearLR(
    #         optimizer, 
    #         start_factor=warmup_factor,
    #         total_iters=num_warmup_epochs
    #     ),
    #     # 余弦退火：从 lr 降到 min_lr
    #     CosineAnnealingLR(
    #         optimizer,
    #         T_max=cosine_T_max,  # 余弦退火的周期
    #         eta_min=min_lr  # 最小学习率
    #     )
    # ])
   
    # 恒定学习率调度器
    angle_scheduler = ConstantLR(angle_optimizer, factor=1.0, total_iters=0)
    speed_scheduler = ConstantLR(speed_optimizer, factor=1.0, total_iters=0)

    # 加载最佳模型权重，默认加载最新保存的模型
    start_epoch = 0  # 起始epoch变量初始化
    try:    
        checkpoint = torch.load(pth_load_path, weights_only=True, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True) # 严格加载
        if resume_training:
            start_epoch = checkpoint['epoch'] + 1  # 设置起始epoch（未训练的epoch）# 从0计数
            print(f"加载权重文件完成, 将从第{start_epoch + 1}轮继续训练") # 从1计数
            try:
                if keep_optimizer:
                    # 恢复优化器状态
                    angle_optimizer.load_state_dict(checkpoint['angle_optimizer_state_dict'])
                    speed_optimizer.load_state_dict(checkpoint['speed_optimizer_state_dict'])
                    print("优化器加载成功")
                else:
                    print("未加载优化器")
                if keep_scheduler:
                    # 恢复学习率调度器状态
                    angle_scheduler.load_state_dict(checkpoint['angle_scheduler_state_dict']) # 前提有相同调度器
                    speed_scheduler.load_state_dict(checkpoint['speed_scheduler_state_dict']) # 前提有相同调度器
                    print("学习率调度器加载成功")
                else:
                    print("未加载学习率调度器")
            except:
                print("优化器状态加载失败，将使用初始化状态")
        else:
            print("加载权重文件完成，将从第1轮重新训练")
    except Exception as e:
        print(f"加载权重文件失败,初始化后从头训练") # 模型加载失败则不会加载优化器
    
    
    # 模型初始化，用pytorch默认
    
    ### 训练
###############################################################################    
    # 权重系数初始化
    angle_weight = 1 
    speed_weight = 1
    
###############################################################################
    # 定义传感器量程 （数据集最大标签不一定是传感器量程）
    MAX_SPEED_FULLSCALE = 600
    MAX_ANGLE_FULLSCALE = 2 * np.pi
    # 每10轮保存一次中间权重
    SAVE_WEIGHT_EPOCH_NUM = 1
    # 每20轮更新一次后处理，累计，覆盖之前的
    SAVE_DATA_EPOCH_NUM = 2
    # best_Loss初始化
    best_angle_loss = float('inf')
    best_speed_loss = float('inf')
    # 记录训练和测试损失
    train_losses = []
    train_angle_losses = []
    train_speed_losses = []
    test_losses = []
    test_angle_losses = []
    test_speed_losses = []
    # 记录绝对误差和相对误差
    speed_mae_list = []
    angle_mae_list = []
    speed_mre_list = []
    angle_mre_list = []
    
    ## 训练循环
    for epoch in range(start_epoch, num_epochs):  # 从start_epoch开始训练
        print(f"\n第 {epoch+1}/{num_epochs} 轮训练开始...")
        print(f"当前角度学习率: {angle_scheduler.get_last_lr()[0]:.6f}")
        print(f"当前速度学习率: {speed_scheduler.get_last_lr()[0]:.6f}")
        
        ## 训练模式
        model.train()
        # 初始化损失计量器
        train_loss_meter = AverageMeter()
        train_angle_loss_meter = AverageMeter()
        train_speed_loss_meter = AverageMeter()
        
        # 单次训练
        train_pbar = tqdm(
            train_dataloader, 
            desc='training',
            ascii=True,  # 使用 ASCII 字符
            mininterval=0.1,  # 最小更新间隔
        )
        for batch_idx, (img_orig, img_flow, img_clus, target) in enumerate(train_pbar):
            # 将所有输入数据移到设备上
            img_orig = img_orig.to(device)
            img_flow = img_flow.to(device)
            img_clus = img_clus.to(device)
            target = target.to(device)
            
            # 梯度清零
            angle_optimizer.zero_grad()
            speed_optimizer.zero_grad()
            
            # 前向传播 - 传入三个输入
            output = model(img_orig, img_flow, img_clus)
            
            # 分离预测值
            speed_pred = output[:, 0]      # [batch]
            angle_pred = output[:, 1:model.out_channels]      # [batch, 2]
            
            # 获取原始标签（未归一化）即真实值
            raw_speed_label = target[:, 0]
            raw_angle_label = target[:, 1]
            ##### 标签转换 #####
            # 将角度转换为cos和sin (不用再归一化)
            angle_target_rad = raw_angle_label  # 角度在第二列
            angle_target = torch.stack([
                torch.cos(angle_target_rad), 
                torch.sin(angle_target_rad)    
            ], dim=1).to(device)            # [batch, 2]
            # 将速度归一化
            speed_target = raw_speed_label / 600 ## 速度最大标签是600     # 速度在第一列
            ##### end #####
            
            # 计算损失
            angle_loss = angle_criterion(angle_pred, angle_target)
            speed_loss = speed_criterion(speed_pred, speed_target)
            total_loss = angle_weight * angle_loss + speed_weight * speed_loss

            # 反向传播
            angle_loss.backward(retain_graph=True)  # 保留计算图以便后续使用
            speed_loss.backward()
            
            # 梯度裁剪，设置更大的阈值以允许更多的梯度信息传递
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            # 优化器更新
            angle_optimizer.step()
            speed_optimizer.step()
            
            # 更新损失计量器
            train_loss_meter.update(total_loss.item(), img_orig.size(0))
            train_angle_loss_meter.update(angle_loss.item(), img_orig.size(0))
            train_speed_loss_meter.update(speed_loss.item(), img_orig.size(0))
            
            # 更新进度条信息
            train_pbar.set_postfix({'train_loss': f'{total_loss.item():.6f}'})
            
        # 保存训练损失
        train_losses.append(train_loss_meter.avg)
        train_angle_losses.append(train_angle_loss_meter.avg)
        train_speed_losses.append(train_speed_loss_meter.avg)
        print(f'训练集总损失: {train_loss_meter.avg:.6f}')
        print(f'训练集角度损失: {train_angle_loss_meter.avg:.6f}')
        print(f'训练集速度损失: {train_speed_loss_meter.avg:.6f}')
        
        # 每个epoch结束后清理
        torch.cuda.empty_cache()
        # 更新学习率（仅在train中）
        angle_scheduler.step()
        speed_scheduler.step()
        

        ## 测试模式
        model.eval()
        # 初始化损失计量器
        test_loss_meter = AverageMeter()
        angle_loss_meter = AverageMeter()
        speed_loss_meter = AverageMeter()
        # 添加误差计量器
        speed_mae_meter = AverageMeter()
        angle_mae_meter = AverageMeter()
        speed_mre_meter = AverageMeter() ## 相对误差的储存其实不必要 
        angle_mre_meter = AverageMeter()
        
        test_pbar = tqdm(
            test_dataloader, 
            desc='testing',
            ascii=True,
            mininterval=0.1,
        )
        with torch.no_grad():
            for batch_idx, (img_orig, img_flow, img_clus, target) in enumerate(test_pbar):
                # 将所有输入数据移到设备上
                img_orig = img_orig.to(device)
                img_flow = img_flow.to(device)
                img_clus = img_clus.to(device)
                target = target.to(device)
                
                # 前向传播 - 传入三个输入
                output = model(img_orig, img_flow, img_clus)
                
                # 分离预测值
                speed_pred = output[:, 0]      # [batch]
                angle_pred = output[:, 1:model.out_channels]      # [batch, 2]
                
                # 获取原始标签（未归一化）即真实值
                raw_speed_label = target[:, 0]
                raw_angle_label = target[:, 1]

                # 标签归一化
                # 将角度转换为cos和sin (-1, 1)
                angle_target_rad = raw_angle_label # rad
                angle_target = torch.stack([ # cos,sin
                    torch.cos(angle_target_rad), # 注意：角度在第二列
                    torch.sin(angle_target_rad)  
                ], dim=1).to(device)         # [batch, 2]
                # 将速度归一化
                speed_target = raw_speed_label / 600 ## 速度最大标签是600   # 速度在第一列
                
                # 计算损失
                angle_loss = angle_criterion(angle_pred, angle_target)
                speed_loss = speed_criterion(speed_pred, speed_target)
                total_loss = angle_weight * angle_loss + speed_weight * speed_loss
                
                # 计算相对误差和绝对误差
                # 反归一化预测值
                speed_pred_value = speed_pred * 600  # 还原速度预测值
                angle_pred_value = std_decode_angle(angle_pred)  # 解码角度预测值
                
                # 计算角度差值（考虑2pi周期性）
                angle_diff = torch.abs(angle_pred_value - raw_angle_label)
                angle_diff = torch.minimum(angle_diff, 2 * np.pi - angle_diff)
                
                # 计算MAE
                speed_mae = torch.abs(speed_pred_value - raw_speed_label).mean()
                angle_mae = angle_diff.mean()
                
                # 计算MRE （相对总量程）
                speed_mre = speed_mae / MAX_SPEED_FULLSCALE ## 传感器速度量程
                angle_mre = angle_mae / MAX_ANGLE_FULLSCALE ## 传感器角度量程
                
                # 更新误差计量器
                speed_mae_meter.update(speed_mae.item(), img_orig.size(0))
                angle_mae_meter.update(angle_mae.item(), img_orig.size(0))
                speed_mre_meter.update(speed_mre.item(), img_orig.size(0))
                angle_mre_meter.update(angle_mre.item(), img_orig.size(0))
                
                # 更新计量器
                test_loss_meter.update(total_loss.item(), img_orig.size(0))
                angle_loss_meter.update(angle_loss.item(), img_orig.size(0))
                speed_loss_meter.update(speed_loss.item(), img_orig.size(0))
                
                # 更新进度条信息
                test_pbar.set_postfix({
                    'test_loss': f'{total_loss.item():.6f}'
                })
        
        # 每个epoch结束后清理
        torch.cuda.empty_cache()
        
        # 保存测试损失
        test_loss = test_loss_meter.avg
        test_angle_loss = angle_loss_meter.avg
        test_speed_loss = speed_loss_meter.avg
        test_losses.append(test_loss)
        test_angle_losses.append(test_angle_loss)
        test_speed_losses.append(test_speed_loss)
        
        # 保存误差数据
        speed_mae_list.append(speed_mae_meter.avg)
        angle_mae_list.append(angle_mae_meter.avg)
        speed_mre_list.append(speed_mre_meter.avg)
        angle_mre_list.append(angle_mre_meter.avg)
        
        print(f'测试集总损失: {test_loss:.6f}')
        print(f'测试集角度损失: {test_angle_loss:.6f}')
        print(f'测试集速度损失: {test_speed_loss:.6f}')
        
        
        # 存最佳模型
        if test_angle_loss < best_angle_loss and test_speed_loss < best_speed_loss:
            best_angle_loss = test_angle_loss
            best_speed_loss = test_speed_loss
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'angle_optimizer_state_dict': angle_optimizer.state_dict(),
                'speed_optimizer_state_dict': speed_optimizer.state_dict(),
                'angle_scheduler_state_dict': angle_scheduler.state_dict(),  # 前提有调度器
                'speed_scheduler_state_dict': speed_scheduler.state_dict(),  # 前提有调度器
            }
        
        # 每5轮保存一次中间权重
        if (epoch + 1) % SAVE_WEIGHT_EPOCH_NUM == 0:
            # 保存当前epoch的模型权重
            current_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'angle_optimizer_state_dict': angle_optimizer.state_dict(),
                'speed_optimizer_state_dict': speed_optimizer.state_dict(),
                'angle_scheduler_state_dict': angle_scheduler.state_dict(),  # 调度器
                'speed_scheduler_state_dict': speed_scheduler.state_dict(),  # 调度器
            }
            # 以epoch命名保存中间权重
            torch.save(current_model_state, f"{pth_save_dir}/epoch_{epoch//SAVE_WEIGHT_EPOCH_NUM*SAVE_WEIGHT_EPOCH_NUM}_{(epoch//SAVE_WEIGHT_EPOCH_NUM+1)*SAVE_WEIGHT_EPOCH_NUM}.pth")
            print(f"保存第{epoch//SAVE_WEIGHT_EPOCH_NUM*SAVE_WEIGHT_EPOCH_NUM}轮到第{(epoch//SAVE_WEIGHT_EPOCH_NUM+1)*SAVE_WEIGHT_EPOCH_NUM}轮模型权重，测试损失: {test_loss:.6f}")
            # 每5轮同时更新一次最优模型
            torch.save(best_model_state, f"{pth_save_dir}/best.pth")
            print(f"保存最优模型, Epoch为{best_model_state['epoch']+1}，角度损失: {best_angle_loss:.6f}，速度损失: {best_speed_loss:.6f}")

        ### 后处理
        # 每20轮更新后处理（覆盖之前的）：为了能中途打断程序
        if (epoch + 1) % SAVE_DATA_EPOCH_NUM == 0:
            ## 调用绘图函数
            plot_losscurve(train_losses, test_losses, save_path=losscurve_save_path / 'losscurve.png', start_epoch=start_epoch+1)
            plot_losscurve(train_angle_losses, test_angle_losses, save_path=losscurve_save_path / 'angle_losscurve.png', start_epoch=start_epoch+1)
            plot_losscurve(train_speed_losses, test_speed_losses, save_path=losscurve_save_path / 'speed_losscurve.png', start_epoch=start_epoch+1)
            ## 保存训练损失数据到csv    
            save_detail_lossdata(train_losses, test_losses, test_angle_losses, test_speed_losses, save_path=lossdata_save_path, start_epoch=start_epoch+1)
            ## 保存误差数据到csv
            save_errordata(speed_mae_list, angle_mae_list, speed_mre_list, angle_mre_list, save_path=errordata_save_path, start_epoch=start_epoch+1)

    print("\n训练完成!")
