import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
import cv2
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from utils.data_postprocess import plot_losscurve, save_lossdata, save_errordata
from model_d19 import GTNet
from utils.misc import AverageMeter
from utils.decode_angle import std_decode_angle

from circleaugment import CircleAugmentation
from dcx_dataset import dcxDataset

###############################################################################
# 图像预处理函数 ### 后续改成（100，100）代价是降低了batch=16 时间加倍
def img_transform(img):
    # img = cv2.resize(img, (64, 64))
    img = cv2.resize(img, (100, 100))
    img = img / 255.0  # 图像归一化
    
    return img
###############################################################################

# 加载预训练权重函数
def load_pretrained_weights(model, pretrained_path, device):
    print("Loading pretrained weights...")
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_path, map_location=device)
    temp = {}
    for k, v in pretrained_dict.items():
        try:    
            if np.shape(model_dict[k]) == np.shape(v):
                temp[k] = v
                # print(f"Loading weight: {k}")
        except:
            pass
    model_dict.update(temp)
    model.load_state_dict(model_dict)
    print("Pretrained weights loaded successfully")


###############################################################################
# 数据集标签是[speed，angle（rad）], 而angle在网络中是使用（cos，sin表示的）
if __name__ == "__main__":
    # 训练超参数
    batch_size = 16
    learning_rate = 0.001 # 0.001
    time_step = 20
    
    
    # 是否进行冻结训练
    Freeze_Train = True
    # 冻结参数设置
    if Freeze_Train:
        # 冻结训练begin
        Init_Epoch = 0
        # 冻结训练end
        Freeze_Epoch = 50
        Freeze_lr = learning_rate * 0.01
        # 解冻训练begin
        UnFreeze_Epoch = 100
        Unfreeze_lr = learning_rate

    
    # 保存路径
    pth_save_dir = 'G:/NN_py/NN_test_py/pth/model_d19'
    pth_load_path = 'G:/NN_py/NN_test_py/pth/model_d19/best.pth'
    pretrained_path = 'G:/NN_py/NN_test_py/pre_weight/darknet19.pth'  # 预训练权重路径
    losscurve_save_path = 'G:/NN_py/NN_test_py/pth/model_d19/loss_curves.png'
    lossdata_save_path = 'G:/NN_py/NN_test_py/pth/model_d19/loss_data.csv'
    errordata_save_path = 'G:/NN_py/NN_test_py/pth/model_d19/error_data.csv'
    

    # 数据定义
    source_dataset = dcxDataset(
        label_txt="G:/NN_py/dcx_mini/dcx_mini.txt", 
        time_step=time_step,
        img_transform=img_transform,
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
    train_dataset.dataset._set_seq_augmentation(CircleAugmentation(), 0.2)
    
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

###################################
    ##### 定义模型接口守则：#####
    # （1）必须定义有self.out_channels的类公有参数 
    # （2）输出[0]是速度，其余输出是角度（cos,sin），一般为2个 
    ##### end ##### 
    
    model = GTNet(
        in_channels=3,
        out_channels=3,
        time_step=time_step,
        height=100,
        width=100
    ).to(device)
    
    # 加载预训练权重
    load_pretrained_weights(model, pretrained_path, device)
    
    print("\nModel:")
    print(model)

    # 定义损失函数
    angle_criterion = nn.MSELoss()  
    speed_criterion = nn.MSELoss() 

    # 权重系数初始化
    angle_weight = 1
    speed_weight = 1
    
    # 定义传感器量程
    MAX_SPEED_FULLSCALE = 600
    MAX_ANGLE_FULLSCALE = 2 * np.pi
    # 每10轮保存一次中间权重
    SAVE_WEIGHT_EPOCH_NUM = 2
    # 每10轮更新一次后处理
    SAVE_DATA_EPOCH_NUM = 5
    # best_Loss初始化
    best_loss = float('inf')
    # 记录训练和测试损失
    train_losses = []
    test_losses = []
    # 记录误差
    speed_mae_list = []
    angle_mae_list = []
    speed_mre_list = []
    angle_mre_list = []

    if Freeze_Train:
        print("Starting freeze training phase...")
        
        # 使用冻结阶段参数
        start_epoch = Init_Epoch
        end_epoch = Freeze_Epoch
        
        # 冻结backbone
        for param in model.backbone.parameters():
            param.requires_grad = False
            
        # 将模型参数分成两组
        backbone_params = model.backbone.parameters()
        other_params = [p for n, p in model.named_parameters() if not n.startswith('backbone.')]
        
        # 分别定义优化器
        optimizer_backbone = torch.optim.Adam(
            backbone_params,
            lr=Freeze_lr  # 主干使用固定的较小学习率
        )
        
        optimizer_other = torch.optim.Adam(
            other_params, 
            lr=learning_rate # 非主干部分使用学习率
        )
        
        # 只对非主干部分使用学习率调度器
        num_warmup_epochs = int(Freeze_Epoch * 0.1)
        warmup_factor = 0.01
        cosine_T_max = (Freeze_Epoch - num_warmup_epochs) // 2
        min_lr = learning_rate * 0.01
        
        scheduler_other = ChainedScheduler([
            LinearLR(optimizer_other, start_factor=warmup_factor, total_iters=num_warmup_epochs),
            CosineAnnealingLR(optimizer_other, T_max=cosine_T_max, eta_min=min_lr)
        ])
        
        # 冻结阶段训练循环
        for epoch in range(start_epoch, end_epoch):
            print(f"\n第 {epoch+1}/{end_epoch} 轮训练开始...")
            print(f"backbone lr: {Freeze_lr}")
            print(f"other lr: {scheduler_other.get_last_lr()[0]:.6f}")
            
            model.train()
            total_train_loss = 0
            
            train_pbar = tqdm(train_dataloader, desc='training')
            for batch_idx, (data, target) in enumerate(train_pbar):
                data, target = data.to(device), target.to(device)
                
                # 清空两个优化器的梯度
                optimizer_backbone.zero_grad()
                optimizer_other.zero_grad()
                
                output = model(data)
                
                # 分离预测值
                speed_pred = output[:, 0]
                angle_pred = output[:, 1:model.out_channels]
                
                # 获取原始标签
                raw_speed_label = target[:, 0]
                raw_angle_label = target[:, 1]
                
                # 标签转换
                angle_target_rad = raw_angle_label
                angle_target = torch.stack([
                    torch.cos(angle_target_rad),
                    torch.sin(angle_target_rad)
                ], dim=1).to(device)
                speed_target = raw_speed_label / 600
                
                # 计算损失
                angle_loss = angle_criterion(angle_pred, angle_target)
                speed_loss = speed_criterion(speed_pred, speed_target)
                loss = angle_weight * angle_loss + speed_weight * speed_loss
                
                loss.backward()
                
                # 分别更新两部分参数
                optimizer_backbone.step()
                optimizer_other.step()
                
                total_train_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
            # 只对非主干部分调整学习率，主干部分是常值学习率
            scheduler_other.step()
            
            train_loss = total_train_loss / len(train_dataloader)
            train_losses.append(train_loss)
            print(f'训练集平均损失: {train_loss:.6f}')
            
            # 测试部分
            model.eval()
            test_loss_meter = AverageMeter()
            angle_loss_meter = AverageMeter()
            speed_loss_meter = AverageMeter()
            speed_mae_meter = AverageMeter()
            angle_mae_meter = AverageMeter()
            speed_mre_meter = AverageMeter()
            angle_mre_meter = AverageMeter()
            
            test_pbar = tqdm(test_dataloader, desc='testing')
            with torch.no_grad():
                for data, target in test_pbar:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    
                    speed_pred = output[:, 0]
                    angle_pred = output[:, 1:model.out_channels]
                    
                    raw_speed_label = target[:, 0]
                    raw_angle_label = target[:, 1]
                    
                    angle_target_rad = raw_angle_label
                    angle_target = torch.stack([
                        torch.cos(angle_target_rad),
                        torch.sin(angle_target_rad)
                    ], dim=1).to(device)
                    speed_target = raw_speed_label / 600
                    
                    angle_loss = angle_criterion(angle_pred, angle_target)
                    speed_loss = speed_criterion(speed_pred, speed_target)
                    loss = angle_weight * angle_loss + speed_weight * speed_loss
                    
                    speed_pred_value = speed_pred * 600
                    angle_pred_value = std_decode_angle(angle_pred)
                    
                    angle_diff = torch.abs(angle_pred_value - raw_angle_label)
                    angle_diff = torch.minimum(angle_diff, 2 * np.pi - angle_diff)
                    
                    speed_mae = torch.abs(speed_pred_value - raw_speed_label).mean()
                    angle_mae = angle_diff.mean()
                    
                    speed_mre = speed_mae / MAX_SPEED_FULLSCALE
                    angle_mre = angle_mae / MAX_ANGLE_FULLSCALE
                    
                    speed_mae_meter.update(speed_mae.item(), data.size(0))
                    angle_mae_meter.update(angle_mae.item(), data.size(0))
                    speed_mre_meter.update(speed_mre.item(), data.size(0))
                    angle_mre_meter.update(angle_mre.item(), data.size(0))
                    
                    test_loss_meter.update(loss.item(), data.size(0))
                    angle_loss_meter.update(angle_loss.item(), data.size(0))
                    speed_loss_meter.update(speed_loss.item(), data.size(0))
                    
                    test_pbar.set_postfix({'loss': f'{test_loss_meter.avg:.6f}'})
            
            # 更新损失和误差记录
            test_loss = test_loss_meter.avg
            test_losses.append(test_loss)
            speed_mae_list.append(speed_mae_meter.avg)
            angle_mae_list.append(angle_mae_meter.avg)
            speed_mre_list.append(speed_mre_meter.avg)
            angle_mre_list.append(angle_mre_meter.avg)
            
            print(f'测试集平均损失: {test_loss:.6f}')
            print(f'测试集角度损失: {angle_loss_meter.avg:.6f}')
            print(f'测试集速度损失: {speed_loss_meter.avg:.6f}')
            
            if test_loss < best_loss:
                best_loss = test_loss
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer_other.state_dict(),
                    'loss': best_loss,
                }
            
            if (epoch + 1) % SAVE_WEIGHT_EPOCH_NUM == 0:
                current_model_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer_other.state_dict(),
                    'loss': test_loss,
                }
                torch.save(current_model_state, 
                         f"{pth_save_dir}/epoch_{epoch//SAVE_WEIGHT_EPOCH_NUM*SAVE_WEIGHT_EPOCH_NUM}_{(epoch//SAVE_WEIGHT_EPOCH_NUM+1)*SAVE_WEIGHT_EPOCH_NUM}.pth")
                torch.save(best_model_state, f"{pth_save_dir}/best.pth")
            
            if (epoch + 1) % SAVE_DATA_EPOCH_NUM == 0:
                plot_losscurve(train_losses, test_losses, save_path=losscurve_save_path, start_epoch=start_epoch+1)
                save_lossdata(train_losses, test_losses, save_path=lossdata_save_path, start_epoch=start_epoch+1)
                save_errordata(speed_mae_list, angle_mae_list, speed_mre_list, angle_mre_list, 
                             save_path=errordata_save_path, start_epoch=start_epoch+1)
        
        print("Freeze training phase completed")
        print("Starting unfreeze training phase...")
        
        # 解冻backbone
        for param in model.backbone.parameters():
            param.requires_grad = True
        
        # 使用解冻阶段参数
        start_epoch = Freeze_Epoch
        end_epoch = UnFreeze_Epoch
        
        # 重新定义优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=Unfreeze_lr)
        
        # 重新定义学习率调度器
        num_warmup_epochs = int((UnFreeze_Epoch - Freeze_Epoch) * 0.1)
        cosine_T_max = ((UnFreeze_Epoch - Freeze_Epoch) - num_warmup_epochs) // 2
        
        scheduler = ChainedScheduler([
            LinearLR(optimizer, start_factor=warmup_factor, total_iters=num_warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=cosine_T_max, eta_min=min_lr)
        ])
        
        # 解冻阶段训练循环(与冻结阶段相同的训练循环)
        for epoch in range(start_epoch, end_epoch):
            print(f"\n第 {epoch+1}/{end_epoch} 轮训练开始...")
            print(f"当前学习率: {scheduler.get_last_lr()[0]:.6f}")
            
            # 训练模式
            model.train()
            total_train_loss = 0
            
            train_pbar = tqdm(train_dataloader, desc='training')
            for batch_idx, (data, target) in enumerate(train_pbar):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                
                # 分离预测值
                speed_pred = output[:, 0]
                angle_pred = output[:, 1:model.out_channels]
                
                # 获取原始标签
                raw_speed_label = target[:, 0]
                raw_angle_label = target[:, 1]
                
                # 标签转换
                angle_target_rad = raw_angle_label
                angle_target = torch.stack([
                    torch.cos(angle_target_rad),
                    torch.sin(angle_target_rad)
                ], dim=1).to(device)
                speed_target = raw_speed_label / 600
                
                # 计算损失
                angle_loss = angle_criterion(angle_pred, angle_target)
                speed_loss = speed_criterion(speed_pred, speed_target)
                loss = angle_weight * angle_loss + speed_weight * speed_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                
                total_train_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
            train_loss = total_train_loss / len(train_dataloader)
            train_losses.append(train_loss)
            print(f'训练集平均损失: {train_loss:.6f}')
            
            # 测试部分
            model.eval()
            test_loss_meter = AverageMeter()
            angle_loss_meter = AverageMeter()
            speed_loss_meter = AverageMeter()
            speed_mae_meter = AverageMeter()
            angle_mae_meter = AverageMeter()
            speed_mre_meter = AverageMeter()
            angle_mre_meter = AverageMeter()
            
            test_pbar = tqdm(test_dataloader, desc='testing')
            with torch.no_grad():
                for data, target in test_pbar:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    
                    speed_pred = output[:, 0]
                    angle_pred = output[:, 1:model.out_channels]
                    
                    raw_speed_label = target[:, 0]
                    raw_angle_label = target[:, 1]
                    
                    angle_target_rad = raw_angle_label
                    angle_target = torch.stack([
                        torch.cos(angle_target_rad),
                        torch.sin(angle_target_rad)
                    ], dim=1).to(device)
                    speed_target = raw_speed_label / 600
                    
                    angle_loss = angle_criterion(angle_pred, angle_target)
                    speed_loss = speed_criterion(speed_pred, speed_target)
                    loss = angle_weight * angle_loss + speed_weight * speed_loss
                    
                    speed_pred_value = speed_pred * 600
                    angle_pred_value = std_decode_angle(angle_pred)
                    
                    angle_diff = torch.abs(angle_pred_value - raw_angle_label)
                    angle_diff = torch.minimum(angle_diff, 2 * np.pi - angle_diff)
                    
                    speed_mae = torch.abs(speed_pred_value - raw_speed_label).mean()
                    angle_mae = angle_diff.mean()
                    
                    speed_mre = speed_mae / MAX_SPEED_FULLSCALE
                    angle_mre = angle_mae / MAX_ANGLE_FULLSCALE
                    
                    speed_mae_meter.update(speed_mae.item(), data.size(0))
                    angle_mae_meter.update(angle_mae.item(), data.size(0))
                    speed_mre_meter.update(speed_mre.item(), data.size(0))
                    angle_mre_meter.update(angle_mre.item(), data.size(0))
                    
                    test_loss_meter.update(loss.item(), data.size(0))
                    angle_loss_meter.update(angle_loss.item(), data.size(0))
                    speed_loss_meter.update(speed_loss.item(), data.size(0))
                    
                    test_pbar.set_postfix({'loss': f'{test_loss_meter.avg:.6f}'})
            
            # 更新损失和误差记录
            test_loss = test_loss_meter.avg
            test_losses.append(test_loss)
            speed_mae_list.append(speed_mae_meter.avg)
            angle_mae_list.append(angle_mae_meter.avg)
            speed_mre_list.append(speed_mre_meter.avg)
            angle_mre_list.append(angle_mre_meter.avg)
            
            print(f'测试集平均损失: {test_loss:.6f}')
            print(f'测试集角度损失: {angle_loss_meter.avg:.6f}')
            print(f'测试集速度损失: {speed_loss_meter.avg:.6f}')
            
            scheduler.step()
            
            if test_loss < best_loss:
                best_loss = test_loss
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }
            
            if (epoch + 1) % SAVE_WEIGHT_EPOCH_NUM == 0:
                current_model_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': test_loss,
                }
                torch.save(current_model_state, 
                         f"{pth_save_dir}/epoch_{epoch//SAVE_WEIGHT_EPOCH_NUM*SAVE_WEIGHT_EPOCH_NUM}_{(epoch//SAVE_WEIGHT_EPOCH_NUM+1)*SAVE_WEIGHT_EPOCH_NUM}.pth")
                torch.save(best_model_state, f"{pth_save_dir}/best.pth")
            
            if (epoch + 1) % SAVE_DATA_EPOCH_NUM == 0:
                plot_losscurve(train_losses, test_losses, save_path=losscurve_save_path, start_epoch=start_epoch+1)
                save_lossdata(train_losses, test_losses, save_path=lossdata_save_path, start_epoch=start_epoch+1)
                save_errordata(speed_mae_list, angle_mae_list, speed_mre_list, angle_mre_list, 
                             save_path=errordata_save_path, start_epoch=start_epoch+1)

    print("\nTraining completed!")

