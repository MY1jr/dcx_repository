import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
import cv2
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
###############################################################################
# 平均损失计算
class AverageMeter(object):
    """Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
###############################################################################

class CircleDataset(Dataset): 

    def __init__(self, label_txt, img_transform=None):
        self.label_txt = label_txt
        self.img_transform = img_transform 
        # 读取标签文件
        with open(self.label_txt, 'r') as f:
            self.data = [line.strip().split(',') for line in f]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 获取图片路径
        img_path = self.data[idx][0]
        # 获取6个标签值:内圆心x,y,r,外圆心x,y,r
        label = torch.tensor([float(x) for x in self.data[idx][1:7]])
        
        # 检查图像文件是否存在
        if not Path(img_path).exists():
            raise FileNotFoundError(f"图像文件不存在: {img_path}")
            
        # 读取图像
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        
        # 检查图像是否成功读取
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
            
        # 图像预处理
        if self.img_transform:
            img = self.img_transform(img)
            
        # 转换为tensor
        img = torch.from_numpy(img).float()
        # 调整维度顺序为[C,H,W]
        img = img.permute(2, 0, 1)
    
        return img, label
###############################################################################
# 图像预处理函数
def img_transform(img):
    # 图像缩放成64x64
    img = cv2.resize(img, (64, 64))
    img = img / 255.0  # 图像归一化
    
    return img
###############################################################################
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # 定义池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 定义全连接层
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # 假设输入图像大小为64x64
        self.fc2 = nn.Linear(128, self.out_channels)  # 输出一个回归值

    def forward(self, x):
        # 卷积层 + 激活函数 + 池化层
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

###############################################################################
# 数据集标签是[speed，angle（rad）], 而angle在网络中是使用（cos，sin表示的）
if __name__ == "__main__":
    # 训练超参数
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.01 # 0.001
    # 断点重训
    resume_training = True # True
    # 保存路径
    pth_save_dir = 'G:/NN_py/circle_detection/circlenet/pth'
    pth_load_path = 'G:/NN_py/circle_detection/circlenet/pth/best.pth'
    # 确保保存路径存在
    Path(pth_save_dir).mkdir(parents=True, exist_ok=True)
###############################################################################
    # # 设置随机种子以确保可重复性
    # torch.manual_seed(seed=42)
    # torch.cuda.manual_seed_all(seed=42)

    # 数据集定义
    source_dataset = CircleDataset(
        label_txt="G:/NN_py/dcx_mini/circles_label.txt", 
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

###############################################################################
    ##### 模型接口守则：#####
    # （1）必须定义有self.out_channels的类公有参数 
    # （2）输出[0:3]是内圆（x, y, 半径），输出[3:6]是外圆（x, y, 半径）
    ##### end ##### 
    
    model = SimpleCNN(in_channels=3, out_channels=6).to(device)
    # print("\nModel:") 
    # print(model) # 打印模型结构
###############################################################################

###############################################################################
    # 定义损失函数
    circle_criterion = nn.MSELoss()  # 使用MSELoss计算圆心和半径的损失
###############################################################################
    # 定义优化器
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # 学习率调度器
    ## 超参数 ##
    num_warmup_epochs = int(num_epochs * 0.05)  # warmup轮数5% = 5轮
    warmup_factor = 0.1    # warmup起始学习率因子
    cosine_T_max = (num_epochs - num_warmup_epochs) // 2  # 余弦退火半周期
    min_lr = learning_rate * 0.01  # 最小学习率为初始学习率的1%
    ## 调度器
    scheduler = ChainedScheduler([
        # warmup：从 warmup_factor*lr 线性增加到 lr
        LinearLR(
            optimizer, 
            start_factor=warmup_factor,
            total_iters=num_warmup_epochs
        ),
        # 余弦退火：从 lr 降到 min_lr
        CosineAnnealingLR(
            optimizer,
            T_max=cosine_T_max,  # 余弦退火的周期
            eta_min=min_lr  # 最小学习率
        )
    ])

    # 加载最佳模型权重，默认加载最新保存的模型
    start_epoch = 0  # 起始epoch变量初始化
    try:    
        checkpoint = torch.load(pth_load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if resume_training:
            start_epoch = checkpoint['epoch'] + 1  # 设置起始epoch（未训练的epoch）# 从0计数
            print(f"加载权重文件完成, 将从第{start_epoch + 1}轮继续训练") # 从1计数
            try:
                # 恢复优化器状态
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # 恢复学习率调度器状态
                scheduler.load_state_dict(checkpoint['scheduler_state_dict']) # 前提有相同调度器
                print("优化器和学习率调度器状态加载成功")
            except:
                print("优化器状态加载失败，将使用初始化的优化器状态")
        else:
            print("加载权重文件完成，将从第1轮重新训练")
    except Exception as e:
        print(f"加载权重文件失败,初始化后从头训练") # 模型加载失败则不会加载优化器
    
    
    # 模型初始化，用pytorch默认
    
    ### 训练
###############################################################################    
    # 权重系数初始化
    inner_circle_weight = 1.0
    outer_circle_weight = 1.0
###############################################################################
    # 每10轮保存一次中间权重
    SAVE_WEIGHT_EPOCH_NUM = 1
    # best_Loss初始化
    best_loss = float('inf')
    # 记录训练和测试损失
    train_losses = []
    test_losses = []

    ## 训练循环
    for epoch in range(start_epoch, num_epochs):  # 从start_epoch开始训练
        print(f"\n第 {epoch+1}/{num_epochs} 轮训练开始...")
        print(f"当前学习率: {scheduler.get_last_lr()[0]:.6f}")
        
        ## 训练模式
        model.train()
        # 训练损失累加器初始化
        total_train_loss = 0
        # 单次训练
        train_pbar = tqdm(train_dataloader, desc='training')
        for batch_idx, (data, target) in enumerate(train_pbar):
            # 将数据和标签移动到GPU
            data, target = data.to(device), target.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            output = model(data)
            
            # 分离预测值
            inner_circle_pred = output[:, 0:3]  # [batch, 3]
            outer_circle_pred = output[:, 3:6]  # [batch, 3]
            
            # 获取原始标签
            inner_circle_target = target[:, 0:3]  # [batch, 3]
            outer_circle_target = target[:, 3:6]  # [batch, 3]
            
            # 计算损失
            inner_circle_loss = circle_criterion(inner_circle_pred, inner_circle_target)
            outer_circle_loss = circle_criterion(outer_circle_pred, outer_circle_target)
            
            # 权重系数 平衡内外圆损失
            loss = inner_circle_weight * inner_circle_loss + outer_circle_weight * outer_circle_loss
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪，设置更大的阈值以允许更多的梯度信息传递
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            # 优化器更新
            optimizer.step()
            # 累加损失
            total_train_loss += loss.item()
            
            # 更新进度条信息
            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
        # 保存训练损失
        train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(train_loss)
        print(f'训练集平均损失: {train_loss:.6f}')
        
        # 更新学习率（仅在train中）
        scheduler.step()
        

        ## 测试模式
        model.eval()
        # 初始化损失计量器
        test_loss_meter = AverageMeter()

        test_pbar = tqdm(test_dataloader, desc='testing')
        with torch.no_grad():
            for data, target in test_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                # 分离预测值
                inner_circle_pred = output[:, 0:3]
                outer_circle_pred = output[:, 3:6]
                
                # 获取原始标签
                inner_circle_target = target[:, 0:3]
                outer_circle_target = target[:, 3:6]
                
                # 计算损失
                inner_circle_loss = circle_criterion(inner_circle_pred, inner_circle_target)
                outer_circle_loss = circle_criterion(outer_circle_pred, outer_circle_target)
                loss = inner_circle_weight * inner_circle_loss + outer_circle_weight * outer_circle_loss
                
                # 更新损失计量器
                test_loss_meter.update(loss.item(), data.size(0))
                
                # 更新进度条信息
                test_pbar.set_postfix({
                    'loss': f'{test_loss_meter.avg:.6f}'
                })
        
        # 保存测试损失
        test_loss = test_loss_meter.avg
        test_losses.append(test_loss)
        
        print(f'测试集平均损失: {test_loss:.6f}')

        # 存最佳模型
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),  # 前提有调度器                
                'loss': best_loss,
            }
        
        # 每5轮保存一次中间权重
        if (epoch + 1) % SAVE_WEIGHT_EPOCH_NUM == 0:
            # 保存当前epoch的模型权重
            current_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),  # 调度器                
                'loss': test_loss,
            }
            # 以epoch命名保存中间权重
            torch.save(current_model_state, f"{pth_save_dir}/epoch_{epoch//SAVE_WEIGHT_EPOCH_NUM*SAVE_WEIGHT_EPOCH_NUM}_{(epoch//SAVE_WEIGHT_EPOCH_NUM+1)*SAVE_WEIGHT_EPOCH_NUM}.pth")
            print(f"保存第{epoch//SAVE_WEIGHT_EPOCH_NUM*SAVE_WEIGHT_EPOCH_NUM}轮到第{(epoch//SAVE_WEIGHT_EPOCH_NUM+1)*SAVE_WEIGHT_EPOCH_NUM}轮模型权重，测试损失: {test_loss:.6f}")
            # 每5轮同时更新一次最优模型
            torch.save(best_model_state, f"{pth_save_dir}/best.pth")
            print(f"保存最优模型, Epoch为{best_model_state['epoch']+1}，测试损失: {best_loss:.6f}")

    print("\n训练完成!")
    



