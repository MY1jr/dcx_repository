import torch
from tqdm import tqdm

# standard training step
def train_step(model, train_loader, criterion, optimizer, device):
    """
    训练函数
    参数:
    - model: 神经网络模型
    - train_loader: 训练数据加载器
    - criterion: 损失函数
    - optimizer: 优化器
    - device: 训练设备(GPU/CPU)
    
    返回:
    - avg_loss: 平均训练损失
    """
    model.train()
    total_loss = 0
    
    # 创建进度条
    pbar = tqdm(train_loader, 
                desc='training')
                # disable=not torch.utils.data.get_worker_info()#只在主进程显示进度条
    
    for batch_idx, (data, target) in enumerate(pbar):
        # 数据转移到指定设备
        data, target = data.to(device), target.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        
        # 计算损失
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 累计损失
        total_loss += loss.item()
        
        # 更新进度条信息
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
    # 计算平均损失    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

# standard testing step
def test_step(model, test_loader, criterion, device):
    """
    测试函数
    参数:
    - model: 神经网络模型
    - test_loader: 测试数据加载器  
    - criterion: 损失函数
    - device: 测试设备(GPU/CPU)
    
    返回:
    - avg_loss: 平均测试损失
    - mse: 均方误差
    """
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    # 创建进度条
    pbar = tqdm(test_loader, 
                desc='testing')
                # disable=not torch.utils.data.get_worker_info()
    
    with torch.no_grad():
        for data, target in pbar:
            # 数据转移到指定设备
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            
            # 计算损失
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # 保存预测值和真实值用于计算指标
            predictions.append(output.cpu())
            targets.append(target.cpu())
            
            # 更新进度条信息
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
    # 计算平均损失
    avg_loss = total_loss / len(test_loader)
    
    # 合并所有批次的预测值和真实值
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    
    print(f'\n测试集平均损失: {avg_loss:.6f}')
    
    return avg_loss