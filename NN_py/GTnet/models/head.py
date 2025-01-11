import torch
import torch.nn as nn
import torch.nn.functional as F

class HeadSharedFeature(nn.Module):
    """
    检测头共享权重
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shared_connect = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.SiLU()
        )
    def forward(self, x):
        x = self.shared_connect(x)
        return x

class SpeedHead(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        
        # skip connection
        self.skip_connect = nn.ModuleList([
            # 分支1 - 特征处理
            nn.Sequential(
                nn.Linear(in_channels // 2, in_channels // 4),  # 128->64
                nn.BatchNorm1d(in_channels // 4),
                nn.SiLU(),
                nn.Linear(in_channels // 4, in_channels // 2),  # 64->128
                nn.BatchNorm1d(in_channels // 2),
                nn.SiLU()
            ),
            # 分支2 - 恒等映射
            nn.Identity()
        ])
        
        # 速度预测头
        self.task_head = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),  # 256->128
            nn.BatchNorm1d(in_channels // 2),
            nn.SiLU(),
            nn.Linear(in_channels // 2, out_channels)  # 128->1
        )

    def forward(self, x):
        # C2f风格处理
        x1, x2 = x.chunk(2, dim=1)  # 每个分支512/2=256维
        x1 = self.skip_connect[0](x1)  # 256维
        x2 = self.skip_connect[1](x2)  # 256维
        x = torch.cat([x1, x2], dim=1)  # 512维
        
        # 速度预测
        out = self.task_head(x)
        return out

class AngleHead(nn.Module):
    def __init__(self, in_channels, out_channels=2):
        super().__init__()
        
        # skip connection
        self.skip_connect = nn.ModuleList([
            # 分支1 - 特征处理
            nn.Sequential(
                nn.Linear(in_channels // 2, in_channels // 4),  # 128->64
                nn.BatchNorm1d(in_channels // 4),
                nn.SiLU(),
                nn.Linear(in_channels // 4, in_channels // 2),  # 64->128
                nn.BatchNorm1d(in_channels // 2),
                nn.SiLU()
            ),
            # 分支2 - 恒等映射
            nn.Identity()
        ])
        
        # 角度预测头
        self.task_head = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),  # 256->128
            nn.BatchNorm1d(in_channels // 2),
            nn.SiLU(),
            nn.Linear(in_channels // 2, out_channels),  # 128->2
            nn.Tanh()  # 确保cos和sin输出在[-1,1]范围内
        )
    
    def forward(self, x):
        # C2f风格处理
        x1, x2 = x.chunk(2, dim=1)  # 每个分支512/2=256维
        x1 = self.skip_connect[0](x1)  # 256维
        x2 = self.skip_connect[1](x2)  # 256维
        x = torch.cat([x1, x2], dim=1)  # 512维
        
        # 角度预测
        out = self.task_head(x)
        
        # # 正则化输出，使其模为1 ### 训练时不能加，输出正则化本身并不恰当！影响梯度传播，需要加正则化损失
        # if not self.training: ### 但其实角度解码是会作正则化，测试加了也没用
        #     out = F.normalize(out, p=2, dim=1)
        
        return out