import torch
import torch.nn as nn

class BottleneckBlock(nn.Module):
    """Standard Bottleneck Block with depthwise separable convolution"""
    def __init__(
        self,
        in_channels,
        out_channels,
        expansion_factor=0.5
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion_factor)
        
        # 主路径
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.SiLU()
        
        # 深度分离卷积
        self.conv2_dw = nn.Conv2d(
            hidden_channels, 
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.act2 = nn.SiLU()
        
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.act3 = nn.SiLU()

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        out = self.conv2_dw(out)
        out = self.bn2(out)
        out = self.act2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out = out + identity
        out = self.act3(out)
        
        return out

class CSPBlock(nn.Module):
    """Cross Stage Partial Block"""
    def __init__(
        self, 
        in_channels,
        out_channels,
        num_bottlenecks=1,
        expansion_factor=0.5
    ):
        super().__init__()
        hidden_channels = int(in_channels * expansion_factor)
        if hidden_channels % 2 != 0: # hidden_channels必须是偶数
            hidden_channels += 1
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        
        self.norm = nn.BatchNorm2d(hidden_channels)
        self.act = nn.SiLU()
        
        self.bottleneck_blocks = nn.Sequential(
            *[BottleneckBlock(
                in_channels=hidden_channels // 2,
                out_channels=hidden_channels // 2,
                expansion_factor=expansion_factor
            ) for _ in range(num_bottlenecks)]
        )
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x_split = torch.chunk(x, 2, dim=1)
        
        main_path = self.bottleneck_blocks(x_split[0])
        cross_path = x_split[1]
        
        merged = torch.cat((main_path, cross_path), dim=1)
        merged = self.act(self.norm(merged))
        out = self.conv3(merged)
        return out