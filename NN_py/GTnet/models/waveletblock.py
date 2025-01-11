import torch
import torch.nn as nn
import pytorch_wavelets as pw
from models.cspblock import CSPBlock

class WaveletUnit(nn.Module):
    def __init__(self, in_channels, out_channels, wave='db1', mode='zero', groups=1):
        super(WaveletUnit, self).__init__()
        self.groups = groups
        
        # 创建小波变换和逆变换
        self.dwt = pw.DWTForward(J=1, wave=wave, mode=mode).double()
        self.idwt = pw.DWTInverse(wave=wave, mode=mode).double()
        
        # 将所有层设置为double类型
        self.conv_low = nn.Conv2d(in_channels=in_channels, 
                                 out_channels=out_channels,
                                 kernel_size=1, stride=1, padding=0, 
                                 groups=self.groups, bias=False).double()
        
        self.conv_high = nn.Conv2d(in_channels=in_channels * 3, 
                                  out_channels=out_channels * 3,
                                  kernel_size=1, stride=1, padding=0, 
                                  groups=self.groups, bias=False).double()
        
        self.bn_low = nn.BatchNorm2d(out_channels).double()
        self.bn_high = nn.BatchNorm2d(out_channels * 3).double()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 确保输入是double类型
        x = x.double()
        
        # 执行小波变换
        yl, yh = self.dwt(x)
        
        # 处理低频分量
        low_freq = self.conv_low(yl)
        low_freq = self.relu(self.bn_low(low_freq))
        
        # 处理高频分量
        batch, channel, direction, height, width = yh[0].shape
        high_freq = yh[0].permute(0, 2, 1, 3, 4).contiguous()
        high_freq = high_freq.view(batch, channel * 3, height, width)
        
        high_freq = self.conv_high(high_freq)
        high_freq = self.relu(self.bn_high(high_freq))
        
        # 重新整理高频分量的形状
        high_freq = high_freq.view(batch, 3, -1, height, width)
        high_freq = high_freq.permute(0, 2, 1, 3, 4).contiguous()
        
        # 将处理后的系数重新组合
        yh_processed = [high_freq]
        
        # 执行逆变换
        output = self.idwt((low_freq, yh_processed))
        
        # 转回float类型（如果需要）
        output = output.float()
        
        return output


class WaveletBlock(nn.Module):
    def __init__(self, dims, wave='db1', mode='zero', num_bottlenecks=3):
        super(WaveletBlock, self).__init__()
        
        # CSP主路径
        self.csp_branch = CSPBlock(
            in_channels=dims,
            out_channels=dims,
            num_bottlenecks=num_bottlenecks,
            expansion_factor=0.5
        ).double()
        
        # 小波变换处理分支
        self.wavelet_branch = WaveletUnit(
            in_channels=dims,
            out_channels=dims,
            wave=wave,
            mode=mode
        ).double()
        
        # 融合模块
        self.fusion = nn.Sequential(
            nn.Conv2d(dims, dims, 1),
            nn.BatchNorm2d(dims),
            nn.SiLU()
        ).double()
        
        self.norm = nn.BatchNorm2d(dims).double()
    
    def forward(self, x):
        x = x.double()
        # CSP处理
        out = self.csp_branch(x)
        
        # 小波处理（带残差连接）
        identity = out
        out = self.wavelet_branch(out)
        # 最终融合
        out = self.fusion(out+identity)
        out = self.norm(out)
        return out.float()

# 测试代码
if __name__ == '__main__':
    # 创建测试输入
    x = torch.randn(1, 64, 32, 32)
    
    # 创建WaveletMixer实例
    waveletblock = WaveletBlock(dims=64, num_bottlenecks=3)
    
    # 前向传播
    output = waveletblock(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)