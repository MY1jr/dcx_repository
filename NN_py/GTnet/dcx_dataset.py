import torch
from pathlib import Path
from torch.utils.data import Dataset
import cv2
import numpy as np
from utils.dcx_txt2data import txt2dataseq

class dcxDataset(Dataset): 
    """
    *************单触须数据集读取：多维标签(speed, angle)形式*************
    参数说明:
    - label_txt: 包含图像路径和标签的txt文件路径
    - time_step: 图像序列长度,默认为5
    - img_transform: 图像预处理函数,默认为None
    
    返回值:
    - img_seq: 图像序列tensor, 形状为[time_step, channels, height, width]
    - label_seq: 标签tensor, 包含速度，和角度值，不进行归一化
    """
    def __init__(self, label_txt, time_step=5, img_transform=None):
        self.label_txt = label_txt
        self.img_transform = img_transform 
        self.time_step = time_step
        self.data = txt2dataseq(self.label_txt, self.time_step)
        self.seq_augmentation = None # 默认不开启，不设置传入参数
        
    def __len__(self):
        return len(self.data)
    
    def _set_seq_augmentation(self, seq_augmentation):
        # 开启序列增强，仅training时使用
        ## seq_augmentation：增强函数 （func）
        self.seq_augmentation = seq_augmentation
    
    def __getitem__(self, idx):
        ### 获取标签并归一化处理
        speed_label, angle_label = self.data[idx][1]
        label = torch.tensor([float(speed_label), (float(angle_label)/4032*2*np.pi)]) # 不归一化
        
        if self.seq_augmentation:
            # 增强随机dice
            seq_aug_dice = np.random.rand()
    
        ### 加载图像序列
        img_seq = []
        for img_path in self.data[idx][0]:
            
            ## 检查图像文件是否存在
            if not Path(img_path).exists():
                raise FileNotFoundError(f"图像文件不存在: {img_path}")
            ##
            
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

            ## 检查图像是否成功读取
            if img is None:
                raise ValueError(f"无法读取图像: {img_path}")
            ##
            
            if self.seq_augmentation:
                img = self.seq_augmentation(img, seq_aug_dice)

            if self.img_transform:
                img = self.img_transform(img)
                
            img = torch.from_numpy(img).float()
            img_seq.append(img)
            
        img_seq = torch.stack(img_seq)    # [T, H, W, C] 图像标准格式是[H, W, C]
        # 重要：调整维度顺序为 [T, C, H, W]
        img_seq = img_seq.permute(0, 3, 1, 2)
    
        return img_seq, label
    
