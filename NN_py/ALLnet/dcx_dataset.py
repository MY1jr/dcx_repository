import torch
from pathlib import Path
from torch.utils.data import Dataset
import cv2
import numpy as np

# muti input version
def mi_txt2data(loadfile):
    # 读取txt文件
    raw = np.genfromtxt(loadfile, dtype=None, encoding='utf-8', delimiter=',')
    
    data = []
    
    for row in raw:
        img_orig_path, img_clus_path, img_flow_path, speed, position = row[0], row[1], row[2], int(row[3]), int(row[4])
        label = (speed, position)
        img_paths = img_orig_path, img_clus_path, img_flow_path
        data.append((img_paths, label))
        
    return data

def img_transform(img):
    img = cv2.resize(img, (64, 64))
    img = img / 255.0  # 图像归一化
    return img

# muti input version
class mi_dcxDataset(Dataset): 

    def __init__(self, label_txt,img_transform=None):
        self.label_txt = label_txt
        self.img_transform = img_transform 
        self.data = mi_txt2data(self.label_txt)
        self.seq_augmentation = None # 默认不开启，不设置传入参数
        self.p_augment = 0
        
    def __len__(self):
        return len(self.data)
    
    def _set_seq_augmentation(self, seq_augmentation, p_augment):
        self.seq_augmentation = seq_augmentation
        self.p_augment = p_augment
    
    def __getitem__(self, idx):
        ### 获取标签并归一化处理
        speed_label, angle_label = self.data[idx][-1]
        label = torch.tensor([float(speed_label), (float(angle_label)/4032*2*np.pi)]) # 不归一化
        
        if self.seq_augmentation:
            # 增强随机dice
            seq_aug_dice = np.random.rand()
    
        img_detected_path, img_cluster_path, img_flow_path = self.data[idx][0]
        # 检查图像文件是否存在
        if not (Path(img_detected_path).exists() and Path(img_cluster_path).exists() and Path(img_flow_path).exists()):
            raise FileNotFoundError(f"Image files do not exist: {img_detected_path} or {img_cluster_path}")
        
        # 读取图像
        img_detected = cv2.imread(str(img_detected_path), cv2.IMREAD_COLOR)
        img_cluster = cv2.imread(str(img_cluster_path), cv2.IMREAD_COLOR)
        img_flow = cv2.imread(str(img_flow_path), cv2.IMREAD_COLOR)

        # 检查图像是否成功读取
        if img_detected is None or img_cluster is None or img_flow is None:
            raise ValueError(f"Unable to read images: {img_detected_path} or {img_cluster_path}")
        
        # 序列增强
        if self.seq_augmentation and seq_aug_dice > self.p_augment:
            img_detected = self.seq_augmentation(img_detected)

        # 图像预处理
        if self.img_transform:
            img_detected = self.img_transform(img_detected)
            img_cluster = self.img_transform(img_cluster)
            img_flow = self.img_transform(img_flow)
        
        # 转换为tensor
        img_detected = torch.from_numpy(img_detected).float()
        img_cluster = torch.from_numpy(img_cluster).float()
        img_flow = torch.from_numpy(img_flow).float()
            
        
        # 图像标准格式是[H, W, C]
        img_detected = img_detected.permute(2, 0, 1)
        img_cluster = img_cluster.permute(2, 0, 1)
        img_flow = img_flow.permute(2, 0, 1) 
    
        return img_detected, img_cluster, img_flow, label

# 测试示例
if __name__ == '__main__':
    # 创建数据集实例
    dataset = mi_dcxDataset(
        label_txt='g:/dcx_repository/NN_py/dcx_mini/dcx_mini_merge.txt',
        img_transform=img_transform
    )
    
    # 测试数据集长度
    print(f"数据集长度: {len(dataset)}")
    
    # 获取一个样本
    (detected_seq, cluster_seq, distance_seq), label = dataset[-1]
    
    # 打印数据形状
    print(f"\n检测图像序列形状: {detected_seq.shape}")
    print(f"聚类图像序列形状: {cluster_seq.shape}")
    print(f"距离序列形状: {distance_seq.shape}")
    print(f"标签形状: {label.shape}")
    
    # 检查数据类型
    print(f"\n检测图像序列类型: {detected_seq.dtype}")
    print(f"聚类图像序列类型: {cluster_seq.dtype}")
    print(f"距离序列类型: {distance_seq.dtype}")
    print(f"标签类型: {label.dtype}")
    
    # 打印标签值
    print(f"\n速度标签: {label[0]:.2f}")
    print(f"角度标签: {label[1]:.2f} rad")
    
    # 可视化第一帧图像
    detected_img = detected_seq.permute(1, 2, 0).numpy()  # 修正维度转换 [C,H,W] -> [H,W,C]
    cluster_img = cluster_seq.permute(1, 2, 0).numpy()
    flow_img = distance_seq.permute(1, 2, 0).numpy()

    # 确保图像是uint8格式且值在0-255范围内
    detected_img = (detected_img * 255).astype(np.uint8)
    cluster_img = (cluster_img * 255).astype(np.uint8)
    flow_img = (flow_img * 255).astype(np.uint8)

    # 显示图像
    cv2.imshow('Detected Image', detected_img)
    cv2.imshow('Cluster Image', cluster_img)
    cv2.imshow('Flow Image', flow_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


