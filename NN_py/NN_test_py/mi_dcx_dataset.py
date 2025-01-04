import torch
from pathlib import Path
from torch.utils.data import Dataset
import cv2
import numpy as np

# muti input version
def mi_txt2dataseq(loadfile, time_step=20):
    # 读取txt文件
    raw = np.genfromtxt(loadfile, dtype=None, encoding='utf-8', delimiter=',')
    
    dataseq = []
    # 按标签分组
    current_seq = []
    current_label = None
    
    for row in raw:
        img_orig_path, img_argu_path, img_dot_path, speed, position = row[0], row[1], row[2], int(row[3]), int(row[4])
        label = (speed, position)
        
        # 如果是新的标签组或当前序列已满
        if current_label != label or len(current_seq) == time_step:
            # 处理上一组数据(若存在)
            if current_seq:
                # 若不足time_step则用最后一张图片填充
                while len(current_seq) < time_step:
                    current_seq.append(current_seq[-1])
                dataseq.append((current_seq, current_label))
            # 开始新的序列    
            current_seq = [(img_orig_path, img_argu_path, img_dot_path)]
            current_label = label
        else:
            current_seq.append((img_orig_path, img_argu_path, img_dot_path))
    
    # 处理最后一组数据
    if current_seq:
        while len(current_seq) < time_step:
            current_seq.append(current_seq[-1])
        dataseq.append((current_seq, current_label))
        
    return dataseq

# # 测试示例
# if __name__ == '__main__':
#     # 测试函数
#     dataseq = mi_txt2dataseq('G:/NN_py/ssstest_merge/ssstest_merge_final.txt', time_step=20)
#     print(dataseq[-1])

# muti input version
class mi_dcxDataset(Dataset): 
    """
    *************多输入单触须数据集读取：多维标签(speed, angle)形式*************
    参数说明:
    - label_txt: 包含图像路径和标签的txt文件路径
    - time_step: 图像序列长度,默认为20
    - img_transform: 图像预处理函数,默认为None
    
    返回值:
    - img_tuple: (detected_seq, cluster_seq, distance_seq)，其中:
      detected_seq: 原始图像序列tensor, 形状为[time_step, channels, height, width]
      cluster_seq: 聚类图像序列tensor, 形状为[time_step, channels, height, width]
      distance_seq: 距离图像序列tensor, 形状为[time_step, channels, height, width]
    - label_seq: 标签tensor, 包含速度，和角度值，不进行归一化
    """
    def __init__(self, label_txt, time_step=20, img_transform=None):
        self.label_txt = label_txt
        self.img_transform = img_transform 
        self.time_step = time_step
        self.data = mi_txt2dataseq(self.label_txt, self.time_step)
        self.seq_augmentation = None # 默认不开启，不设置传入参数
        self.p_augment = 0
        
    def __len__(self):
        return len(self.data)
    
    def _set_seq_augmentation(self, seq_augmentation, p_augment):
        # 开启序列增强，仅training时使用
        ## seq_augmentation：增强函数 （func）, p_augment：增强概率 （float）
        self.seq_augmentation = seq_augmentation
        self.p_augment = p_augment
    
    def __getitem__(self, idx):
        ### 获取标签并归一化处理
        speed_label, angle_label = self.data[idx][-1]
        label = torch.tensor([float(speed_label), (float(angle_label)/4032*2*np.pi)]) # 不归一化
        
        if self.seq_augmentation:
            # 增强随机dice
            seq_aug_dice = np.random.rand()
    
        ### 加载图像序列
        detected_seq = []
        cluster_seq = []
        distance_seq = []
        
        for img_detected_path, img_cluster_path, distance in self.data[idx][0]:
            # 检查图像文件是否存在
            if not (Path(img_detected_path).exists() and Path(img_cluster_path).exists()):
                raise FileNotFoundError(f"Image files do not exist: {img_detected_path} or {img_cluster_path}")
            
            # 读取两种图像
            img_detected = cv2.imread(str(img_detected_path), cv2.IMREAD_COLOR)
            img_cluster = cv2.imread(str(img_cluster_path), cv2.IMREAD_COLOR)
            
            # 检查图像是否成功读取
            if img_detected is None or img_cluster is None:
                raise ValueError(f"Unable to read images: {img_detected_path} or {img_cluster_path}")
            
            # 序列增强
            if self.seq_augmentation and seq_aug_dice > self.p_augment:
                img_detected = self.seq_augmentation(img_detected)

            # 图像预处理
            if self.img_transform:
                img_detected = self.img_transform(img_detected)
                img_cluster = self.img_transform(img_cluster)
            
            # 转换为tensor
            img_detected = torch.from_numpy(img_detected).float()
            img_cluster = torch.from_numpy(img_cluster).float()
            distance = torch.tensor([distance]).float()
            
            detected_seq.append(img_detected)
            cluster_seq.append(img_cluster)
            distance_seq.append(distance)
            
        # 堆叠序列
        detected_seq = torch.stack(detected_seq)    # [T, H, W, C]
        cluster_seq = torch.stack(cluster_seq)      # [T, H, W, C]
        distance_seq = torch.stack(distance_seq).unsqueeze(2)   # [T, 1, 1] （[T, l, C]）
        
        # 图像标准格式是[H, W, C]
        # 重要：需要调整维度顺序为 [T, C, H, W]
        detected_seq = detected_seq.permute(0, 3, 1, 2)
        cluster_seq = cluster_seq.permute(0, 3, 1, 2)
        distance_seq = distance_seq.permute(0, 2, 1) # [T, C, l]
    
        return (detected_seq, cluster_seq, distance_seq), label

# # 测试示例
# if __name__ == '__main__':
#     # 图像预处理函数
#     def img_transform(img):
#         img = cv2.resize(img, (64, 64))
#         img = img / 255.0  # 图像归一化
#         return img
        
#     # 创建数据集实例
#     dataset = mi_dcxDataset(
#         label_txt='G:/NN_py/ssstest_merge/ssstest_merge_final.txt',
#         time_step=20,
#         img_transform=img_transform
#     )
    
#     # 测试数据集长度
#     print(f"数据集长度: {len(dataset)}")
    
#     # 获取一个样本
#     (detected_seq, cluster_seq, distance_seq), label = dataset[-1]
    
#     # 打印数据形状
#     print(f"\n检测图像序列形状: {detected_seq.shape}")  # [T, C, H, W]
#     print(f"聚类图像序列形状: {cluster_seq.shape}")  # [T, C, H, W] 
#     print(f"距离序列形状: {distance_seq.shape}")  # [T, 1]
#     print(f"标签形状: {label.shape}")  # [2]
    
#     # 检查数据类型
#     print(f"\n检测图像序列类型: {detected_seq.dtype}")
#     print(f"聚类图像序列类型: {cluster_seq.dtype}")
#     print(f"距离序列类型: {distance_seq.dtype}")
#     print(f"标签类型: {label.dtype}")
    
#     # 打印标签值
#     print(f"\n速度标签: {label[0]:.2f}")
#     print(f"角度标签: {label[1]:.2f} rad")
    
#     # 可视化第一帧图像
#     detected_img = detected_seq[0].permute(1, 2, 0).numpy()  # [H, W, C]
#     cluster_img = cluster_seq[0].permute(1, 2, 0).numpy()  # [H, W, C]
    
#     cv2.imshow('Detected Image', detected_img)
#     cv2.imshow('Cluster Image', cluster_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

