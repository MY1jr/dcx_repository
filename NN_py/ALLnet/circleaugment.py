import cv2
import numpy as np
from sklearn.cluster import KMeans
import pickle
from pathlib import Path

### 应用权重，无训练设置 ###
# 使用K-means进行聚类，圆检测
class KmeansCircleDetector:
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        # 人为对应外圆和内圆的簇
        if self.n_clusters == 4:
            self.outer_cluster = 3 # 外圆簇 
            self.inner_cluster = 2 # 内圆簇 
        elif self.n_clusters == 3:
            self.outer_cluster = 2 # 外圆簇 
            self.inner_cluster = 1 # 内圆簇  
        
    def img_transform(self, gray, num_levels=4):  ### 灰度图增强

        # 1. 创建多尺度金字塔 (从大到小)
        gaussian_pyramid = [gray]  # 第0层是原始图像
        for i in range(num_levels):  # 根据参数创建金字塔
            down = cv2.pyrDown(gaussian_pyramid[i])
            gaussian_pyramid.append(down)
        
        # 2. 对每一层进行对比度增强
        enhanced_pyramid = []
        for layer in gaussian_pyramid:
            # 限制对比度直方图均衡化
            clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(1,1))
            enhanced_layer = clahe.apply(layer)
            enhanced_pyramid.append(enhanced_layer)
        
        # 3. 金字塔重建 (从小到大)
        enhanced_img = enhanced_pyramid[num_levels]  # 从最小层开始
        for i in range(num_levels-1, -1, -1):  # 从最小层到最大的层
            # 上采样
            enhanced_img = cv2.pyrUp(enhanced_img) 
            # 确保尺寸匹配（因为pyrDown()和pyrUp()在处理奇数尺寸的图像时，有舍入问题）
            if enhanced_img.shape != enhanced_pyramid[i].shape:
                enhanced_img = cv2.resize(enhanced_img, 
                                        (enhanced_pyramid[i].shape[1], enhanced_pyramid[i].shape[0]))
            # 加权融合
            enhanced_img = cv2.addWeighted(enhanced_img, 0.5, enhanced_pyramid[i], 0.5, 0)
        
        gray = enhanced_img
        return gray
    

    def detect_circles(self, img):
        if self.kmeans is None:
            raise ValueError("Model weights not found")
            
        # 转换为灰度图！
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 图像增强
        gray = self.img_transform(gray)
        
        # 使用训练好的模型进行预测
        pixels = gray.reshape((-1, 1))
        segments = self.kmeans.predict(pixels).reshape(gray.shape)
        
        # 提取内圆和外圆
        mask = np.zeros_like(gray)
        mask[segments == self.outer_cluster] = 255  # 假设标签2对应外圆

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        # 找到最大的轮廓（外圆）
        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
            (x2, y2), r2 = cv2.minEnclosingCircle(max_contour) # cv2.minEnclosingCircle 最小外接圆
            
            # 找内圆
            mask_inner = np.zeros_like(gray)
            mask_inner[segments == self.inner_cluster] = 255  # 假设标签3对应内圈区域
            contours_inner, _ = cv2.findContours(mask_inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 找到最大的轮廓（内圆）
            if len(contours_inner) > 0:
                inner_contour = max(contours_inner, key=cv2.contourArea)
                (x1, y1), r1 = cv2.minEnclosingCircle(inner_contour)
                return [(x1, y1, r1, x2, y2, r2)]
        
        return []
       
    def load_model(self, filepath):
        """加载已保存的K-means模型"""
        with open(filepath, 'rb') as f:
            self.kmeans = pickle.load(f)

def visualize_circle_detection(image, detector):
    # 读取并检测图片
    concentric_circles = detector.detect_circles(image)
    # 在原图上绘制检测结果
    for circles in concentric_circles:
        x1, y1, r1, x2, y2, r2 = circles
        # 绘制圆心
        cv2.circle(image, (int(x1), int(y1)), 1, (0, 0, 255), 1)
        cv2.circle(image, (int(x2), int(y2)), 1, (0, 0, 255), 1)
        # 绘制内圆（绿色）##
        cv2.circle(image, (int(x1), int(y1)), int(r1), (0, 255, 0), 1)
        # 绘制外圆（绿色）##
        cv2.circle(image, (int(x2), int(y2)), int(r2), (0, 255, 0), 1)
    # 返回修改后的图片
    return image


class CircleAugmentation:
    def __init__(self):
        self.detector = KmeansCircleDetector(n_clusters=4)
        
        # 获取项目根目录
        project_root = Path(__file__).parent.parent # NN_py
        
        # 构建模型路径
        model_path = project_root / 'circle_detection' / 'kmeans_model_4.pkl'
        
        # 加载模型
        self.detector.load_model(str(model_path))
        
    def __call__(self, img): # 实例作为函数调用
        img = visualize_circle_detection(img, self.detector)
        return img
    
def get_cluster_circle(img):
    # 创建检测器实例
    detector = KmeansCircleDetector(n_clusters=4)
    # 获取项目根目录
    project_root = Path(__file__).parent.parent # NN_py
    # 构建模型路径
    model_path = project_root / 'circle_detection' / 'kmeans_model_4.pkl'
    # 加载模型
    detector.load_model(str(model_path))
    # 检测并绘制圆
    img = visualize_circle_detection(img, detector)
    return img

def get_cluster_mask(img):
    # 获取内圆聚类簇的二值图像
    detector = KmeansCircleDetector(n_clusters=4)
    project_root = Path(__file__).parent.parent
    model_path = project_root / 'circle_detection' / 'kmeans_model_4.pkl'
    detector.load_model(str(model_path))
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 图像增强
    gray = detector.img_transform(gray)
    
    # 使用训练好的模型进行预测
    pixels = gray.reshape((-1, 1))
    segments = detector.kmeans.predict(pixels).reshape(gray.shape)
    
    # 创建掩码
    mask = np.zeros_like(gray)
    mask[segments == 2] = 255
    
    # 转换为三通道图像
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return mask_3ch


def cluster_and_save_dataset(root_dir, txt_path, output_dir, detector):
    """检测数据集中的同心圆并保存结果,保持原数据集的目录结构
    
    Args:
        txt_path: 数据集图片路径的txt文件
        output_dir: 检测结果保存目录
        detector: 训练好的检测器实例
    """
    # 读取所有图片路径
    with open(txt_path, 'r') as f:
        image_paths = [line.strip().split(',')[0] for line in f]
        
    # 处理每张图片
    for i, img_path in enumerate(image_paths):
        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            print(f'无法读取图片: {img_path}')
            continue
            
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 图像增强
        gray = detector.img_transform(gray)
        
        # 使用训练好的模型进行预测
        pixels = gray.reshape((-1, 1))
        segments = detector.kmeans.predict(pixels).reshape(gray.shape)
        
        # 提取2号簇
        mask = np.zeros_like(gray)
        mask[segments == 2] = 255
        
        # 转换为三通道图像
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # 保持原目录结构,替换根目录
        img_path = Path(img_path)
        relative_path = img_path.relative_to(root_dir)
        output_path = output_dir / relative_path
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存2号簇掩码
        cv2.imwrite(str(output_path), mask)
        print(f'\r处理进度: {i+1}/{len(image_paths)}', end='')
        
    print('\n数据集聚类完成')
    
    
    
if __name__ == '__main__': 
    # 聚类数据集,生成内圆簇二值化图数据集
    # 加载已有模型
    detector = KmeansCircleDetector(n_clusters=4)
    project_root = Path(__file__).parent.parent #
    model_path = project_root / 'circle_detection' / 'kmeans_model_4.pkl'
    detector.load_model(str(model_path))
    print('已加载基础模型')
    
    # 读取数据集路径
    root_dir = Path(project_root / 'dcx')
    txt_path = Path(project_root / 'dcx' / 'dcx.txt')
    output_dir = Path(project_root / 'dcx_cluster')
    cluster_and_save_dataset(root_dir, txt_path, output_dir, detector)

