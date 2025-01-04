import cv2
import numpy as np
from sklearn.cluster import KMeans
import pickle

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
        
    

    def detect_circles(self, img):
        if self.kmeans is None:
            raise ValueError("Model weights not found")
            
        # 转换为灰度图！
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
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
        cv2.circle(image, (int(x1), int(y1)), 1, (255, 0, 0), 1)
        cv2.circle(image, (int(x2), int(y2)), 1, (255, 0, 0), 1)
        # 绘制内圆（红色）##
        cv2.circle(image, (int(x1), int(y1)), int(r1), (0, 255, 0), 1)
        # 绘制外圆（绿色）##
        cv2.circle(image, (int(x2), int(y2)), int(r2), (0, 255, 0), 1)
    # 返回修改后的图片
    return image


class CircleAugmentation:
    def __init__(self):
        self.detector = KmeansCircleDetector(n_clusters=4)
        self.detector.load_model('G:/NN_py/circle_detection/kmeans_model_4.pkl')
    
    def __call__(self, img): # 实例作为函数调用
        img = visualize_circle_detection(img, self.detector)
        return img


if __name__ == '__main__':
    # 测试
    image = cv2.imread('G:/NN_py/dcx_mini/600/320/15.png')
    circle_augmentation = CircleAugmentation()
    image = circle_augmentation(image)
    cv2.imshow('Detected Circles', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
