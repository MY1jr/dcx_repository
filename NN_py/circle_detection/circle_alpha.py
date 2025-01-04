import cv2
import numpy as np
from sklearn.cluster import KMeans
import pickle
from pathlib import Path

# 使用K-means进行聚类, 可增量训练
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

        # #########################################
        # # 显示灰度图
        # dimg = cv2.resize(gray, (480,480))
        # cv2.imshow('gray', dimg)
        # #########################################
        
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
        
        # #########################################
        # # 显示对比度提升后图像
        # dimg = cv2.resize(enhanced_img, (480,480))
        # cv2.imshow('enhanced_img', dimg)
        # #########################################  

        gray = enhanced_img
        return gray
    
    def partial_fit(self, image_batch):
        """批量训练模型
        
        Args:
            image_batch: 一批图片列表
        """
        # 收集当前批次的像素数据
        batch_pixels = []
        for img in image_batch:
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # ### 训练数据不一定需要图像增强 ### 
            # gray = self.img_transform(gray)
            pixels = gray.reshape(-1, 1)
            batch_pixels.append(pixels)
        
        batch_pixels = np.vstack(batch_pixels)
        
        # 如果模型未初始化，进行第一次完整训练
        if not hasattr(self.kmeans, 'cluster_centers_'):
            self.kmeans.fit(batch_pixels)
        else:
            # 使用当前批次数据更新聚类中心
            # 获取当前批次数据的聚类结果
            labels = self.kmeans.predict(batch_pixels)
            # 更新聚类中心
            for i in range(self.kmeans.n_clusters):
                if np.sum(labels == i) > 0:
                    self.kmeans.cluster_centers_[i] = np.mean(
                        batch_pixels[labels == i], axis=0
                    )
    
    def fit_circle_least_squares(self, contour):
        # 将轮廓点转换为二维数组
        points = np.squeeze(contour)
        
        # 构建矩阵 A 和向量 b
        A = np.hstack((2 * points, np.ones((points.shape[0], 1))))
        b = np.sum(points**2, axis=1)
        
        # 使用最小二乘法求解
        c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # 提取圆心和半径
        center_x, center_y = c[0], c[1]
        radius = np.sqrt(c[2] + center_x**2 + center_y**2)
        
        return (center_x, center_y), radius
    
    
    def detect_circles(self, img):
        if self.kmeans is None:
            raise ValueError("请先使用 train_on_images 方法训练模型")
            
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
        
        # ########################################
        # # 显示每个簇的掩码
        # for i in range(self.kmeans.n_clusters):
        #     mask_class = np.zeros_like(gray)
        #     mask_class[segments == i] = 255
        #     showmask = mask_class.copy() 
        #     showmask = cv2.resize(showmask, (480,480))
        #     cv2.imshow(f'mask_class_{i}', showmask)
        # ########################################
             
        # 使用轮廓检测
        # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        # 找到最大的轮廓（外圆）
        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
            (x2, y2), r2 = cv2.minEnclosingCircle(max_contour) # 最小外接圆
            # (x2, y2), r2 = self.fit_circle_least_squares(max_contour) # 最小二乘法
            
            # 找内圆
            mask_inner = np.zeros_like(gray)
            mask_inner[segments == self.inner_cluster] = 255  # 假设标签3对应内圈区域
            contours_inner, _ = cv2.findContours(mask_inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 找到最大的轮廓（内圆）
            if len(contours_inner) > 0:
                inner_contour = max(contours_inner, key=cv2.contourArea)
                (x1, y1), r1 = cv2.minEnclosingCircle(inner_contour) # 最小外接圆
                # (x1, y1), r1 = self.fit_circle_least_squares(inner_contour) # 最小二乘法
                return [(x1, y1, r1, x2, y2, r2)]
        
        return []
    
    def save_model(self, filepath):
        """保存训练好的K-means模型"""
        if self.kmeans is None:
            raise ValueError("没有训练好的模型可以保存")
        with open(filepath, 'wb') as f:
            pickle.dump(self.kmeans, f)
            
    def load_model(self, filepath):
        """加载已保存的K-means模型"""
        with open(filepath, 'rb') as f:
            self.kmeans = pickle.load(f)

def train_detector(txt_path, batch_size=10, detector=None, n_clusters=4):
    """批量训练同心圆检测器
    
    Args:
        txt_path: 包含训练图片路径的txt文件
        batch_size: 每批处理的图片数量
        detector: 已有的检测器实例(用于增量学习),默认为None则创建新实例
    
    Returns:
        训练好的检测器实例
    """
    if detector is None:
        detector = KmeansCircleDetector(n_clusters=n_clusters)
    
    # 读取所有图片路径
    image_paths = []
    with open(txt_path, 'r') as f:
        image_paths = [line.strip().split(',')[0] for line in f]
    
    # 批量处理图片
    current_batch = []
    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is not None:
            current_batch.append(img)
            
        # 当收集够一个批次或是最后一批时，进行训练
        if len(current_batch) >= batch_size or i == len(image_paths) - 1:
            if current_batch:  # 确保批次不为空
                print(f"\r训练批次 {i//batch_size + 1}/{len(image_paths)//batch_size + 1}", end="")
                detector.partial_fit(current_batch)
                current_batch = []  # 清空当前批次
    
    return detector

def detect_and_visualize(image_path, detector):
    """检测并可视化同心圆
    
    Args:
        image_path: 待检测图片的路径
        detector: 训练好的检测器实例
    """
    # 读取并检测图片
    test_image = cv2.imread(image_path)
    original_image = test_image.copy()
    concentric_circles = detector.detect_circles(test_image)

    # 在原图上绘制检测结果
    for circles in concentric_circles:
        x1, y1, r1, x2, y2, r2 = circles
        # 绘制圆心
        cv2.circle(test_image, (int(x1), int(y1)), 1, (0, 0, 255), 1)
        cv2.circle(test_image, (int(x2), int(y2)), 1, (0, 255, 0), 1)
        # 绘制内圆（红色）
        cv2.circle(test_image, (int(x1), int(y1)), int(r1), (0, 0, 255), 1)
        # 绘制外圆（绿色）
        cv2.circle(test_image, (int(x2), int(y2)), int(r2), (0, 255, 0), 1)

    # 显示结果
    original_image = cv2.resize(original_image, (480,480))
    test_image = cv2.resize(test_image, (480,480))
    cv2.imshow('original_image', original_image)
    cv2.imshow('Detected Circles', test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def detect_and_visualize_multiple(image_path, detector, circle_distribution=(5,5)):
    """检测并可视化多组同心圆
    
    Args:
        image_path: 待检测图片的路径
        detector: 训练好的检测器实例
    """
    # 读取原图
    original_image = cv2.imread(image_path)
    result_image = original_image.copy()
    
    # 获取图像尺寸
    height, width = original_image.shape[:2]    
    
    # 计算单个圆组的大致尺寸
    circle_height = height // circle_distribution[0]  # 假设是5x5的网格
    circle_width = width // circle_distribution[1]
    
    # 遍历每个圆组位置
    for i in range(circle_distribution[0]):
        for j in range(circle_distribution[1]):
            # 计算当前圆组的区域
            top = i * circle_height
            bottom = (i + 1) * circle_height
            left = j * circle_width
            right = (j + 1) * circle_width
            
            # 提取当前圆组图像
            circle_group = original_image[top:bottom, left:right]
            # 调整尺寸(不调整，按照切割尺寸，避免聚类重复像素点)
            # circle_group = cv2.resize(circle_group, (100,100))
            
            # # 保存切割的图像
            # circle_group = cv2.resize(circle_group, (480,480))
            # cv2.imwrite(f'G:/NN_py/circle_detection/circle_group_{i}_{j}.png', circle_group)
            
            # 对当前圆组进行检测
            concentric_circles = detector.detect_circles(circle_group)
            
            # 在结果图像上绘制检测结果
            for circles in concentric_circles:
                x1, y1, r1, x2, y2, r2 = circles
                # 调整坐标到原图位置
                x1, y1 = int(x1 + left), int(y1 + top)
                x2, y2 = int(x2 + left), int(y2 + top)
                
                # 绘制圆心和圆
                cv2.circle(result_image, (x1, y1), 1, (0, 0, 255), 1)  # 内圆心
                cv2.circle(result_image, (x2, y2), 1, (0, 255, 0), 1)  # 外圆心
                cv2.circle(result_image, (x1, y1), int(r1), (0, 0, 255), 1)  # 内圆
                cv2.circle(result_image, (x2, y2), int(r2), (0, 255, 0), 1)  # 外圆
    
    # 显示结果
    original_image = cv2.resize(original_image, (480,480))
    result_image = cv2.resize(result_image, (480,480))
    cv2.imshow('Original Image', original_image)
    cv2.imshow('Detected Circles', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_and_visualize_video(video_path, detector, output_path=None):
    """检测视频中的同心圆
    
    Args:
        video_path: 输入视频的路径
        detector: 训练好的检测器实例
        output_path: 输出视频的路径（可选）
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('无法打开视频文件')
        return
    
    # 获取视频参数
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # print(f'FPS: {fps}')

    # 如果指定了输出路径，创建视频写入器
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 复制原始帧用于绘制
        result_frame = frame.copy()
        
        # 检测同心圆
        concentric_circles = detector.detect_circles(frame)
        
        # 在帧上绘制检测结果
        for circles in concentric_circles:
            x1, y1, r1, x2, y2, r2 = circles
            # 绘制圆心
            cv2.circle(result_frame, (int(x1), int(y1)), 1, (0, 0, 255), 1)
            cv2.circle(result_frame, (int(x2), int(y2)), 1, (0, 255, 0), 1)
            # 绘制内圆（红色）
            cv2.circle(result_frame, (int(x1), int(y1)), int(r1), (0, 0, 255), 1)
            # 绘制外圆（绿色）
            cv2.circle(result_frame, (int(x2), int(y2)), int(r2), (0, 255, 0), 1)
        
        # 显示结果 (单独变量存储，避免resize影响检测结果，避免影响视频打开)
        display_frame = cv2.resize(result_frame, (480,480))
        cv2.imshow('Video Detection', display_frame)
        
        # 写入输出视频
        if writer:
            writer.write(result_frame)
            
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    if writer:
        writer.release()
        print(f'视频已保存到 {output_path}')
    cv2.destroyAllWindows()

def cluster_and_save_dataset(txt_path, output_dir, detector, root_dir):
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

def detect_and_save_dataset(txt_path, output_dir, detector, root_dir):
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
            continue
            
        # 复制原图用于绘制
        result_img = img.copy()
        
        # 检测同心圆
        concentric_circles = detector.detect_circles(img)
        
        # 在图片上绘制检测结果
        for circles in concentric_circles:
            x1, y1, r1, x2, y2, r2 = circles
            # 绘制圆心
            cv2.circle(result_img, (int(x1), int(y1)), 1, (0, 0, 255), 1)
            cv2.circle(result_img, (int(x2), int(y2)), 1, (0, 255, 0), 1)
            # 绘制内圆（红色）
            cv2.circle(result_img, (int(x1), int(y1)), int(r1), (0, 0, 255), 1)
            # 绘制外圆（绿色）
            cv2.circle(result_img, (int(x2), int(y2)), int(r2), (0, 255, 0), 1)
        
        # 保持原目录结构,替换根目录
        img_path = Path(img_path)
        relative_path = img_path.relative_to(root_dir)
        output_path = output_dir / relative_path
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存检测结果
        cv2.imwrite(str(output_path), result_img)
        print(f'\r处理进度: {i+1}/{len(image_paths)}', end='')
    print('\n数据集检测圆完成')

def calculate_distance_dataset(txt_path, output_dir, detector):
    """检测数据集中的同心圆并保存圆心距离txt
    Args:
        txt_path: 数据集图片路径的txt文件
        output_dir: 检测结果保存目录
        detector: 训练好的检测器实例
    """
    # 读取所有图片路径
    with open(txt_path, 'r') as f:
        image_paths = [line.strip().split(',')[0] for line in f]
        
    # 创建输出txt文件
    output_txt = output_dir / 'circle_distances.txt'
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    
    # 处理每张图片
    with open(output_txt, 'w') as f:
        for i, img_path in enumerate(image_paths):
            # 读取图片
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            # 检测同心圆
            concentric_circles = detector.detect_circles(img)
            
            # 计算并保存圆心距离
            if concentric_circles:
                for circles in concentric_circles:
                    x1, y1, r1, x2, y2, r2 = circles
                    # 计算两圆心之间的欧氏距离
                    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    # 计算两圆心之间的曼式距离
                    # distance = np.abs(x2-x1) + np.abs(y2-y1)
                    # 写入txt文件
                    f.write(f'{distance}\n')
            else:
                # 如果没有检测到圆,写入-1
                f.write('-1\n')
            
            print(f'\r处理进度: {i+1}/{len(image_paths)}', end='')
    
    print(f'\n圆心距离已保存到 {output_txt}')

def detect_dataset_cluster_distance(txt_path, output_detect_dir, output_cluster_dir, output_distance_dir, detector, root_dir):
    """检测数据集中的同心圆并保存结果,保持原数据集的目录结构，同时保存圆心距离和聚类掩码
    
    Args:
        txt_path: 数据集图片路径的txt文件
        output_dir: 检测结果保存目录
        detector: 训练好的检测器实例
        root_dir: 原始数据集根目录
    """
    # 创建子目录
    detect_dir = output_detect_dir   # 存放检测结果
    cluster_dir = output_cluster_dir  # 存放聚类掩码
    
    # 读取所有图片路径
    with open(txt_path, 'r') as f:
        image_paths = [line.strip().split(',')[0] for line in f]
        
    # 创建输出txt文件
    output_txt = output_distance_dir / 'circle_distances.txt'
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    
    # 处理每张图片
    with open(output_txt, 'w') as f:
        for i, img_path in enumerate(image_paths):
            # 读取图片
            img = cv2.imread(str(img_path))
            if img is None:
                print(f'无法读取图片: {img_path}')
                continue
                
            # 复制原图用于绘制检测结果
            result_img = img.copy()
            
            # 转换为灰度图并增强
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = detector.img_transform(gray)
            
            # 使用训练好的模型进行聚类预测
            pixels = gray.reshape((-1, 1))
            segments = detector.kmeans.predict(pixels).reshape(gray.shape)
            
            # 创建聚类掩码
            mask = np.zeros_like(gray)
            mask[segments == 2] = 255  # 提取2号簇
            
            # 检测同心圆
            concentric_circles = detector.detect_circles(img)
            
            # 计算并保存圆心距离
            if concentric_circles:
                for circles in concentric_circles:
                    x1, y1, r1, x2, y2, r2 = circles
                    # 计算两圆心之间的欧氏距离
                    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    f.write(f'{distance}\n')
                    
                    # 在结果图上绘制检测结果
                    cv2.circle(result_img, (int(x1), int(y1)), 1, (0, 0, 255), 1)  # 内圆心
                    cv2.circle(result_img, (int(x2), int(y2)), 1, (0, 255, 0), 1)  # 外圆心
                    cv2.circle(result_img, (int(x1), int(y1)), int(r1), (0, 0, 255), 1)  # 内圆
                    cv2.circle(result_img, (int(x2), int(y2)), int(r2), (0, 255, 0), 1)  # 外圆
            else:
                f.write('-1\n')
            
            # 保持原目录结构,替换根目录
            img_path = Path(img_path)
            relative_path = img_path.relative_to(root_dir)
            
            # 设置检测结果和聚类掩码的输出路径
            detect_path = detect_dir / relative_path
            cluster_path = cluster_dir / relative_path
            
            # 确保输出目录存在
            detect_path.parent.mkdir(parents=True, exist_ok=True)
            cluster_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存检测结果和聚类掩码
            cv2.imwrite(str(detect_path), result_img)
            cv2.imwrite(str(cluster_path), mask)
            
            print(f'\r处理进度: {i+1}/{len(image_paths)}', end='')
            
    print('\n数据集处理完成')
    print(f'检测结果保存在: {detect_dir}')
    print(f'聚类掩码保存在: {cluster_dir}')
    print(f'圆心距离保存在: {output_txt}')

def labeling_dataset(dataset_txt_path,detector,output_txt_path,std_h=64,std_w=64):
    """检测数据集中的同心圆并保存结果,保持原数据集的目录结构
    
    Args:
        dataset_txt_path: 数据集图片路径的txt文件
        detector: 训练好的检测器实例
        output_txt_path: 保存检测结果的txt文件路径
    """
    # 读取所有图片路径
    with open(dataset_txt_path, 'r') as f:
        image_paths = [line.strip().split(',')[0] for line in f]
    
    # 检查输出文本文件是否存在，不存在则创建
    output_txt_path = Path(output_txt_path)
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_txt_path.exists():
        output_txt_path.touch()
        
    # 打开输出文本文件
    with output_txt_path.open('w') as output_file:
        # 处理每张图片
        for i, img_path in enumerate(image_paths):
            # 读取图片
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # 图像缩放成std_w, std_h，标签也是std_w, std_h
            img = cv2.resize(img, (std_w, std_h))
                
            # 检测同心圆
            concentric_circles = detector.detect_circles(img)
            
            # 写入检测结果到文本文件
            for circles in concentric_circles:
                x1, y1, r1, x2, y2, r2 = circles
                output_file.write(f"{img_path},{x1:.2f},{y1:.2f},{r1:.2f},{x2:.2f},{y2:.2f},{r2:.2f}\n")
            
            print(f'\r处理进度: {i+1}/{len(image_paths)}', end='')
    print('\n数据集打标签完成')
    
if __name__ == "__main__":
    ### n_clusters设定；outer_cluster, inner_cluster对应；model.pkl 命名；###
    
    # 模式选择
    mode = 'labeling'
    
    ##############################################################################
    if mode == 'train':
        txt_path = Path('G:/NN_py/dcx_mini/dcx_mini.txt') #"G:/NN_py/ssstest/ssstest.txt" 
        detector = train_detector(txt_path, batch_size=1000, n_clusters=4)  # 设置批次大小
        # 保存模型
        detector.save_model('G:/NN_py/circle_detection/kmeans_model.pkl')
        print('模型已保存')

    elif mode == 'detect':
        # 加载已保存的模型
        detector = KmeansCircleDetector(n_clusters=4)
        detector.load_model('G:/NN_py/circle_detection/kmeans_model_4.pkl')
        print('模型已加载')
        # 检测单张图片
        test_image_path = 'G:/NN_py/dcx_mini/600/320/15.png' 
        #'G:/NN_py/circle_detection/circle_group_2_2.png'
        detect_and_visualize(test_image_path, detector)
        
    elif mode == 'implement':
        # 加载已有模型
        detector = KmeansCircleDetector(n_clusters=4)
        detector.load_model('G:/NN_py/circle_detection/kmeans_model.pkl')
        print('已加载基础模型')
        # 继续训练新数据
        txt_path = Path('G:/NN_py/dcx/dcx.txt')  # 新数据的路径
        detector = train_detector(txt_path, batch_size=1000, detector=detector)
        # 保存更新后的模型
        detector.save_model('G:/NN_py/circle_detection/kmeans_model.pkl')
        print('\n更新后的模型已保存')

    elif mode == 'multiple':
        # 加载已有模型
        detector = KmeansCircleDetector(n_clusters=4)
        detector.load_model('G:/NN_py/circle_detection/kmeans_model_4.pkl')
        print('已加载基础模型')
        # 检测多组同心圆
        test_image_path = 'G:/NN_py/circle_detection/muticx.jpg'
        detect_and_visualize_multiple(test_image_path, detector, circle_distribution=(5,5))
    
    elif mode == 'video':
        # 加载已有模型
        detector = KmeansCircleDetector(n_clusters=4)
        detector.load_model('G:/NN_py/circle_detection/kmeans_model_4.pkl')
        print('已加载基础模型')
        # 检测视频中的同心圆
        video_output_path = Path('G:/NN_py/circle_detection/dcx_video.mp4')
        video_path = Path('G:/NN_py/dcx_video.mp4')
        detect_and_visualize_video(str(video_path), detector, output_path=str(video_output_path))
    
    
    
    ##############################################################################
    elif mode == 'cluster':
        # 聚类数据集,生成内圆簇二值化图数据集
        # 加载已有模型
        detector = KmeansCircleDetector(n_clusters=4)
        detector.load_model('G:/NN_py/circle_detection/kmeans_model_4.pkl')
        print('已加载基础模型')
        
        # 读取数据集路径
        txt_path = Path('G:/NN_py/ssstest/ssstest.txt')
        output_dir = Path('G:/NN_py/ssstest_cluster/')
        root_dir = Path('G:/NN_py/ssstest/')
        
        # 检测并保存数据集
        cluster_and_save_dataset(txt_path, output_dir, detector, root_dir)
    
    elif mode == 'dataset':
        # 检测数据集中的同心圆, 生成检测后的数据集
        # 加载已有模型
        detector = KmeansCircleDetector(n_clusters=4)
        detector.load_model('G:/NN_py/circle_detection/kmeans_model_4.pkl')
        print('已加载基础模型')
        
        # 读取数据集路径
        txt_path = Path('G:/NN_py/ssstest/ssstest.txt')
        output_dir = Path('G:/NN_py/ssstest_detected/')
        root_dir = Path('G:/NN_py/ssstest/')
        
        # 检测并保存数据集
        detect_and_save_dataset(txt_path, output_dir, detector, root_dir)
    
    elif mode == 'distance':
        # 检测数据集中的同心圆并保存圆心距离txt
        # 加载已有模型
        detector = KmeansCircleDetector(n_clusters=4)
        detector.load_model('G:/NN_py/circle_detection/kmeans_model_4.pkl')
        print('已加载基础模型')
        
        # 读取数据集路径
        txt_path = Path('G:/NN_py/ssstest/ssstest.txt')
        output_dir = Path('G:/NN_py/ssstest_distance/')
    
        # 检测并保存数据集
        calculate_distance_dataset(txt_path, output_dir, detector)
    
    elif mode == '3func': 
        # 三合一
        # 加载已有模型
        detector = KmeansCircleDetector(n_clusters=4)
        detector.load_model('G:/NN_py/circle_detection/kmeans_model_4.pkl')
        print('已加载基础模型')
        
        # 读取数据集路径
        txt_path = Path('G:/NN_py/dcx_mini/dcx_mini.txt')
        output_detect_dir = Path('G:/NN_py/dcx_mini_detected/')
        output_cluster_dir = Path('G:/NN_py/dcx_mini_cluster/')
        output_distance_dir = Path('G:/NN_py/dcx_mini_distance/')
        root_dir = Path('G:/NN_py/dcx_mini/')
    
        # 检测并保存数据集
        detect_dataset_cluster_distance(txt_path, output_detect_dir, output_cluster_dir, output_distance_dir, detector, root_dir)

    elif mode == 'labeling':
        # 检测同心圆，打标签txt：图片地址、内圆心x、内圆心y、内圆半径、外圆心x、外圆心y、外圆半径
        # 加载已有模型
        detector = KmeansCircleDetector(n_clusters=4)
        detector.load_model('G:/NN_py/circle_detection/kmeans_model_4.pkl')
        print('已加载基础模型')
        
        # 读取数据集路径
        dataset_txt_path = Path('G:/NN_py/dcx_mini/dcx_mini.txt')
        output_txt_path = Path('G:/NN_py/dcx_mini/circles_label.txt')
        
        # 检测并保存标签
        labeling_dataset(dataset_txt_path, detector, output_txt_path, std_h=64, std_w=64)