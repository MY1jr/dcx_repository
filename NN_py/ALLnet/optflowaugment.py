import cv2
import numpy as np
from pathlib import Path

def rgb_optical_flow_2():
    # 读取两张图片
    frame1 = cv2.imread('g:/dcx_repository/NN_py/dcx/600/320/8.png')
    frame2 = cv2.imread('g:/dcx_repository/NN_py/dcx/600/320/9.png')
    
    # 检查图片是否正确读取
    if frame1 is None or frame2 is None:
        print("错误：无法读取图片文件") 
        return
    
    # 确保两张图片尺寸相同
    if frame1.shape != frame2.shape:
        print("错误：两张图片尺寸不一致")
        return
    
    # 转换为灰度图
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # 计算光流
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 
                                       pyr_scale=0.5,  # 金字塔上下层之间的尺度关系
                                       levels=3,       # 金字塔层数
                                       winsize=7,     # 均值窗口大小
                                       iterations=3,    # 每层金字塔的迭代次数
                                       poly_n=5,       # 多项式展开中的邻域大小
                                       poly_sigma=2,  # 高斯标准差
                                       flags=0)
    
    # 计算光流的大小和方向
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # 创建HSV图像用于可视化
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    hsv[..., 0] = angle * 180 / np.pi / 2  # 角度转换为色调
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # 归一化幅值
    
    # 转换回BGR颜色空间
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # 显示结果
    cv2.imshow('原始图片 1', frame1)
    cv2.imshow('原始图片 2', frame2)
    cv2.imshow('光流可视化', rgb)
    
    # 保存结果
    cv2.imwrite('optical_flow_result.jpg', rgb)
    
    # 等待按键并关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# def get_optical_flow_EMA(images, alpha=0.8):
#     """
#     使用EMA指数移动平均法计算光流
#     alpha: 平滑系数，越大表示越重视当前帧，越小表示越重视历史信息
#     """
#     if len(images) < 2:
#         print("错误：图像序列至少需要2张图片")
#         return None
        
#     flow_maps = []
#     seq_len = len(images)
    
#     # 使用第一帧
#     prev_frame = images[0]
#     if prev_frame is None:
#         print("错误：第一帧为空")
#         return None
    
#     # 初始化EMA光流
#     ema_hsv = np.zeros_like(prev_frame, dtype=np.float32)
#     ema_hsv[..., 1] = 255
#     flow_maps.append(ema_hsv.astype(np.uint8))
    
#     for i in range(1, seq_len):
#         curr_frame = images[i]
#         if curr_frame is None:
#             continue
            
#         # 计算当前帧的光流
#         prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#         curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
#         flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
#                                           pyr_scale=0.5,
#                                           levels=3,
#                                           winsize=7,
#                                           iterations=3,
#                                           poly_n=5,
#                                           poly_sigma=2,
#                                           flags=0)
                                          
#         magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
#         # 当前帧的HSV光流图
#         curr_hsv = np.zeros_like(curr_frame, dtype=np.float32)
#         curr_hsv[..., 1] = 255
#         curr_hsv[..., 0] = angle * 180 / np.pi / 2
#         curr_hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
#         # 更新EMA
#         ema_hsv = alpha * curr_hsv + (1 - alpha) * ema_hsv
        
#         flow_maps.append(ema_hsv.astype(np.uint8))
#         prev_frame = curr_frame.copy()
    
#     return flow_maps

def get_optical_flow(images):
    """
    计算光流并叠加历史帧
    若作为在线增强使用则必须提前建立窗口队列限制images长度
    """

    if len(images) < 1:
        print("错误：图像序列至少需要1张图片")
        return None
    
    if len(images) == 1:
        images = [images[0], images[0]]
        # print("First frame fill ")
        
    # flow_maps = [] # 光流图列表
    seq_len = len(images) 
    
    # 使用第一帧
    prev_frame = images[0]
    if prev_frame is None:
        print("错误：第一帧为空")
        return None
    
    # 初始化叠加光流
    accumulated_hsv = np.zeros_like(prev_frame, dtype=np.float32)
    accumulated_hsv[..., 1] = 255
    
    # 初始化当前帧的HSV光流图
    curr_hsv = np.zeros_like(prev_frame, dtype=np.float32)
    curr_hsv[..., 1] = 255
    
    for i in range(1, seq_len):
        curr_frame = images[i]
        if curr_frame is None:
            continue
            
        # 计算当前帧的光流
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                          pyr_scale=0.5,
                                          levels=3,
                                          winsize=7,
                                          iterations=3,
                                          poly_n=5,
                                          poly_sigma=2,
                                          flags=0)
                                          
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # 更新当前帧的HSV光流图
        curr_hsv[..., 0] = angle * 180 / np.pi / 2
        curr_hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        # 叠加历史帧
        accumulated_hsv += curr_hsv
        
        # flow_maps.append(accumulated_hsv.astype(np.uint8)) # 光流图列表
        prev_frame = curr_frame.copy()
    
    # return flow_maps # 返回光流图列表
    return accumulated_hsv.astype(np.uint8)

# ## 双向光流也是一种思路


def generate_flow_dataset(txt_path, output_dir, root_dir, window_size=2):
    """为数据集生成光流图并保存，保持原数据集的目录结构
    
    Args:
        txt_path: 数据集图片路径的txt文件
        output_dir: 光流图保存目录
        root_dir: 原始数据集根目录
        window_size: 滑动窗口大小，用于计算光流
    边界逻辑:
        1. 如果窗口获得的图像数量小于1，则返回None
        2. 如果窗口获得的图像数量为1，则将第一帧复制一份作为第二帧
        3. 如果窗口获得的图像数量大于1但是小于窗口，则按照实际数量计算光流
        4. 如果窗口获得的图像数量大于窗口，则按照窗口大小计算光流
    """
    # 读取所有图片路径
    with open(txt_path, 'r') as f:
        image_paths = [line.strip().split(',')[0] for line in f]
    
    # 按文件夹分组处理
    folder_groups = {}
    for img_path in image_paths:
        folder = str(Path(img_path).parent)
        if folder not in folder_groups:
            folder_groups[folder] = []
        folder_groups[folder].append(img_path)
    
    total_processed = 0
    total_images = len(image_paths)
    
    # 处理每个文件夹
    for folder, folder_paths in folder_groups.items():
        
        # 按文件名排序，确保时序正确
        folder_paths.sort()
        
        # 读取该文件夹下的所有图片
        images = []
        for img_path in folder_paths:
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
        
        if len(images) < 2:
            print(f"警告：文件夹 {folder} 中的有效图片少于2张，无法得到有效光流，跳过")
            print(f"文件夹 {folder} 中的有效图片数量: {len(images)}")
            continue
        
        # 生成光流图
        for i in range(len(images)):
            # 创建当前帧的滑动窗口
            start_idx = max(0, i - window_size + 1)
            window_images = images[start_idx:i+1]
            # 函数包含第一帧处理
            flow_maps = get_optical_flow(window_images)
            
            if flow_maps is None:
                print('\n No flow maps generated')
                continue
            
            # 保存当前帧对应的光流图
            img_path = Path(folder_paths[i])
            relative_path = img_path.relative_to(root_dir)
            output_path = output_dir / relative_path
            
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存最后一张光流图（对应当前帧）
            cv2.imwrite(str(output_path), flow_maps)
            
            total_processed += 1
            print(f'\r处理进度: {total_processed}/{total_images}', end='')
    
    print("\n光流图生成完成！")


def flow_generation():
    """光流图生成函数"""
    root_dir = Path("g:/dcx_repository/NN_py/ssstest")
    txt_path = str("g:/dcx_repository/NN_py/ssstest/ssstest.txt")
    output_dir = Path("g:/dcx_repository/NN_py/ssstest_flow")
    
    # 生成光流图
    generate_flow_dataset(
        root_dir=root_dir,
        txt_path=txt_path,
        output_dir=output_dir,
        window_size=2,  # 使用3帧窗口
    )

if __name__ == "__main__":
    flow_generation()
    # rgb_optical_flow_2()