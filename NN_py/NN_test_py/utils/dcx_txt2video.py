import cv2
from tqdm import tqdm

def create_video_from_txt(txt_path, output_path, pic_size=None, num_frames=None, fps=30, start_index=0):
    """
    从txt文件中读取图片路径并合成视频
    Args:
        txt_path: txt文件路径
        output_path: 输出视频路径
        pic_size: 图片尺寸，默认None表示不调整尺寸(height, width)
        num_frames: 需要的帧数，默认None表示使用所有帧
        fps: 视频帧率，默认30fps
        start_index: 起始序号，默认0, 与num_frames配合使用
    """
    # 根据帧数和起点序号读取图片路径
    image_paths = []
    with open(txt_path, 'r') as f:
        for i, line in enumerate(f):
            if i < start_index:
                continue
            if num_frames is not None and i - start_index >= num_frames:
                break
            image_paths.append(line.strip().split(',')[0])
    
    if not image_paths:
        print("没有找到图片路径")
        return
    
    if pic_size is None:
        # 读取第一张图片来获取尺寸
        first_img_path = image_paths[0]
        first_frame = cv2.imread(first_img_path)
        if first_frame is None:
            print(f"无法读取第一张图片: {first_img_path}")
            return  
        
        height, width = first_frame.shape[:2] # shape返回(高,宽,通道数)
    else:
        height, width = pic_size # pic_size应该是(高,宽)格式
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 遍历所有图片并写入视频
    for img_path in tqdm(image_paths, desc="video writing"):
        frame = cv2.imread(img_path)
        if frame is not None:
            if frame.shape[:2] != (height, width): # shape返回(高,宽,通道数)
                frame = cv2.resize(frame, (width, height)) # resize需要(宽,高)格式
            out.write(frame)
        else:
            print(f"无法读取图片: {img_path}")
    
    # 释放资源
    out.release()
    print(f"video saved to: {output_path}")

if __name__ == '__main__':
    label_txt = "G:/NN_py/dcx/dcx.txt"
    output_video_path = 'G:/NN_py/dcx_video.mp4'
    
    create_video_from_txt(
        txt_path=label_txt,
        output_path=output_video_path,
        pic_size=(100,100),
        num_frames=1800, # 可以根据需要修改帧数，设为None则使用所有帧
        fps=30,
        start_index=400000
    )