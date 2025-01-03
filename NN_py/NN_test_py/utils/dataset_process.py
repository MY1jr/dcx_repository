from pathlib import Path
import os
from natsort import natsorted

################################################# 数据集裁减 ###################################################

# 处理单个数据集,删除不需要的速度和位置文件夹,并限制每个文件夹中的图片数量
def process_func1(source_path, speedkeeplist, positionkeeplist, image_num):
    """
    处理数据集,删除不需要的速度和位置文件夹,并限制每个文件夹中的图片数量
    
    Args:
        source_path: str, 源文件夹路径
        speedkeeplist: list, 需要保留的速度标签文件夹列表
        positionkeeplist: list, 需要保留的位置标签文件夹列表
        image_num: int, 每个文件夹中保留的图片数量
    """
    source_folder = Path(source_path)

    ## 遍历源文件夹中的所有子文件夹 
    for speed_label_dir in source_folder.iterdir():
        if speed_label_dir.is_dir():
            # 检查速度文件夹名称是否在保留列表中
            if speed_label_dir.name not in speedkeeplist:
                print(f"删除速度标签: {speed_label_dir}")
                # 删除整个速度标签文件夹
                for path in speed_label_dir.glob('**/*'):
                    if path.is_file():
                        path.unlink()
                for path in reversed(list(speed_label_dir.glob('**/*'))):
                    path.rmdir()
                speed_label_dir.rmdir()
                continue
                
            # 遍历位置文件夹
            for position_label_dir in speed_label_dir.iterdir():
                if position_label_dir.is_dir():
                    # 检查位置文件夹名称是否在保留列表中
                    if position_label_dir.name not in positionkeeplist:
                        print(f"删除位置标签: {position_label_dir}")
                        # 删除整个位置标签文件夹
                        for path in position_label_dir.glob('**/*'):
                            if path.is_file():
                                path.unlink()
                        position_label_dir.rmdir()
                        continue
                        
                    # 处理需要保留的文件夹中的图片
                    image_files = []
                    for img_path in position_label_dir.glob('*.[pPjJ][nNpP][gGgG]*'):
                        image_files.append(img_path)
                    # 按自然数重排序    
                    image_files = natsorted(image_files)
                    
                    # 如果图片数量超过限制,删除多余的图片
                    if len(image_files) > image_num:
                        files_to_delete = image_files[image_num:]
                        for file in files_to_delete:
                            print(f"删除图片: {file}")
                            os.remove(file)

    print("func1处理完成")

# 处理单个数据集,限制每个文件夹中的图片数量
def process_func2(source_path, image_num=200):
    """
    处理数据集,保留每个文件夹中前image_num张图片
    Args:
        source_path: str, 源文件夹路径
        image_num: int, 每个文件夹中保留的图片数量
    """

    source_folder = Path(source_path) 

    # 遍历所有子文件夹
    for speed_label_dir in source_folder.iterdir():
        if speed_label_dir.is_dir():
            for position_label_dir in speed_label_dir.iterdir():
                if position_label_dir.is_dir():
                    # 获取所有图片并排序
                    image_files = []
                    for img_path in position_label_dir.glob('*.[pPjJ][nNpP][gGgG]*'):
                        image_files.append(img_path)
                    image_files = natsorted(image_files)
                    
                    # 如果图片数量超过限制,删除多余的图片
                    if len(image_files) > image_num:
                        files_to_delete = image_files[image_num:]
                        for file in files_to_delete:
                            print(f"删除图片: {file}")
                            os.remove(file)
                            
    print("func2处理完成")


if __name__ == "__main__":
    #####################func1####################
    # ### train 
    # train_path = "G:/NN_py/NN_test_py/dcx_train_mini"
    # train_speedkeeplist = ['0', '100', '225', '350', '425', '500','600'] # 示例速度列表
    # train_positionkeeplist = ['0', '832', '1600', '2112', '2944', '3648', '3968'] # 示例位置列表
    # train_image_num = 100
    # process_func1(train_path, train_speedkeeplist, train_positionkeeplist, train_image_num)  # 处理训练集
    # ### test
    # test_path = "G:/NN_py/NN_test_py/dcx_test_mini"
    # test_speedkeeplist = ['200', '450', '550'] # 示例速度列表
    # test_positionkeeplist = ['960', '2176', '3840', '64', '2944', '1600'] # 示例位置列表
    # test_image_num = 50
    # process_func1(test_path, test_speedkeeplist, test_positionkeeplist, test_image_num)  # 处理测试集
    #####################func1####################
    
    #####################func2####################
    dataset_path = "G:/NN_py/dcx_mini"
    dataset_image_num = 60
    process_func2(dataset_path, dataset_image_num)
    #####################func2####################