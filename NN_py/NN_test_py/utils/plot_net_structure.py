####################################没调好####################################
######################存在时间步循环导致重复计算层的问题######################  
##########################################################################
from torchviz import make_dot
import torch.nn as nn
import torch
from pathlib import Path
import sys
sys.path.append('G:/NN_py/NN_test_py/')
from models import CNN_LSTM, ConvLSTM, GTNet
from torchinfo import summary
def visualize_network(model, input_size, save_path):
    """
    使用 torchinfo 可视化神经网络结构
    
    参数:
        model: PyTorch 神经网络模型
        input_size: 输入数据的尺寸
        save_path: 完整的保存路径，包含文件夹路径和文件名 (不需要扩展名)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 设置为评估模式并移至CPU
    model = model.cpu().eval()
    
    # 生成模型摘要
    model_summary = summary(
        model, 
        input_size=input_size,
        col_names=["input_size", "output_size", "num_params"],  # 简化显示的列
        depth=10,  # 控制显示深度
        verbose=1,  # 显示详细信息
        device='cpu',
        mode='eval',  # 使用评估模式
        row_settings=["depth", "var_names"],  # 显示层级深度
        col_width=20
    )
    
    # 将摘要保存到文件
    with open(f"{save_path}.txt", 'w', encoding='utf-8') as f:
        # 添加模型结构的字符串表示
        f.write("Model Architecture:\n")
        f.write("=" * 80 + "\n")
        f.write(str(model) + "\n\n")
        
        # 添加模型摘要
        f.write("Model Summary:\n")
        f.write("=" * 80 + "\n")
        f.write(str(model_summary))
    
    print(f"网络结构已保存至: {save_path}.txt")
    print("\n模型结构摘要:")
    print(model_summary)

if __name__ == "__main__":
    # 创建模型实例
    model = GTNet( 
        in_channels=3,
        out_channels=3,
        height=64,
        width=64,
        time_step=20
    )

    # 定义输入尺寸 [batch_size, time_step, channels, height, width]
    input_size = [32, 20, 3, 64, 64]

    # 可视化网络结构
    visualize_network(model, input_size, '../network_structure/test_network')