import matplotlib.pyplot as plt
import numpy as np


# 绘制损失曲线
def plot_losscurve(train_losses, test_losses, save_path='loss_curves.png', start_epoch=1):
    """
    绘制训练和测试损失曲线
    参数:
    - train_losses: 训练损失列表
    - test_losses: 测试损失列表
    - save_path: 保存图像的路径文件名,默认为'loss_curves.png'
    - start_epoch: 起始epoch编号，默认为1
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    plt.figure(figsize=(10,6))
    epochs = range(start_epoch, start_epoch + len(train_losses))
    plt.plot(epochs, train_losses, label='训练损失')
    plt.plot(epochs, test_losses, label='测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和测试损失曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to: {save_path}")
    
# 保存训练和测试损失数据到csv文件
def save_lossdata(train_losses, test_losses, save_path, start_epoch=1):
    """
    保存训练和测试损失数据到csv文件
    Args:
        train_losses: 训练损失列表
        test_losses: 测试损失列表 
        save_path: 保存路径文件名
        start_epoch: 起始epoch编号，默认为1
    """
    with open(save_path, 'w') as f:
        f.write("epoch,train_loss,test_loss\n")  # 写入表头
        for epoch, (train_loss, test_loss) in enumerate(zip(train_losses, test_losses)):
            f.write(f"{epoch+start_epoch},{train_loss:.6f},{test_loss:.6f}\n")
    print(f"Loss data saved to: {save_path}")

# 保存训练和测试及其详细的角度损失与速度损失数据到csv文件
def save_detail_lossdata(train_losses, test_losses, angle_losses, speed_losses, save_path, start_epoch=1):
    """
    保存训练和测试损失数据到csv文件
    Args:
        train_losses: 训练损失列表
        test_losses: 测试损失列表
        angle_losses: 角度损失列表
        speed_losses: 速度损失列表
        save_path: 保存路径文件名
        start_epoch: 起始epoch编号，默认为1
    """
    with open(save_path, 'w') as f:
        f.write("epoch,train_loss,test_loss,angle_loss,speed_loss\n")  # 写入表头
        for epoch, (train_loss, test_loss, angle_loss, speed_loss) in enumerate(zip(train_losses, test_losses, angle_losses, speed_losses)):
            f.write(f"{epoch+start_epoch},{train_loss:.6f},{test_loss:.6f},{angle_loss:.6f},{speed_loss:.6f}\n")
    print(f"Loss data saved to: {save_path}")

# 保存速度和角度误差数据到csv文件
def save_errordata(speed_mae_list, angle_mae_list, speed_mre_list, angle_mre_list, save_path, start_epoch=1):
    """
    保存速度和角度的MAE和MRE数据到csv文件
    Args:
        speed_mae_list: 速度MAE列表
        angle_mae_list: 角度MAE列表
        speed_mre_list: 速度MRE列表
        angle_mre_list: 角度MRE列表
        save_path: 保存路径文件名
        start_epoch: 起始epoch编号，默认为1
    """
    with open(save_path, 'w') as f:
        f.write("epoch,speed_mae,angle_mae,speed_mre,angle_mre\n")  # 写入表头
        for epoch, (speed_mae, angle_mae, speed_mre, angle_mre) in enumerate(zip(speed_mae_list, angle_mae_list, speed_mre_list, angle_mre_list)):
            f.write(f"{epoch+start_epoch},{speed_mae:.6f},{angle_mae:.6f},{speed_mre*100:.4f}%,{angle_mre*100:.4f}%\n")
        # 计算并写入平均速度MAE、平均角度MAE、平均速度MRE和平均角度MRE
        avg_speed_mae = sum(speed_mae_list) / len(speed_mae_list)
        avg_angle_mae = sum(angle_mae_list) / len(angle_mae_list)
        avg_speed_mre = sum(speed_mre_list) / len(speed_mre_list)
        avg_angle_mre = sum(angle_mre_list) / len(angle_mre_list)
        f.write(f"Speed_MAE: {avg_speed_mae:.6f}\nAngle_MAE: {avg_angle_mae:.6f}\nSpeed_MRE: {avg_speed_mre*100:.4f}%\nAngle_MRE: {avg_angle_mre*100:.4f}%\n")
    print(f"Error data saved to: {save_path}")


##########################################待测试##########################################
def plot_losscurve_from_losscsv(csv_path, save_path='loss_curves_from_losscsv.png'):
    """
    从CSV文件读取损失数据并绘制训练和测试损失曲线
    参数:
    - csv_path: 损失数据CSV文件的路径
    - save_path: 保存图像的路径文件名,默认为'loss_curves_from_losscsv.png'
    """
    # 读取CSV文件数据
    epochs = []
    train_losses = []
    test_losses = []
    
    with open(csv_path, 'r') as f:
        # 读取表头,获取列索引
        header = next(f).strip().split(',') # 定义为',' 分隔
        epoch_idx = header.index('epoch')
        train_loss_idx = header.index('train_loss') 
        test_loss_idx = header.index('test_loss')
        
        # 根据列索引读取数据
        for line in f:
            data = line.strip().split(',')
            epochs.append(int(data[epoch_idx]))
            train_losses.append(float(data[train_loss_idx]))
            test_losses.append(float(data[test_loss_idx]))
    
    # 绘制损失曲线
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    plt.figure(figsize=(10,6))
    plt.plot(epochs, train_losses, label='训练损失')
    plt.plot(epochs, test_losses, label='测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和测试损失曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to: {save_path}")
    
##########################################待测试##########################################
def merge_lossdata(csv_path1, csv_path2, save_path):
    """
    合并两个损失数据CSV文件
    参数:
    - csv_path1: 第一个CSV文件路径
    - csv_path2: 第二个CSV文件路径 
    - save_path: 合并后的保存路径
    """
    # 读取第一个文件数据
    data1 = {'epoch': [], 'train_loss': [], 'test_loss': []}
    with open(csv_path1, 'r') as f:
        # 读取表头,获取列索引
        header = next(f).strip().split(',')
        epoch_idx = header.index('epoch')
        train_loss_idx = header.index('train_loss')
        test_loss_idx = header.index('test_loss')
        
        # 根据列索引读取数据
        for line in f:
            data = line.strip().split(',')
            data1['epoch'].append(int(data[epoch_idx]))
            data1['train_loss'].append(float(data[train_loss_idx]))
            data1['test_loss'].append(float(data[test_loss_idx]))
    
    # 读取第二个文件数据
    data2 = {'epoch': [], 'train_loss': [], 'test_loss': []}
    with open(csv_path2, 'r') as f:
        # 读取表头,获取列索引
        header = next(f).strip().split(',')
        epoch_idx = header.index('epoch')
        train_loss_idx = header.index('train_loss')
        test_loss_idx = header.index('test_loss')
        
        # 根据列索引读取数据
        for line in f:
            data = line.strip().split(',')
            data2['train_loss'].append(float(data[train_loss_idx]))
            data2['test_loss'].append(float(data[test_loss_idx]))
    
    # 合并数据并保存
    with open(save_path, 'w') as f:
        f.write("epoch,train_loss,test_loss\n")  # 写入表头
        
        # 写入第一个文件的数据
        for epoch, train_loss, test_loss in zip(data1['epoch'], data1['train_loss'], data1['test_loss']):
            f.write(f"{epoch},{train_loss:.6f},{test_loss:.6f}\n")
            
        # 写入第二个文件的数据，epoch从最后一个epoch继续
        last_epoch = data1['epoch'][-1] if data1['epoch'] else 0
        for i, (train_loss, test_loss) in enumerate(zip(data2['train_loss'], data2['test_loss'])):
            new_epoch = last_epoch + i + 1
            f.write(f"{new_epoch},{train_loss:.6f},{test_loss:.6f}\n")
    
    print(f"Merged loss data saved to: {save_path}")
    
##########################################待测试##########################################

def merge_errordata(csv_path1, csv_path2, save_path):
    """
    合并两个误差数据CSV文件 ### 统计量开始行必须是'Speed_MAE'
    参数:
    - csv_path1: 第一个CSV文件路径
    - csv_path2: 第二个CSV文件路径
    - save_path: 合并后的保存路径
    """
    # 读取第一个文件数据
    data1 = {'epoch': [], 'speed_mae': [], 'angle_mae': [], 'speed_mre': [], 'angle_mre': []}
    with open(csv_path1, 'r') as f:
        lines = f.readlines()
        # 找到统计量开始的行
        for i, line in enumerate(lines):
            if line.startswith('Speed_MAE'): ### 依据找到Speed_MAE判断统计量开始 ###
                lines = lines[1:i]  # 裁剪，只保留表头之后到统计量之前的数据
                break
        
        # 读取表头,获取列索引
        header = lines[0].strip().split(',')
        epoch_idx = header.index('epoch')
        speed_mae_idx = header.index('speed_mae')
        angle_mae_idx = header.index('angle_mae')
        speed_mre_idx = header.index('speed_mre')
        angle_mre_idx = header.index('angle_mre')
                
        # 根据列索引读取数据
        for line in lines[1:]:  # 跳过表头
            data = line.strip().split(',')
            data1['epoch'].append(int(data[epoch_idx]))
            data1['speed_mae'].append(float(data[speed_mae_idx]))
            data1['angle_mae'].append(float(data[angle_mae_idx]))
            data1['speed_mre'].append(float(data[speed_mre_idx].strip('%'))/100)
            data1['angle_mre'].append(float(data[angle_mre_idx].strip('%'))/100)
    
    # 读取第二个文件数据
    data2 = {'epoch': [], 'speed_mae': [], 'angle_mae': [], 'speed_mre': [], 'angle_mre': []}
    with open(csv_path2, 'r') as f:
        lines = f.readlines()
        # 找到统计量开始的行
        for i, line in enumerate(lines):
            if line.startswith('Speed_MAE'): ### 依据找到Speed_MAE判断统计量开始 ###
                lines = lines[1:i]  # 裁剪，只保留表头之后到统计量之前的数据
                break
        
        # 读取表头,获取列索引
        header = lines[0].strip().split(',')
        epoch_idx = header.index('epoch')
        speed_mae_idx = header.index('speed_mae')
        angle_mae_idx = header.index('angle_mae')
        speed_mre_idx = header.index('speed_mre')
        angle_mre_idx = header.index('angle_mre')
                
        # 根据列索引读取数据
        for line in lines[1:]:  # 跳过表头
            data = line.strip().split(',')
            data2['speed_mae'].append(float(data[speed_mae_idx]))
            data2['angle_mae'].append(float(data[angle_mae_idx]))
            data2['speed_mre'].append(float(data[speed_mre_idx].strip('%'))/100)
            data2['angle_mre'].append(float(data[angle_mre_idx].strip('%'))/100)
    
    # 合并数据并保存
    with open(save_path, 'w') as f:
        f.write("epoch,speed_mae,angle_mae,speed_mre,angle_mre\n")  # 写入表头
        
        # 写入第一个文件的数据
        for i in range(len(data1['epoch'])):
            f.write(f"{data1['epoch'][i]},{data1['speed_mae'][i]:.6f},{data1['angle_mae'][i]:.6f},"
                   f"{data1['speed_mre'][i]*100:.4f}%,{data1['angle_mre'][i]*100:.4f}%\n")
            
        # 写入第二个文件的数据，epoch从最后一个epoch继续
        last_epoch = data1['epoch'][-1] if data1['epoch'] else 0
        for i in range(len(data2['speed_mae'])):
            new_epoch = last_epoch + i + 1
            f.write(f"{new_epoch},{data2['speed_mae'][i]:.6f},{data2['angle_mae'][i]:.6f},"
                   f"{data2['speed_mre'][i]*100:.4f}%,{data2['angle_mre'][i]*100:.4f}%\n")
        
        # 计算并写入所有数据的平均值
        all_speed_mae = data1['speed_mae'] + data2['speed_mae']
        all_angle_mae = data1['angle_mae'] + data2['angle_mae']
        all_speed_mre = data1['speed_mre'] + data2['speed_mre']
        all_angle_mre = data1['angle_mre'] + data2['angle_mre']
        
        avg_speed_mae = sum(all_speed_mae) / len(all_speed_mae)
        avg_angle_mae = sum(all_angle_mae) / len(all_angle_mae)
        avg_speed_mre = sum(all_speed_mre) / len(all_speed_mre)
        avg_angle_mre = sum(all_angle_mre) / len(all_angle_mre)
        
        f.write(f"Speed_MAE: {avg_speed_mae:.6f}\n")
        f.write(f"Angle_MAE: {avg_angle_mae:.6f}\n")
        f.write(f"Speed_MRE: {avg_speed_mre*100:.4f}%\n")
        f.write(f"Angle_MRE: {avg_angle_mre*100:.4f}%\n")
    
    print(f"Merged error data saved to: {save_path}")
    
##########################################待测试##########################################

# 断点数据合并
if __name__ == "__main__":
    
    # 测试损失数据合并
    loss_csv1 = "G:/NN_py/NN_test_py/pth/processing/01/loss_data.csv" 
    loss_csv2 = "G:/NN_py/NN_test_py/pth/processing/02/loss_data.csv"
    merged_loss_path = "G:/NN_py/NN_test_py/pth/processing/merged_loss.csv"
    
    merge_lossdata(loss_csv1, loss_csv2, merged_loss_path)
    print(f"损失数据合并完成: {merged_loss_path}")
    
    # 测试误差数据合并
    error_csv1 = "G:/NN_py/NN_test_py/pth/processing/01/error_data.csv"  
    error_csv2 = "G:/NN_py/NN_test_py/pth/processing/02/error_data.csv"
    merged_error_path = "G:/NN_py/NN_test_py/pth/processing/merged_error.csv"
    
    merge_errordata(error_csv1, error_csv2, merged_error_path)
    print(f"误差数据合并完成: {merged_error_path}")
    
    # 测试损失曲线绘制
    plot_losscurve_from_losscsv(merged_loss_path, save_path='G:/NN_py/NN_test_py/pth/processing/merged_loss_curve.png')
