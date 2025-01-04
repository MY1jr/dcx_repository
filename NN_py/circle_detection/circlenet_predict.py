import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from circlenet import SimpleCNN, CircleDataset, img_transform, AverageMeter
import cv2

def test_model(model_path, label_txt, batch_size=32):
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(in_channels=3, out_channels=6).to(device)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()

    # 加载数据集
    test_dataset = CircleDataset(label_txt=label_txt, img_transform=img_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义损失函数
    criterion = nn.MSELoss()

    # 初始化损失计量器
    test_loss_meter = AverageMeter()

    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # 分离预测值
            inner_circle_pred = output[:, 0:3]
            outer_circle_pred = output[:, 3:6]

            # 获取原始标签
            inner_circle_target = target[:, 0:3]
            outer_circle_target = target[:, 3:6]

            # 计算损失
            inner_circle_loss = criterion(inner_circle_pred, inner_circle_target)
            outer_circle_loss = criterion(outer_circle_pred, outer_circle_target)
            loss = inner_circle_loss + outer_circle_loss

            # 更新损失计量器
            test_loss_meter.update(loss.item(), data.size(0))

    print(f'测试集平均损失: {test_loss_meter.avg:.6f}')
    
def test_single_image(model_path, image_path):
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(in_channels=3, out_channels=6).to(device)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()

    # 加载并预处理单张图片
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
    image = img_transform(image)  # 应用图像预处理
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)  # 转换为Tensor并添加批次维度

    with torch.no_grad():
        output = model(image)

        # 分离预测值
        inner_circle_pred = output[:, 0:3]
        outer_circle_pred = output[:, 3:6]

    print(f'内圆预测: {inner_circle_pred.cpu().numpy()}')
    print(f'外圆预测: {outer_circle_pred.cpu().numpy()}')

if __name__ == "__main__":
    # # 设置模型路径和标签文件路径
    # model_path = 'G:/NN_py/circle_detection/circlenet/pth/best.pth'
    # label_txt = 'G:/NN_py/dcx_mini/circles_label.txt'
    
    # # 运行测试
    # test_model(model_path, label_txt)
    
    # 设置模型路径和图片路径
    model_path = 'G:/NN_py/circle_detection/circlenet/pth/best.pth'
    image_path = 'G:/NN_py/dcx_mini/200/128/15.png'
    
    # 运行单张图片测试
    test_single_image(model_path, image_path)