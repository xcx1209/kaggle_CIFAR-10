import os
import torch
from PIL import Image
from torchvision import transforms
from model import CIFAR10Model
from torch.utils.data import Dataset, DataLoader
import torch

#定义类别名称
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #数据预处理
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载模型
    model = CIFAR10Model().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()  # 设置为评估模式

    # 测试图片目录
    test_dir = './test'
    # 输出文件
    output_file = 'predictions.csv'

    # 获取测试图片列表并排序
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.png')],
                        key=lambda x: int(x.split('.')[0]))

    # 创建预测结果文件
    with open(output_file, 'w') as f:
        f.write('id,label\n')  # 写入标题行

        # 遍历所有测试图片
        for filename in test_files:
            img_path = os.path.join(test_dir, filename)
            img_id = os.path.splitext(filename)[0]  # 获取图片ID

            # 加载并预处理图像
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)  # 添加批次维度

            # 模型预测
            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)
                class_index = predicted.item()
                class_name = class_names[class_index]  # 获取类别名称

            # 写入结果
            f.write(f'{img_id},{class_name}\n')


    print(f'Predictions saved to {output_file}')

