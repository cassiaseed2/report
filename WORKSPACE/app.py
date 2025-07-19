import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from flask import Flask, request, render_template
import numpy as np
import matplotlib.pyplot as plt
import os
import traceback
import requests
from torchvision.datasets import MNIST

# 定义神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化Flask应用
app = Flask(__name__)

# 加载模型
model_path = '/workspace/best_model.pth'
model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # 加载模型文件
    print(f"正在加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 修复：尝试不同的键名加载模型
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        # 尝试直接加载整个检查点
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print("模型加载成功!")
    
    # 验证权重是否加载成功
    print("验证第一层卷积权重:")
    print("权重形状:", model.conv1.weight.shape)
    print("权重均值:", model.conv1.weight.data.mean().item())
    print("权重标准差:", model.conv1.weight.data.std().item())
    
    # 测试模型在MNIST测试集上的表现
    test_accuracy = test_model_on_mnist(model, device)
    print(f"在MNIST测试集上的准确率: {test_accuracy:.2f}%")
    
    # 测试特定样本
    test_samples(model, device)

except Exception as e:
    print(f"加载模型时出错: {str(e)}")
    print(traceback.format_exc())
    print("使用随机初始化的模型（仅用于演示）")
    model.to(device)
    model.eval()

# 图像预处理 - 使用与训练时相同的参数
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准参数
])

# 创建调试图像目录
debug_dir = "/workspace/debug_images"
os.makedirs(debug_dir, exist_ok=True)

def save_debug_image(tensor, filename):
    """保存调试图像"""
    # 反归一化
    tensor = (tensor * 0.3081) + 0.1307
    # 转换为PIL图像
    image = transforms.ToPILImage()(tensor.squeeze(0))
    # 保存图像
    image.save(os.path.join(debug_dir, filename))
    print(f"保存调试图像: {filename}")

def invert_image(image):
    """反转图像颜色（黑变白，白变黑）"""
    return ImageOps.invert(image)

def test_model_on_mnist(model, device, num_samples=1000):
    """在MNIST测试集上测试模型性能"""
    try:
        # 加载MNIST测试集
        test_dataset = MNIST(
            root='./mnist_data', 
            train=False, 
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=64, 
            shuffle=False
        )
        
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if total >= num_samples:
                    break
        
        accuracy = 100 * correct / total
        return accuracy
    except Exception as e:
        print(f"测试MNIST时出错: {str(e)}")
        return 0.0

def test_samples(model, device):
    """测试特定样本以诊断模型问题"""
    samples = [
        ("数字0", "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/MnistExamples.png/320px-MnistExamples.png", (0, 28)),
        ("数字1", "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/MnistExamples.png/320px-MnistExamples.png", (28, 28)),
        ("数字8", "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/MnistExamples.png/320px-MnistExamples.png", (196, 28))
    ]
    
    print("\n测试特定样本:")
    for name, url, crop_box in samples:
        try:
            # 下载并处理样本图像
            response = requests.get(url)
            img = Image.open(io.BytesIO(response.content))
            img = img.crop((crop_box[0], crop_box[1], crop_box[0]+28, crop_box[1]+28))
            img = img.convert('L')
            
            # 应用预处理
            img_tensor = transform(img).to(device).unsqueeze(0)
            
            # 预测
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = F.softmax(output, dim=1)
                pred = output.argmax(dim=1).item()
                
            print(f"{name} - 预测: {pred}, 概率: {probabilities[0][pred]:.4f}")
        except Exception as e:
            print(f"测试{name}时出错: {str(e)}")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return '没有上传文件'
    
    file = request.files['file']
    if file.filename == '':
        return '没有选择文件'
    
    try:
        # 处理上传的图片
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # 保存原始图像用于调试
        original_path = os.path.join(debug_dir, "original.png")
        img.save(original_path)
        
        # 检查图像模式并转换为灰度
        if img.mode != 'L':
            img = img.convert('L')
            print("已将图像转换为灰度")
        
        # 应用预处理
        img_tensor = transform(img).to(device)
        
        # 保存预处理后的图像用于调试
        save_debug_image(img_tensor.cpu(), "preprocessed.png")
        
        # 增加batch维度
        img_tensor = img_tensor.unsqueeze(0)
        
        # 预测
        with torch.no_grad():
            output = model(img_tensor)
            print("模型原始输出:", output)
            
            # 应用softmax获取概率
            probabilities = F.softmax(output, dim=1)
            print("预测概率:", probabilities)
            
            pred = output.argmax(dim=1).item()
            
            # 获取前3个预测结果
            top_probs, top_preds = torch.topk(probabilities, 3, dim=1)
            top_probs = top_probs.squeeze().tolist()
            top_preds = top_preds.squeeze().tolist()
            
            # 格式化结果
            result = f"识别结果: {pred}\n"
            result += "预测详情:\n"
            for i, (prob, pred_class) in enumerate(zip(top_probs, top_preds)):
                result += f"{i+1}. 数字 {pred_class} - 概率: {prob*100:.2f}%\n"
        
        return result
    
    except Exception as e:
        return f'错误: {str(e)}'

if __name__ == '__main__':
    # 打印环境信息
    print("=" * 50)
    print(f"Python版本: {torch.__version__}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"设备: {device}")
    print("=" * 50)
    
    # 运行应用
    app.run(host='0.0.0.0', port=8280, debug=True)
