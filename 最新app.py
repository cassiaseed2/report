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
from torch.utils.data import DataLoader

# 定义神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)  # 添加dropout层防止过拟合
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 应用dropout
        x = self.fc2(x)
        return x

# 初始化Flask应用
app = Flask(__name__)

# 加载模型
model_path = '/workspace/best_model.pth'
model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    tensor = torch.clamp(tensor, 0, 1)  # 确保值在0-1之间
    # 转换为PIL图像
    image = transforms.ToPILImage()(tensor.squeeze(0))
    # 保存图像
    image.save(os.path.join(debug_dir, filename))
    print(f"保存调试图像: {filename}")

def invert_image(image):
    """反转图像颜色（黑变白，白变黑）"""
    return ImageOps.invert(image)

def test_model_on_mnist(model, device, num_samples=None):
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
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=64, 
            shuffle=False
        )
        
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 如果指定了样本数量限制
                if num_samples and total >= num_samples:
                    break
        
        accuracy = 100 * correct / total
        print(f"在MNIST测试集上的准确率: {accuracy:.2f}% (样本数: {total})")
        return accuracy
    except Exception as e:
        print(f"测试MNIST时出错: {str(e)}")
        return 0.0

def preprocess_image(img):
    """预处理图像：转换为灰度、反转颜色、调整大小、标准化"""
    # 转换为灰度
    if img.mode != 'L':
        img = img.convert('L')
    
    # 反转颜色：白底黑字 -> 黑底白字
    img = invert_image(img)
    
    # 应用预处理
    img_tensor = transform(img).to(device)
    
    return img_tensor

# 加载模型
try:
    print(f"正在加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 兼容不同的模型保存格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print("模型加载成功!")
    
    # 验证权重是否加载成功
    print("验证第一层卷积权重:")
    print("权重形状:", model.conv1.weight.shape)
    print("权重均值:", model.conv1.weight.data.mean().item())
    
    # 测试模型在完整MNIST测试集上的表现
    test_model_on_mnist(model, device)
    
except Exception as e:
    print(f"加载模型时出错: {str(e)}")
    print(traceback.format_exc())
    print("使用预训练的MNIST模型作为后备")
    try:
        # 尝试加载预训练模型
        model = CNN()
        model.load_state_dict(torch.load('/workspace/pretrained_mnist_cnn.pth', map_location=device))
        model.to(device)
        model.eval()
        print("后备模型加载成功!")
        test_model_on_mnist(model, device, num_samples=1000)
    except:
        print("无法加载后备模型，使用随机初始化的模型（仅用于演示）")
        model.to(device)
        model.eval()

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
        
        # 预处理图像
        img_tensor = preprocess_image(img)
        
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
