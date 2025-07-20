# MNIST 手写数字识别 Web 服务

基于 PyTorch 的卷积神经网络(CNN)实现的 MNIST 手写数字识别系统，通过 Flask 框架提供 Web API 服务。

## 功能特性

1. **高精度识别**：使用优化的 CNN 模型架构，在 MNIST 测试集上实现 >93% 的准确率
2. **预处理增强**：
   - 自动灰度转换
   - 颜色反转（白底黑字 → 黑底白字）
   - 标准化处理（使用 MNIST 标准参数）
3. **调试支持**：
   - 保存原始/预处理图像至 `/workspace/debug_images`
   - 详细的预测概率输出
4. **错误恢复机制**：
   - 主模型加载失败时自动启用后备模型
   - 多重异常捕获机制
5. **预测详情**：
   - 返回 top 3 预测结果及置信度
   - 原始模型输出和 softmax 概率输出

## 系统要求

- Python 3.7+
- 依赖库：
  ```bash
  torch==2.0.0
  torchvision==0.15.1
  flask==2.2.3
  pillow==9.4.0

## 文件结构

project/
├── app.py                 # 主应用文件
├── best_model.pth         # 主模型权重
├── pretrained_mnist_cnn.pth # 后备模型权重
├── templates/             # HTML 模板
│   └── index.html         
└── debug_images/          # 调试图像存储（自动创建）

## 模型架构

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)  # 防过拟合
        
