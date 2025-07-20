# Handwritten Digit Recognition with CNN and Flask

## 项目概述
本项目实现了一个基于卷积神经网络(CNN)的手写数字识别系统，结合Flask框架构建交互式Web应用。系统使用PyTorch实现CNN模型，在MNIST数据集上训练达到93.7%的平均分类准确率，Top-3准确率高达99.4%。项目提供完整的机器学习工作流实现，包括数据预处理、模型训练、评估和部署。

## 主要特性
- **高性能CNN架构**：双层卷积+池化+全连接结构，支持Dropout防止过拟合
- **高级训练框架**：实现梯度裁剪、学习率调度和早停机制
- **交互式Web界面**：通过Flask提供实时手写数字识别服务
- **详细可视化分析**：提供混淆矩阵、置信度分布等诊断工具
- **超参数优化**：集成随机搜索策略寻找最优超参数组合

## 安装指南
### 环境要求
- Python 3.7+
- PyTorch 1.8+
- Flask 2.0+
- Pillow 9.0+

### 安装步骤
```bash
# 克隆项目仓库
git clone https://github.com/your-repo/handwritten-digit-recognition.git
cd handwritten-digit-recognition

# 安装依赖
pip install torch torchvision flask pillow numpy matplotlib
