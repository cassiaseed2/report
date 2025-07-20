import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import csv
import os
from datetime import datetime


import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):  # 示例：10分类任务
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # 输入通道3（RGB）
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # 尺寸计算见下方说明
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [bs, 16, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))    # [bs, 32, 16, 16] → [bs, 32, 8, 8]
        x = x.view(-1, 32 * 8 * 8)             # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 根据你的数据集调整
])

# 创建数据集
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# 定义 train_loader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,  # 调整批次大小
    shuffle=True,
    num_workers=2   # 根据CPU核心数调整
)

# 现在你可以安全使用 train_loader
for batch_idx, (data, target) in enumerate(train_loader):
    # 你的训练代码...
    print(f"Batch {batch_idx}: 数据尺寸 {data.size()}, 标签尺寸 {target.size()}")

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=10, device="cpu", model_name="model"):
    """
    完整的训练和验证流程
    返回: 训练历史记录, 最佳模型权重
    """
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'epoch_time': []
    }
    
    best_val_acc = 0.0
    best_model_weights = None
    
    model.to(device)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # 训练阶段
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计指标
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # 验证阶段
        val_loss, val_correct, val_total = evaluate_model(model, val_loader, criterion, device)
        
        # 计算指标
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # 更新学习率
        if scheduler:
            scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(time.time() - epoch_start)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = model.state_dict().copy()
            torch.save(best_model_weights, f'best_{model_name}.pth')
        
        # 打印进度
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"Time: {history['epoch_time'][-1]:.2f}s")
    
    return history, best_model_weights

def evaluate_model(model, data_loader, criterion, device="cpu"):
    """评估模型在给定数据集上的表现"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return total_loss, correct, total

def save_results_to_csv(results, filename="hyperparam_results.csv"):
    """将超参数搜索结果保存到CSV文件"""
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # 如果文件不存在，写入标题行
        if not file_exists:
            headers = list(results[0].keys())
            writer.writerow(headers)
        
        # 写入数据行
        for result in results:
            writer.writerow(list(result.values()))

def hyperparameter_search(model_class, train_loader, val_loader, param_grid, epochs=10, device="cpu"):
    """
    执行超参数网格搜索
    返回: 所有参数组合的结果列表
    """
    results = []
    
    # 生成所有参数组合
    all_params = []
    for lr in param_grid.get('lr', [0.001]):
        for batch_size in param_grid.get('batch_size', [64]):
            for weight_decay in param_grid.get('weight_decay', [0]):
                all_params.append({
                    'lr': lr,
                    'batch_size': batch_size,
                    'weight_decay': weight_decay
                })
    
    print(f"Starting hyperparameter search with {len(all_params)} combinations...")
    
    for i, params in enumerate(all_params):
        print(f"\n=== Testing combination {i+1}/{len(all_params)}: {params} ===")
        
        # 创建新模型实例
        model = model_class()
        
        # 创建新的优化器
        optimizer = optim.Adam(
            model.parameters(), 
            lr=params['lr'],
            weight_decay=params['weight_decay']
        )
        
        # 训练模型
        model_name = f"model_lr{params['lr']}_bs{params['batch_size']}_wd{params['weight_decay']}"
        history, best_weights = train_and_evaluate(
            model, train_loader, val_loader, 
            nn.CrossEntropyLoss(), optimizer,
            epochs=epochs, device=device,
            model_name=model_name
        )
        
        # 记录结果
        result = {
            'lr': params['lr'],
            'batch_size': params['batch_size'],
            'weight_decay': params['weight_decay'],
            'best_val_acc': max(history['val_acc']),
            'final_train_acc': history['train_acc'][-1],
            'total_time': sum(history['epoch_time']),
            'model_name': model_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        results.append(result)
        
        # 保存结果到CSV
        save_results_to_csv([result])
    
    return results

# 使用示例
if __name__ == "__main__":
    # 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 假设这些数据加载器已由队友提供
    # train_loader = DataLoader(...)
    # val_loader = DataLoader(...)
    
    # 定义参数网格
    param_grid = {
        'lr': [0.1, 0.01, 0.001, 0.0001],
        'batch_size': [32, 64, 128],
        'weight_decay': [0, 1e-5, 1e-4]
    }
    
    # 执行超参数搜索
    search_results = hyperparameter_search(
        CNN,  # 假设CNN类已定义
        train_loader,
        val_loader,
        param_grid,
        epochs=15,
        device=device
    )
    
    # 找到最佳参数组合
    best_result = max(search_results, key=lambda x: x['best_val_acc'])
    print(f"\nBest hyperparameters: lr={best_result['lr']}, "
          f"batch_size={best_result['batch_size']}, "
          f"weight_decay={best_result['weight_decay']} "
          f"with val_acc={best_result['best_val_acc']:.2f}%")
