#模型训练与调优
import torch
import torch.nn as nn
import torch.optim as optim

# 检测可用设备
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输入通道1, 输出32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 尺寸减半
        self.fc1 = nn.Linear(64*7*7, 128)  # 两次池化后为7x7
        self.fc2 = nn.Linear(128, 10)  # 输出10类
    
    def forward(self, x):
        x = F.relu(self.conv1(x))  # 28x28 -> 28x28
        x = self.pool(x)           # 28x28 -> 14x14
        x = F.relu(self.conv2(x))  # 14x14 -> 14x14
        x = self.pool(x)           # 14x14 -> 7x7
        x = x.view(-1, 64*7*7)     # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# 从队友处获取预定义的模型（成员B提供）
model = CNN().to(device)  # 假设CNN类已定义

# 定义损失函数（分类任务常用交叉熵）
criterion = nn.CrossEntropyLoss()

# 选择优化器（Adam常用，SGD可选）
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学习率调度器（可选）
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:  # 从成员A处获取DataLoader
            # 数据转移到设备
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计指标
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 计算epoch统计量
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        # 打印进度
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")
        
        # 学习率调整
        if scheduler:
            scheduler.step()
    
    return model
# 网格搜索示例（简化版）
for lr in [0.1, 0.01, 0.001]:
    for batch_size in [32, 64, 128]:
        # 重新初始化模型
        model = CNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # 更新DataLoader（需与成员A协作）
        train_loader = DataLoader(..., batch_size=batch_size)
        
        # 训练并记录结果
        train_model(...)
        accuracy = evaluate(model, val_loader)  # 验证集评估
        save_results(lr, batch_size, accuracy)  # 记录参数组合效果
