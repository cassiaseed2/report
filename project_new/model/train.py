# model/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import CNN
import os
#from project_new.data.load_data import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

learning_rates = [0.001, 0.0005]
batch_sizes = [32, 64]

best_acc = 0.0
os.makedirs("../saved_models", exist_ok=True)

for lr in learning_rates:
    for batch_size in batch_sizes:
        print(f"训练中：学习率={lr}, batch_size={batch_size}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = CNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.NLLLoss()

        for epoch in range(1, 5):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            # 验证
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            acc = correct / total
            print(f"Epoch {epoch} 验证准确率: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                torch.save({'model_state_dict': model.state_dict()}, "../saved_models/best_model.pth")
                print(f"保存新最优模型 acc={acc:.4f}")

print(f"训练完成。最优验证准确率={best_acc:.4f}")
