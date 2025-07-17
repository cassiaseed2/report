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