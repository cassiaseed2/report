import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_data(batch_size=64, data_dir='./data'):
    """
    下载并加载MNIST数据集，返回训练集和验证集的 DataLoader。

    参数:
    - batch_size: 批大小
    - data_dir: 数据存储路径

    返回:
    - train_loader: 训练集 DataLoader
    - val_loader: 验证集 DataLoader
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 下载并加载 MNIST 数据
    dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)

    # 按 8:2 划分训练和验证
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
