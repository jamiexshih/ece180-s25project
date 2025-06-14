import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

def get_loaders(batch_size):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = datasets.STL10('./data', split='train', download=True, transform=transform)
    test_data = datasets.STL10('./data', split='test', download=True, transform=transform)

    val_data, test_data = random_split(test_data, [300, 500])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return train_loader, val_loader, test_loader
