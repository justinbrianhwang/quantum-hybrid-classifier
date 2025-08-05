from torchvision import datasets

def get_cifar10(data_dir, download, train_transform, test_transform):
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=test_transform)
    return train_dataset, test_dataset

