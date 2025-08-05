from .custom_dataset import CustomImageDataset
from .cifar10 import get_cifar10

def get_dataset_and_loaders(name, data_dir, batch_size, download=False):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    name = name.lower()

    # 공용 Transform
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    if name == "custom":
        train_dataset = CustomImageDataset(data_dir, 'train', train_transform)
        val_dataset = CustomImageDataset(data_dir, 'val', test_transform)
        test_dataset = CustomImageDataset(data_dir, 'test', test_transform)
        num_classes = len(train_dataset.classes)
    elif name == "cifar10":
        train_dataset, test_dataset = get_cifar10(data_dir, download, train_transform, test_transform)
        # CIFAR-10에는 별도 val 없음 → train의 10%를 val로 split
        val_size = int(0.1 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, num_classes

