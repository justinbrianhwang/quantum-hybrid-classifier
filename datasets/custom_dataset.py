from torch.utils.data import Dataset
from PIL import Image
import os

class CustomImageDataset(Dataset):
    """
    data_dir/train/class_x/*.jpg
    data_dir/val/class_x/*.jpg
    data_dir/test/class_x/*.jpg
    구조를 자동 인식
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split

        split_dir = os.path.join(root_dir, split)
        self.classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        self.image_paths, self.labels = [], []

        for label, cls in enumerate(self.classes):
            class_dir = os.path.join(split_dir, cls)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

