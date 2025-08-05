import torch
import torch.nn as nn
import torchvision.models as models

class ClassicalResNet(nn.Module):
    """순수 고전 ResNet-18"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return torch.sigmoid(self.resnet(x)) if self.resnet.fc.out_features == 1 else self.resnet(x)

class CompressedClassical(nn.Module):
    """하이브리드와 유사한 파라미터 수의 고전 모델"""
    def __init__(self):
        super().__init__()
        base_model = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(base_model.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

