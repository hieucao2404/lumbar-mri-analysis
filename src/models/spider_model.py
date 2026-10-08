import torch
from torch import nn


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class SpiderPfirrmannCNN(nn.Module):
    """Predict Pfirrmann grade I–V from one sagittal disc crop."""

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            conv_block(1, 16),     # 160 -> 80
            conv_block(16, 32),    # 80 -> 40
            conv_block(32, 64),    # 40 -> 20
            conv_block(64, 128),   # 20 -> 10
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [batch, 128, 1, 1]
            nn.Flatten(),             # [batch, 128]
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 5),         # Five grade scores
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(features)