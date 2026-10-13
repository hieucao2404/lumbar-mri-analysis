import torch
from torch import nn


def conv_block(
    in_channels: int,
    out_channels: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        ),
        nn.GroupNorm(4, out_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        ),
        nn.GroupNorm(4, out_channels),
        nn.ReLU(inplace=True),

        nn.MaxPool2d(2),
    )


class SpiderPosteriorCNN(nn.Module):
    """
    Input: [batch, 2, 160, 160]

    Output: [batch, 2] raw logits
        column 0 = disc bulging
        column 1 = disc herniation
    """

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            conv_block(2, 16),     # 160 -> 80
            conv_block(16, 32),    # 80 -> 40
            conv_block(32, 64),    # 40 -> 20
            conv_block(64, 96),    # 20 -> 10
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, 48),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(48, 2),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features)