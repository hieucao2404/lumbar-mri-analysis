import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, inputs):
        return self.block(inputs)


class ForaminalSeverityCNN(nn.Module):
    """
    Input:
        [batch_size, 1, 128, 128]

    Output:
        [batch_size, 4]

    Class order:
        0 = Normal
        1 = Mild
        2 = Moderate
        3 = Severe
    """

    def __init__(self, number_of_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1, 16),      # 128 -> 64
            ConvBlock(16, 32),     # 64  -> 32
            ConvBlock(32, 64),     # 32  -> 16
            ConvBlock(64, 128),    # 16  -> 8
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, number_of_classes),
        )

    def forward(self, images):
        features = self.features(images)
        logits = self.classifier(features)
        return logits


if __name__ == "__main__":
    model = ForaminalSeverityCNN()

    example = torch.randn(8, 1, 128, 128)
    output = model(example)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Input shape:", example.shape)
    print("Output shape:", output.shape)
    print("Parameters:", parameter_count)