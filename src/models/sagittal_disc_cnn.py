from torch import nn

class SagittalDiscCNN(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []
        input_channels = 1

        for output_channels in (16, 32, 64, 128):
            layers.extend([
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            ])
            input_channels = output_channels

        self.backbone = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.pfirrman_head = nn.Linear(128, 5)
        self.bulge_head = nn.Linear(128, 1)
        self.herniation_head = nn.Linear(128, 1)

    def forward(self, images):
        features = self.backbone(images)

        return{
            "pfirrmann": self.pfirrman_head(features),
            "bulge": self.bulge_head(features).squeeze(1),
            "herniation": self.herniation_head(features).squeeze(1),
        }

    