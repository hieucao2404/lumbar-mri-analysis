from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.models.spider_disc_unet import ConvBlock


class SpiderDiscMultitaskUNet(nn.Module):
    """
    Shared U-Net with two output heads:

    1. Binary head:
       Detect every intervertebral disc.

    2. Coordinate-aware level head:
       Classify disc pixels as:
           0 = L1-L2
           1 = L2-L3
           2 = L3-L4
           3 = L4-L5
           4 = L5-S1
           5 = Upper/non-lumbar disc
    """

    def __init__(
        self,
        number_of_level_classes: int = 6,
    ) -> None:
        super().__init__()

        self.encoder1 = ConvBlock(1, 16)
        self.encoder2 = ConvBlock(16, 32)
        self.encoder3 = ConvBlock(32, 64)
        self.bottleneck = ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.decoder3 = ConvBlock(
            128 + 64,
            64,
        )

        self.decoder2 = ConvBlock(
            64 + 32,
            32,
        )

        self.decoder1 = ConvBlock(
            32 + 16,
            16,
        )

        # This layer is copied exactly from the successfully trained
        # binary disc-localization U-Net.
        self.binary_output_layer = nn.Conv2d(
            16,
            1,
            kernel_size=1,
        )

        # Decoder features contain 16 channels.
        # Add normalized X and Y coordinates:
        #
        # 16 decoder channels + 2 coordinate channels = 18.
        self.level_output_layer = nn.Sequential(
            nn.Conv2d(
                18,
                32,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=4,
                num_channels=32,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                32,
                number_of_level_classes,
                kernel_size=1,
            ),
        )

    @staticmethod
    def upsample_to(
        tensor: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        return F.interpolate(
            tensor,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def create_coordinate_channels(
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create normalized coordinate maps.

        X ranges from -1 at the left to +1 at the right.
        Y ranges from -1 at the top to +1 at the bottom.

        Output shape:
            [batch, 2, height, width]
        """
        batch_size = features.shape[0]
        height = features.shape[2]
        width = features.shape[3]

        y_values = torch.linspace(
            -1.0,
            1.0,
            steps=height,
            device=features.device,
            dtype=features.dtype,
        )

        x_values = torch.linspace(
            -1.0,
            1.0,
            steps=width,
            device=features.device,
            dtype=features.dtype,
        )

        y_grid, x_grid = torch.meshgrid(
            y_values,
            x_values,
            indexing="ij",
        )

        coordinates = torch.stack(
            [x_grid, y_grid],
            dim=0,
        )

        coordinates = coordinates.unsqueeze(0)

        return coordinates.expand(
            batch_size,
            -1,
            -1,
            -1,
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder1 = self.encoder1(images)

        encoder2 = self.encoder2(
            self.pool(encoder1)
        )

        encoder3 = self.encoder3(
            self.pool(encoder2)
        )

        center = self.bottleneck(
            self.pool(encoder3)
        )

        decoder3 = self.upsample_to(
            center,
            encoder3,
        )

        decoder3 = self.decoder3(
            torch.cat(
                [decoder3, encoder3],
                dim=1,
            )
        )

        decoder2 = self.upsample_to(
            decoder3,
            encoder2,
        )

        decoder2 = self.decoder2(
            torch.cat(
                [decoder2, encoder2],
                dim=1,
            )
        )

        decoder1 = self.upsample_to(
            decoder2,
            encoder1,
        )

        decoder1 = self.decoder1(
            torch.cat(
                [decoder1, encoder1],
                dim=1,
            )
        )

        binary_logits = self.binary_output_layer(
            decoder1
        )

        coordinates = self.create_coordinate_channels(
            decoder1
        )

        coordinate_aware_features = torch.cat(
            [decoder1, coordinates],
            dim=1,
        )

        level_logits = self.level_output_layer(
            coordinate_aware_features
        )

        return binary_logits, level_logits


def load_binary_pretraining(
    model: SpiderDiscMultitaskUNet,
    checkpoint_path: str | Path,
    device: torch.device,
) -> None:
    """
    Transfer the existing binary U-Net into the multitask model.

    The original output_layer is renamed binary_output_layer.
    The coordinate-aware level head remains randomly initialized.
    """
    checkpoint_path = Path(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        old_state = checkpoint[
            "model_state_dict"
        ]
    else:
        old_state = checkpoint

    if not isinstance(old_state, dict):
        raise TypeError(
            "Checkpoint does not contain a valid state dictionary"
        )

    transferred_state = {}

    for key, value in old_state.items():
        if key == "output_layer.weight":
            transferred_state[
                "binary_output_layer.weight"
            ] = value

        elif key == "output_layer.bias":
            transferred_state[
                "binary_output_layer.bias"
            ] = value

        else:
            transferred_state[key] = value

    incompatible = model.load_state_dict(
        transferred_state,
        strict=False,
    )

    unexpected_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(
            "level_output_layer."
        )
    ]

    if unexpected_missing:
        raise ValueError(
            "Unexpected missing parameters: "
            f"{unexpected_missing}"
        )

    if incompatible.unexpected_keys:
        raise ValueError(
            "Unexpected checkpoint parameters: "
            f"{incompatible.unexpected_keys}"
        )

    print(
        "Loaded binary U-Net checkpoint:",
        checkpoint_path,
    )

    print(
        "Transferred tensors:",
        len(transferred_state),
    )

    print(
        "New coordinate-aware level-head tensors:",
        incompatible.missing_keys,
    )