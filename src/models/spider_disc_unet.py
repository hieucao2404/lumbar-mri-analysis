import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
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
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.layers(x)


class SpiderDiscUNet(nn.Module):
    """
    Lightweight U-Net for sagittal disc segmentation.

    output_channels=1:
        Binary disc segmentation. The output contains one logit map
        and should be used with BCEWithLogitsLoss.

    output_channels=6:
        Level-aware segmentation:

        0 = Background
        1 = L1-L2
        2 = L2-L3
        3 = L3-L4
        4 = L4-L5
        5 = L5-S1

        The output contains six logit maps and should be used with
        CrossEntropyLoss.
    """

    def __init__(
        self,
        output_channels: int = 1,
    ) -> None:
        super().__init__()

        if output_channels < 1:
            raise ValueError(
                "output_channels must be at least 1"
            )

        self.output_channels = output_channels

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

        self.output_layer = nn.Conv2d(
            16,
            output_channels,
            kernel_size=1,
        )

    @staticmethod
    def upsample_to(
        x: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        return F.interpolate(
            x,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        e1 = self.encoder1(x)
        # [B, 16, 256, 256]

        e2 = self.encoder2(
            self.pool(e1)
        )
        # [B, 32, 128, 128]

        e3 = self.encoder3(
            self.pool(e2)
        )
        # [B, 64, 64, 64]

        center = self.bottleneck(
            self.pool(e3)
        )
        # [B, 128, 32, 32]

        d3 = self.upsample_to(
            center,
            e3,
        )

        d3 = self.decoder3(
            torch.cat(
                [d3, e3],
                dim=1,
            )
        )

        d2 = self.upsample_to(
            d3,
            e2,
        )

        d2 = self.decoder2(
            torch.cat(
                [d2, e2],
                dim=1,
            )
        )

        d1 = self.upsample_to(
            d2,
            e1,
        )

        d1 = self.decoder1(
            torch.cat(
                [d1, e1],
                dim=1,
            )
        )

        # Return raw logits:
        #
        # Binary model:
        #   sigmoid is applied by the loss or inference code.
        #
        # Multiclass model:
        #   softmax is applied by the loss or inference code.
        return self.output_layer(d1)