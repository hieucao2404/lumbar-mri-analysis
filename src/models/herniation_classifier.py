import torch
from torch import nn


class HerniationStackClassifier(nn.Module):
    """
    Input:
        [batch, slices, 1, 160, 160]

    Output:
        One raw herniation logit per disc.
    """

    def __init__(self) -> None:
        super().__init__()

        self.slice_encoder = nn.Sequential(
            nn.Conv2d(1, 24, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 24),
            nn.ReLU(inplace=True),

            nn.Conv2d(24, 48, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 48),
            nn.ReLU(inplace=True),

            nn.Conv2d(48, 96, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 128, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )

        self.slice_head = nn.Linear(128, 1)

    def forward(self, stacks: torch.Tensor) -> torch.Tensor:
        if stacks.ndim != 5:
            raise ValueError(
                "Expected [batch, slices, channels, height, width], "
                f"received {tuple(stacks.shape)}"
            )

        batch, slices, channels, height, width = stacks.shape

        if channels != 1:
            raise ValueError(f"Expected one MRI channel, got {channels}")

        images = stacks.reshape(
            batch * slices,
            channels,
            height,
            width,
        )

        features = self.slice_encoder(images).flatten(1)
        slice_logits = self.slice_head(features).reshape(batch, slices)

        # A positive finding may be visible on only a few slices.
        number_to_pool = min(2, slices)
        return slice_logits.topk(
            k=number_to_pool,
            dim=1,
        ).values.mean(dim=1)