"""Frozen VGG16 encoder shared across eight sagittal disc slices."""

import torch
from torch import nn
from torchvision.models import VGG16_Weights, vgg16


class VGG16AbnormalDisc(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = vgg16(weights=weights)

        # Keep the convolutional encoder; discard VGG16's large
        # ImageNet classification layers.
        self.slice_encoder = backbone.features
        self.slice_encoder.requires_grad_(False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reset_head()

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def reset_head(self):
        self.slice_head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

    @torch.no_grad()
    def encode(self, stacks: torch.Tensor) -> torch.Tensor:
        if stacks.ndim != 5 or stacks.shape[1:] != (8, 1, 160, 160):
            raise ValueError(
                "Expected [batch, 8, 1, 160, 160], "
                f"received {tuple(stacks.shape)}"
            )

        batch = stacks.shape[0]
        images = stacks.reshape(batch * 8, 1, 160, 160)
        images = images.expand(-1, 3, -1, -1)
        images = (images - self.mean) / self.std

        features = self.pool(self.slice_encoder(images)).flatten(1)
        return features.reshape(batch, 8, 512)

    def logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        slice_logits = self.slice_head(features).squeeze(-1)
        return slice_logits.topk(k=2, dim=1).values.mean(dim=1)

    def forward(self, stacks: torch.Tensor) -> torch.Tensor:
        return self.logits_from_features(self.encode(stacks))