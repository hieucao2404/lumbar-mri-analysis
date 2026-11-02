import torch
from torch import nn

from src.models.abnormal_disc_vgg16 import VGG16AbnormalDisc


class TrainableVGG16Disc(VGG16AbnormalDisc):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(pretrained=pretrained)
        self.slice_encoder.requires_grad_(True)
        self.slice_head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    # Override the frozen model's @torch.no_grad() encode method.
    def encode(self, stacks: torch.Tensor) -> torch.Tensor:
        if stacks.ndim != 5 or stacks.shape[1:] != (8, 1, 160, 160):
            raise ValueError(f"Unexpected stack shape: {tuple(stacks.shape)}")

        batch = stacks.shape[0]
        images = stacks.reshape(batch * 8, 1, 160, 160)
        images = images.expand(-1, 3, -1, -1)
        images = (images - self.mean) / self.std
        features = self.pool(self.slice_encoder(images)).flatten(1)
        return features.reshape(batch, 8, 512)