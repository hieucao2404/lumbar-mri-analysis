import torch
from torch import nn

from src.models.trainable_vgg16_disc import TrainableVGG16Disc


class AttentionVGG16Disc(TrainableVGG16Disc):
    def __init__(self, pretrained=True, num_classes=1):
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        self.slice_attention = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def logits_from_features(self, features):
        # features: [batch, 8, 512]
        attention_scores = self.slice_attention(features).squeeze(-1)
        weights = torch.softmax(attention_scores.float(), dim=1)
        slice_logits = self.slice_head(features).float()
        return (weights.unsqueeze(-1) * slice_logits).sum(dim=1).squeeze(-1)
