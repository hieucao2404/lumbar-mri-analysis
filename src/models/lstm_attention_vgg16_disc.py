import torch
from torch import nn

from src.models.trainable_vgg16_disc import TrainableVGG16Disc


class LSTMAttentionVGG16Disc(TrainableVGG16Disc):
    def __init__(self, pretrained: bool = True):
        super().__init__(num_classes=1, pretrained=pretrained)

        # Classification happens after the LSTM, so the inherited
        # per-slice classification head is unused.
        self.slice_head = nn.Identity()

        self.input_norm = nn.LayerNorm(512)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.2)

        self.slice_attention = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.disc_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        # features: [batch, 8, 512] from the inherited VGG16 encoder
        sequence = self.input_norm(features)
        sequence, _ = self.lstm(sequence)  # [batch, 8, 256]
        sequence = self.dropout(sequence)

        attention_scores = self.slice_attention(sequence).squeeze(-1)
        weights = torch.softmax(attention_scores.float(), dim=1)
        pooled = (
            weights.unsqueeze(-1) * sequence.float()
        ).sum(dim=1)  # [batch, 256]

        return self.disc_head(pooled).squeeze(-1)  # [batch]