from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.spider_disc_level_dataset import (
    SpiderDiscLevelDataset,
)
from src.models.spider_disc_unet import (
    SpiderDiscUNet,
)


MANIFEST = Path(
    "data/processed/spider_disc_localization/"
    "manifest.csv"
)

NUMBER_OF_CLASSES = 6
IGNORE_INDEX = 255


def multiclass_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate soft Dice loss for classes 1 through 5.

    Background class 0 is excluded.
    Target value 255 is ignored.
    """

    if logits.ndim != 4:
        raise ValueError(
            f"Expected four-dimensional logits, "
            f"got {logits.shape}"
        )

    if targets.ndim != 3:
        raise ValueError(
            f"Expected three-dimensional targets, "
            f"got {targets.shape}"
        )

    valid_pixels = targets != IGNORE_INDEX

    safe_targets = targets.clone()
    safe_targets[~valid_pixels] = 0

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    one_hot = F.one_hot(
        safe_targets,
        num_classes=NUMBER_OF_CLASSES,
    )

    one_hot = one_hot.permute(
        0,
        3,
        1,
        2,
    ).float()

    valid_mask = valid_pixels.unsqueeze(1).float()

    probabilities = probabilities * valid_mask
    one_hot = one_hot * valid_mask

    # Exclude background from Dice.
    probabilities = probabilities[:, 1:]
    one_hot = one_hot[:, 1:]

    dimensions = (0, 2, 3)

    intersection = (
        probabilities * one_hot
    ).sum(dim=dimensions)

    probability_total = probabilities.sum(
        dim=dimensions
    )

    target_total = one_hot.sum(
        dim=dimensions
    )

    dice_per_class = (
        2.0 * intersection + 1.0
    ) / (
        probability_total
        + target_total
        + 1.0
    )

    return 1.0 - dice_per_class.mean()


def main() -> None:
    dataset = SpiderDiscLevelDataset(
        MANIFEST,
        subset="training",
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    images, targets = next(iter(loader))

    model = SpiderDiscUNet(
        output_channels=NUMBER_OF_CLASSES,
    )

    logits = model(images)

    expected_output_shape = (
        images.shape[0],
        NUMBER_OF_CLASSES,
        images.shape[2],
        images.shape[3],
    )

    if logits.shape != expected_output_shape:
        raise ValueError(
            f"Expected output shape "
            f"{expected_output_shape}, "
            f"got {tuple(logits.shape)}"
        )

    # Background receives a lower weight because it greatly
    # outnumbers every disc-level class.
    class_weights = torch.tensor(
        [
            0.05,  # Background
            1.00,  # L1-L2
            1.00,  # L2-L3
            1.00,  # L3-L4
            1.00,  # L4-L5
            1.00,  # L5-S1
        ],
        dtype=torch.float32,
    )

    cross_entropy = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=IGNORE_INDEX,
    )

    ce_loss = cross_entropy(
        logits,
        targets,
    )

    dice_loss = multiclass_dice_loss(
        logits,
        targets,
    )

    total_loss = ce_loss + dice_loss

    total_loss.backward()

    gradient_is_finite = all(
        parameter.grad is None
        or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    predictions = logits.argmax(dim=1)

    print("Images:", images.shape)
    print("Targets:", targets.shape)
    print("Logits:", logits.shape)
    print("Predictions:", predictions.shape)

    print()
    print(
        "Target values:",
        sorted(
            int(value)
            for value in torch.unique(targets)
        ),
    )

    print(
        "Predicted values:",
        sorted(
            int(value)
            for value in torch.unique(predictions)
        ),
    )

    print()
    print(
        "Cross-entropy loss:",
        round(float(ce_loss.item()), 4),
    )

    print(
        "Dice loss:",
        round(float(dice_loss.item()), 4),
    )

    print(
        "Total loss:",
        round(float(total_loss.item()), 4),
    )

    print(
        "Trainable parameters:",
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )

    print(
        "Finite gradients:",
        gradient_is_finite,
    )

    assert logits.shape == (
        2,
        6,
        256,
        256,
    )

    assert predictions.shape == (
        2,
        256,
        256,
    )

    assert torch.isfinite(total_loss)
    assert gradient_is_finite

    print()
    print(
        "Spider disc-level model smoke test passed"
    )


if __name__ == "__main__":
    main()