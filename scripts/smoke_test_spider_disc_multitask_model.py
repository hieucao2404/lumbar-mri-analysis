import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.data.spider_disc_multitask_dataset import (
    SpiderDiscMultitaskDataset,
)
from src.models.spider_disc_multitask_unet import (
    SpiderDiscMultitaskUNet,
    load_binary_pretraining,
)


MANIFEST = (
    "data/processed/spider_disc_localization/"
    "manifest.csv"
)

BINARY_CHECKPOINT = (
    "outputs/spider_disc_localization_softdice_v1/"
    "best_model.pt"
)


def binary_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)

    dimensions = (1, 2, 3)

    intersection = (
        probabilities * targets
    ).sum(dim=dimensions)

    total = (
        probabilities.sum(dim=dimensions)
        + targets.sum(dim=dimensions)
    )

    dice = (
        2.0 * intersection + 1.0
    ) / (
        total + 1.0
    )

    return 1.0 - dice.mean()


def create_level_targets(
    level_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Convert stored masks into level-head targets.

    Stored:
        0   = background
        1-5 = lumbar levels
        255 = upper discs

    Training targets:
        -100 = ignored background
        0-4  = lumbar levels
        5    = upper discs
    """
    targets = torch.full_like(
        level_mask,
        fill_value=-100,
    )

    lumbar_pixels = (
        (level_mask >= 1)
        & (level_mask <= 5)
    )

    targets[lumbar_pixels] = (
        level_mask[lumbar_pixels] - 1
    )

    targets[level_mask == 255] = 5

    return targets


def main() -> None:
    dataset = SpiderDiscMultitaskDataset(
        MANIFEST,
        subset="training",
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    images, binary_targets, level_masks = next(
        iter(loader)
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SpiderDiscMultitaskUNet().to(device)

    load_binary_pretraining(
        model,
        BINARY_CHECKPOINT,
        device,
    )

    images = images.to(device)
    binary_targets = binary_targets.to(device)
    level_masks = level_masks.to(device)

    level_targets = create_level_targets(
        level_masks
    )

    binary_logits, level_logits = model(images)

    binary_bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [20.0],
            device=device,
        )
    )

    binary_bce_loss = binary_bce(
        binary_logits,
        binary_targets,
    )

    binary_dice = binary_dice_loss(
        binary_logits,
        binary_targets,
    )

    binary_loss = (
        binary_bce_loss + binary_dice
    )

    level_loss = F.cross_entropy(
        level_logits,
        level_targets,
        ignore_index=-100,
    )

    total_loss = binary_loss + level_loss

    total_loss.backward()

    finite_gradients = all(
        parameter.grad is None
        or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    with torch.no_grad():
        binary_predictions = (
            torch.sigmoid(binary_logits) >= 0.5
        )

        level_predictions = level_logits.argmax(
            dim=1
        )

        supervised_level_pixels = (
            level_targets != -100
        )

        level_accuracy = (
            level_predictions[
                supervised_level_pixels
            ]
            == level_targets[
                supervised_level_pixels
            ]
        ).float().mean()

    print("Device:", device)
    print("Images:", images.shape)
    print(
        "Binary targets:",
        binary_targets.shape,
    )
    print("Level masks:", level_masks.shape)
    print(
        "Binary logits:",
        binary_logits.shape,
    )
    print(
        "Level logits:",
        level_logits.shape,
    )

    print()
    print(
        "Stored level values:",
        sorted(level_masks.unique().tolist()),
    )
    print(
        "Training level values:",
        sorted(level_targets.unique().tolist()),
    )

    print()
    print(
        "Binary BCE loss:",
        round(binary_bce_loss.item(), 4),
    )
    print(
        "Binary Dice loss:",
        round(binary_dice.item(), 4),
    )
    print(
        "Level loss:",
        round(level_loss.item(), 4),
    )
    print(
        "Total loss:",
        round(total_loss.item(), 4),
    )

    print(
        "Binary predicted fraction:",
        round(
            binary_predictions.float().mean().item(),
            4,
        ),
    )

    print(
        "Level pixel accuracy:",
        round(level_accuracy.item(), 4),
    )

    print(
        "Trainable parameters:",
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )

    print("Finite gradients:", finite_gradients)

    assert binary_logits.shape == (
        2,
        1,
        256,
        256,
    )

    assert level_logits.shape == (
        2,
        6,
        256,
        256,
    )

    assert torch.isfinite(total_loss)
    assert finite_gradients

    print()
    print("Multitask model smoke test passed")


if __name__ == "__main__":
    main()