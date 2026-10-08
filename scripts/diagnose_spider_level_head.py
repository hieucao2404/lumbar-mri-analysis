import torch
import torch.nn.functional as F
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

LEVEL_NAMES = [
    "L1-L2",
    "L2-L3",
    "L3-L4",
    "L4-L5",
    "L5-S1",
    "Upper",
]


def create_level_targets(
    level_masks: torch.Tensor,
) -> torch.Tensor:
    targets = torch.full_like(
        level_masks,
        fill_value=-100,
    )

    lumbar_pixels = (
        (level_masks >= 1)
        & (level_masks <= 5)
    )

    targets[lumbar_pixels] = (
        level_masks[lumbar_pixels] - 1
    )

    targets[level_masks == 255] = 5

    return targets


def calculate_confusion(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    valid = targets != -100

    predicted_values = predictions[valid]
    target_values = targets[valid]

    encoded = (
        target_values * 6
        + predicted_values
    )

    return torch.bincount(
        encoded,
        minlength=36,
    ).reshape(6, 6)


def main() -> None:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = SpiderDiscMultitaskDataset(
        MANIFEST,
        subset="training",
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    images, _, stored_level_masks = next(
        iter(loader)
    )

    images = images.to(device)
    stored_level_masks = stored_level_masks.to(device)

    level_targets = create_level_targets(
        stored_level_masks
    )

    model = SpiderDiscMultitaskUNet().to(device)

    load_binary_pretraining(
        model,
        BINARY_CHECKPOINT,
        device,
    )

    # Freeze the complete pretrained U-Net.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Train only the coordinate-aware level head.
    for parameter in (
        model.level_output_layer.parameters()
    ):
        parameter.requires_grad = True

    optimizer = torch.optim.Adam(
        model.level_output_layer.parameters(),
        lr=1e-2,
    )

    print("Device:", device)
    print("Images:", tuple(images.shape))
    print(
        "Supervised level pixels:",
        int((level_targets != -100).sum().item()),
    )

    print()
    print("One-batch level-head overfit diagnostic")

    for step in range(301):
        model.train()

        optimizer.zero_grad(set_to_none=True)

        _, level_logits = model(images)

        loss = F.cross_entropy(
            level_logits,
            level_targets,
            ignore_index=-100,
        )

        loss.backward()
        optimizer.step()

        if step % 25 == 0:
            model.eval()

            with torch.inference_mode():
                _, evaluation_logits = model(images)

                predictions = (
                    evaluation_logits.argmax(dim=1)
                )

                valid = level_targets != -100

                accuracy = (
                    predictions[valid]
                    == level_targets[valid]
                ).float().mean().item()

                confusion = calculate_confusion(
                    predictions,
                    level_targets,
                )

                predicted_counts = torch.bincount(
                    predictions[valid],
                    minlength=6,
                )

            print(
                f"step={step:3d} "
                f"loss={loss.item():.4f} "
                f"accuracy={accuracy:.4f} "
                f"predicted_counts="
                f"{predicted_counts.cpu().tolist()}"
            )

    model.eval()

    with torch.inference_mode():
        _, level_logits = model(images)

        predictions = level_logits.argmax(dim=1)

        confusion = calculate_confusion(
            predictions,
            level_targets,
        )

    print()
    print(
        "Final confusion matrix "
        "[true row, predicted column]:"
    )

    for class_name, row in zip(
        LEVEL_NAMES,
        confusion.cpu().tolist(),
    ):
        print(f"{class_name:<6}", row)

    final_accuracy = (
        predictions[level_targets != -100]
        == level_targets[level_targets != -100]
    ).float().mean().item()

    print()
    print(
        "Final supervised-pixel accuracy:",
        round(final_accuracy, 4),
    )

    if final_accuracy < 0.80:
        raise RuntimeError(
            "The coordinate-aware level head could not "
            "overfit one batch"
        )

    print(
        "Coordinate-aware level-head diagnostic passed"
    )


if __name__ == "__main__":
    main()