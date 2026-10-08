from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.data.lumbar_dataset import LumbarLocalizationDataset
from src.data.splits import create_split_indices
from src.inference.lumbar_localizer import load_lumbar_localizer


CHECKPOINT_PATH = Path(
    "artifacts/lumbar_localizer_augmented/best_model.pt"
)

MARGINS = [
    0.00,
    0.05,
    0.08,
    0.10,
    0.12,
    0.15,
    0.18,
    0.20,
    0.25,
]


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center_x = boxes[:, 0]
    center_y = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]

    return torch.stack(
        [
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ],
        dim=1,
    )


def expand_boxes(
    boxes: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    expanded = boxes.clone()

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    expanded[:, 0] -= widths * margin
    expanded[:, 1] -= heights * margin
    expanded[:, 2] += widths * margin
    expanded[:, 3] += heights * margin

    return expanded.clamp(0.0, 1.0)


def target_coverage(
    crop_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
) -> torch.Tensor:
    """
    Return the fraction of each ground-truth box covered
    by its corresponding crop box.
    """
    intersection_x1 = torch.maximum(
        crop_boxes[:, 0],
        target_boxes[:, 0],
    )
    intersection_y1 = torch.maximum(
        crop_boxes[:, 1],
        target_boxes[:, 1],
    )
    intersection_x2 = torch.minimum(
        crop_boxes[:, 2],
        target_boxes[:, 2],
    )
    intersection_y2 = torch.minimum(
        crop_boxes[:, 3],
        target_boxes[:, 3],
    )

    intersection_width = (
        intersection_x2 - intersection_x1
    ).clamp(min=0.0)

    intersection_height = (
        intersection_y2 - intersection_y1
    ).clamp(min=0.0)

    intersection_area = (
        intersection_width * intersection_height
    )

    target_width = (
        target_boxes[:, 2] - target_boxes[:, 0]
    ).clamp(min=0.0)

    target_height = (
        target_boxes[:, 3] - target_boxes[:, 1]
    ).clamp(min=0.0)

    target_area = target_width * target_height

    return intersection_area / target_area.clamp(min=1e-8)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0.0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0.0)

    return widths * heights


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = LumbarLocalizationDataset(
        "data/raw/lss_mri_dataset",
        target_size=(128, 128),
    )

    splits = create_split_indices(
        number_of_samples=len(dataset),
        train_fraction=0.8,
        validation_fraction=0.1,
        seed=42,
    )

    validation_dataset = Subset(
        dataset,
        splits["validation"],
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=16,
        shuffle=False,
    )

    model = load_lumbar_localizer(
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
    )

    predictions = []
    targets = []

    with torch.inference_mode():
        for images, batch_targets in validation_loader:
            batch_predictions = model(
                images.to(device)
            ).cpu()

            predictions.append(batch_predictions)
            targets.append(batch_targets)

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    predicted_xyxy = cxcywh_to_xyxy(predictions)
    target_xyxy = cxcywh_to_xyxy(targets)

    print(f"Device: {device}")
    print(f"Validation samples: {len(validation_dataset)}")
    print()
    print(
        f"{'Margin':>8} "
        f"{'Mean coverage':>15} "
        f"{'>=95%':>8} "
        f"{'>=99%':>8} "
        f"{'Mean crop area':>16}"
    )

    for margin in MARGINS:
        crop_boxes = expand_boxes(
            predicted_xyxy,
            margin,
        )

        coverage = target_coverage(
            crop_boxes,
            target_xyxy,
        )

        crop_areas = box_area(crop_boxes)

        number_95 = int(
            (coverage >= 0.95).sum().item()
        )

        number_99 = int(
            (coverage >= 0.99).sum().item()
        )

        print(
            f"{margin:>8.2f} "
            f"{coverage.mean().item():>15.4f} "
            f"{number_95:>5}/{len(validation_dataset)} "
            f"{number_99:>5}/{len(validation_dataset)} "
            f"{crop_areas.mean().item():>16.4f}"
        )


if __name__ == "__main__":
    main()