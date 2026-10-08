from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.spider_disc_level_dataset import (
    SpiderDiscLevelDataset,
)


MANIFEST = Path(
    "data/processed/spider_disc_localization/"
    "manifest.csv"
)


def count_target_pixels(
    dataset: SpiderDiscLevelDataset,
) -> Counter:
    counts = Counter()

    for _, target in dataset:
        values, value_counts = torch.unique(
            target,
            return_counts=True,
        )

        for value, count in zip(
            values.tolist(),
            value_counts.tolist(),
        ):
            counts[int(value)] += int(count)

    return counts


def main() -> None:
    training_dataset = SpiderDiscLevelDataset(
        MANIFEST,
        subset="training",
    )

    validation_dataset = SpiderDiscLevelDataset(
        MANIFEST,
        subset="validation",
    )

    training_patients = {
        row["patient_id"]
        for row in training_dataset.rows
    }

    validation_patients = {
        row["patient_id"]
        for row in validation_dataset.rows
    }

    overlap = (
        training_patients
        & validation_patients
    )

    if overlap:
        raise ValueError(
            f"Patient overlap detected: {sorted(overlap)}"
        )

    loader = DataLoader(
        training_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    images, targets = next(iter(loader))

    print("Training images:", len(training_dataset))
    print(
        "Validation images:",
        len(validation_dataset),
    )
    print(
        "Training patients:",
        len(training_patients),
    )
    print(
        "Validation patients:",
        len(validation_patients),
    )
    print("Patient overlap:", len(overlap))

    print()
    print("Image batch:", images.shape)
    print("Target batch:", targets.shape)
    print("Image dtype:", images.dtype)
    print("Target dtype:", targets.dtype)
    print(
        "Image range:",
        float(images.min()),
        float(images.max()),
    )
    print(
        "Target values:",
        sorted(
            int(value)
            for value in torch.unique(targets)
        ),
    )

    assert images.shape == (4, 1, 256, 256)
    assert targets.shape == (4, 256, 256)
    assert images.dtype == torch.float32
    assert targets.dtype == torch.int64

    allowed_values = {
        0,
        1,
        2,
        3,
        4,
        5,
        255,
    }

    observed_values = {
        int(value)
        for value in torch.unique(targets)
    }

    assert observed_values.issubset(
        allowed_values
    )

    training_counts = count_target_pixels(
        training_dataset
    )

    validation_counts = count_target_pixels(
        validation_dataset
    )

    names = {
        0: "Background",
        1: "L1-L2",
        2: "L2-L3",
        3: "L3-L4",
        4: "L4-L5",
        5: "L5-S1",
        255: "Ignore",
    }

    print()
    print("Training pixel counts:")

    for class_id in sorted(training_counts):
        print(
            f"  {class_id:3d} "
            f"{names[class_id]:<12} "
            f"{training_counts[class_id]}"
        )

    print()
    print("Validation pixel counts:")

    for class_id in sorted(validation_counts):
        print(
            f"  {class_id:3d} "
            f"{names[class_id]:<12} "
            f"{validation_counts[class_id]}"
        )

    for class_id in range(1, 6):
        if training_counts[class_id] == 0:
            raise ValueError(
                f"Training class {class_id} has no pixels"
            )

        if validation_counts[class_id] == 0:
            raise ValueError(
                f"Validation class {class_id} has no pixels"
            )

    print()
    print(
        "Spider disc-level dataset smoke test passed"
    )


if __name__ == "__main__":
    main()