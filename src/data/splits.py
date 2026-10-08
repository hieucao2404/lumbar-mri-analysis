import json
from pathlib import Path

import torch

def create_split_indices(
    number_of_samples: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, list[int]]:
    """
    Create deterministic, non-overlapping dataset indices.

    The remaining samples after the training and validation splits
    are assigned to the test split.
    """
    if number_of_samples <= 0:
        raise ValueError("number_of_samples must be positive")

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be between 0 and 1"
        )

    if train_fraction + validation_fraction >= 1.0:
        raise ValueError(
            "Training and validation fractions must total less than 1"
        )

    generator = torch.Generator().manual_seed(seed)

    shuffled_indices = torch.randperm(
        number_of_samples,
        generator=generator,
    ).tolist()

    train_count = int(number_of_samples * train_fraction)
    validation_count = int(
        number_of_samples * validation_fraction
    )
    train_end = train_count
    validation_end = train_end + validation_count

    return {
        "train": shuffled_indices[:train_end],
        "validation": shuffled_indices[
            train_end:validation_end
        ],
        "test": shuffled_indices[validation_end:],
    }

def save_split_manifest(
    splits: dict[str, list[int]],
    sample_names: list[str],
    output_path: str | Path,
    seed: int,
) -> None:
    """
    Save patient/sample names rather than only numeric indices
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "seed": seed,
        "splits": {
            split_name: [
                sample_names[index]
                for index in indices
            ]
            for split_name, indices in splits.items()
        },
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(manifest,file, indent=2)