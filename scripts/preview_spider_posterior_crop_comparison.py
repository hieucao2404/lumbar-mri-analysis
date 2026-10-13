import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(path: Path, subset: str) -> list[dict]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    return [
        row for row in rows
        if row["subset"] == subset
        and int(row["disc_herniation"]) == 1
    ]


def make_sheet(
    rows: list[dict],
    output_path: Path,
    title: str,
) -> None:
    if not rows:
        raise ValueError(f"No rows to display: {title}")

    figure, axes = plt.subplots(
        len(rows),
        2,
        figsize=(7, 3.2 * len(rows)),
        squeeze=False,
    )

    for index, row in enumerate(rows):
        old_crop = np.load(
            row["original_crop_path"],
            allow_pickle=False,
        )
        new_crop = np.load(
            row["crop_path"],
            allow_pickle=False,
        )

        if old_crop.shape != (160, 160):
            raise ValueError(
                f"Unexpected old crop shape: {old_crop.shape}"
            )

        if new_crop.shape != (160, 160):
            raise ValueError(
                f"Unexpected new crop shape: {new_crop.shape}"
            )

        same_slice = bool(
            int(row["same_slice_as_original_crop"])
        )

        label = (
            f"{row['series_name']} IVD {row['ivd_label']}\n"
            f"bulge={row['disc_bulging']} "
            f"hernia={row['disc_herniation']}\n"
            f"old slice={row['original_crop_slice']} "
            f"pipeline slice={row['selected_slice']} "
            f"same={same_slice}"
        )

        axes[index, 0].imshow(
            old_crop,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[index, 0].set_title("Original crop\n" + label)

        axes[index, 1].imshow(
            new_crop,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[index, 1].set_title("Wider pipeline-slice crop")

        for axis in axes[index]:
            axis.axis("off")

    figure.suptitle(title, fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.99))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    print("Saved:", output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/processed/spider_posterior_crops/crops.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/spider_posterior_crop_comparison"
        ),
    )
    parser.add_argument(
        "--training-count",
        type=int,
        default=12,
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    training = load_rows(args.manifest, "training")
    validation = load_rows(args.manifest, "validation")

    rng = random.Random(args.seed)
    rng.shuffle(training)

    # Pick at most one disc per training patient for variety.
    selected_training = []
    selected_patients = set()

    for row in training:
        patient = row["patient_id"]

        if patient in selected_patients:
            continue

        selected_training.append(row)
        selected_patients.add(patient)

        if len(selected_training) >= args.training_count:
            break

    validation.sort(
        key=lambda row: (
            int(row["patient_id"]),
            int(row["ivd_label"]),
        )
    )

    print("Training positives:", len(training))
    print("Validation positives:", len(validation))
    print(
        "Selected training examples:",
        len(selected_training),
    )
    print(
        "Validation examples with matching old/new slice:",
        sum(
            int(row["same_slice_as_original_crop"])
            for row in validation
        ),
        "/",
        len(validation),
    )

    make_sheet(
        selected_training,
        args.output_dir / "training_comparison.png",
        "Herniation-positive training discs",
    )

    make_sheet(
        validation,
        args.output_dir / "validation_comparison.png",
        "Herniation-positive validation discs",
    )


if __name__ == "__main__":
    main()