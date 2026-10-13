import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from scripts.build_spider_paired_crops import normalize_slice
from scripts.run_spider_sagittal_pipeline import (
    load_original_sagittal_slice,
)


LEVEL_NAMES = {
    1: "L5-S1",
    2: "L4-L5",
    3: "L3-L4",
    4: "L2-L3",
    5: "L1-L2",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def choose_training_examples(
    rows: list[dict],
    count: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    candidates = rows.copy()
    rng.shuffle(candidates)

    selected = []
    used_patients = set()

    for row in candidates:
        if row["patient_id"] in used_patients:
            continue

        selected.append(row)
        used_patients.add(row["patient_id"])

        if len(selected) == count:
            break

    return selected


def draw_review(
    rows: list[dict],
    localization_by_series: dict[str, dict],
    raw_root: Path,
    output_path: Path,
    title: str,
) -> None:
    if not rows:
        raise ValueError(f"No examples for {title}")

    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=(13, 3.6 * len(rows)),
        squeeze=False,
    )

    same_slice_count = 0

    for row_index, row in enumerate(rows):
        series_name = row["series_name"]
        ivd_label = int(row["ivd_label"])

        training_slice_index = int(row["sagittal_slice"])

        localization_record = localization_by_series.get(
            series_name
        )

        if localization_record is None:
            raise ValueError(
                f"No localization record for {series_name}"
            )

        localization_slice_index = int(
            localization_record["sagittal_slice"]
        )

        same_slice = (
            training_slice_index == localization_slice_index
        )

        same_slice_count += int(same_slice)

        training_slice = load_original_sagittal_slice(
            raw_root,
            series_name,
            training_slice_index,
        )

        localization_slice = load_original_sagittal_slice(
            raw_root,
            series_name,
            localization_slice_index,
        )

        training_display = normalize_slice(training_slice)
        localization_display = normalize_slice(
            localization_slice
        )

        crop_path = Path(row["crop_path"])
        crop = np.load(
            crop_path,
            allow_pickle=False,
        )

        if crop.shape != (160, 160):
            raise ValueError(
                f"{crop_path}: expected (160, 160), "
                f"got {crop.shape}"
            )

        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])

        if not (
            0 <= x1 < x2 <= training_slice.shape[1]
            and 0 <= y1 < y2 <= training_slice.shape[0]
        ):
            raise ValueError(
                f"{series_name}: disc box is outside "
                f"the training slice"
            )

        full_axis, crop_axis, localization_axis = (
            axes[row_index]
        )

        full_axis.imshow(
            training_display,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        full_axis.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                edgecolor="red",
                facecolor="none",
                linewidth=2,
            )
        )

        full_axis.set_title(
            f"{series_name} | {LEVEL_NAMES[ivd_label]}\n"
            f"Training slice {training_slice_index}"
        )

        crop_axis.imshow(
            crop,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        crop_axis.set_title(
            "Existing training crop\n"
            f"Bulge={row['disc_bulging']}, "
            f"Herniation={row['disc_herniation']}"
        )

        localization_axis.imshow(
            localization_display,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        localization_axis.set_title(
            f"Pipeline slice {localization_slice_index}\n"
            f"{'Same slice' if same_slice else 'Different slice'}"
        )

        for axis in axes[row_index]:
            axis.axis("off")

    fig.suptitle(
        f"{title}\n"
        f"Same training/pipeline slice: "
        f"{same_slice_count}/{len(rows)}",
        fontsize=15,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.965)
    )

    fig.savefig(
        output_path,
        dpi=140,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"{title}: same slice "
        f"{same_slice_count}/{len(rows)}"
    )
    print("Saved:", output_path)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--crop-manifest",
        type=Path,
        default=Path(
            "data/processed/spider_crops/crops.csv"
        ),
    )
    parser.add_argument(
        "--localization-manifest",
        type=Path,
        default=Path(
            "data/processed/spider_disc_localization/"
            "manifest.csv"
        ),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/spider"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/spider_posterior_slice_review"
        ),
    )
    parser.add_argument(
        "--training-count",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    crop_rows = read_csv(args.crop_manifest)
    localization_rows = read_csv(
        args.localization_manifest
    )

    localization_by_series = {
        row["series_name"]: row
        for row in localization_rows
    }

    if len(localization_by_series) != len(
        localization_rows
    ):
        raise ValueError(
            "Duplicate localization series names"
        )

    positive_rows = [
        row
        for row in crop_rows
        if row["series_name"].endswith("_t2")
        and int(row["ivd_label"]) in range(1, 6)
        and row["disc_herniation"] == "1"
    ]

    validation_rows = sorted(
        (
            row
            for row in positive_rows
            if row["subset"] == "validation"
        ),
        key=lambda row: (
            int(row["patient_id"]),
            int(row["ivd_label"]),
        ),
    )

    training_rows = choose_training_examples(
        [
            row
            for row in positive_rows
            if row["subset"] == "training"
        ],
        args.training_count,
        args.seed,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Validation positives:",
        len(validation_rows),
    )
    print(
        "Selected training positives:",
        len(training_rows),
    )

    draw_review(
        validation_rows,
        localization_by_series,
        args.raw_root,
        args.output_dir
        / "validation_herniation_slices.png",
        "Validation herniation-positive discs",
    )

    draw_review(
        training_rows,
        localization_by_series,
        args.raw_root,
        args.output_dir
        / "training_herniation_slices.png",
        "Training herniation-positive discs",
    )


if __name__ == "__main__":
    main()