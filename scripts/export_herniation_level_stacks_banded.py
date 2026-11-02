"""Export level-specific SPIDER stacks without adjacent-disc pixels."""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_spider_multitask_components import (
    load_model,
)
from src.inference.herniation_levels import (
    LEVELS,
    LOCALIZATION_CHECKPOINT,
    load_series_volume,
    localize_levels,
    make_stack,
    read_localization_records,
    level_bands,
)

SOURCE = Path("data/processed/spider_discs.csv")
OUTPUT = Path("data/processed/herniation_level_stacks_banded_v1")
CROP_SCALE = 1.5
POSTERIOR_SHIFT = 0.25

LEVEL_BY_IVD = {
    "1": "L5-S1",
    "2": "L4-L5",
    "3": "L3-L4",
    "4": "L2-L3",
    "5": "L1-L2",
}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for banded stack export")

    with SOURCE.open(newline="") as file:
        source_rows = list(csv.DictReader(file))

    labels_by_series = {}

    for row in source_rows:
        if (
            row["series_name"].endswith("_t2")
            and row["subset"] in {"training", "validation"}
            and row["ivd_label"] in LEVEL_BY_IVD
            and row["disc_herniation"] in {"0", "1"}
        ):
            key = (row["series_name"], row["ivd_label"])
            if key in labels_by_series:
                raise ValueError(f"Duplicate disc label: {key}")
            labels_by_series[key] = row

    records = read_localization_records()
    device = torch.device("cuda")
    localization_model = load_model(
        LOCALIZATION_CHECKPOINT,
        device,
    )

    manifest_rows = []
    missing_rows = []
    counts = Counter()

    for series_name, record in sorted(records.items()):
        volume = load_series_volume(series_name)
        boxes, _ = localize_levels(
            record, volume, localization_model, device
        )
        bands = level_bands(boxes, volume.shape[0])
        selected_slice = int(record["sagittal_slice"])

        for ivd_label, level in LEVEL_BY_IVD.items():
            key = (series_name, ivd_label)
            if key not in labels_by_series:
                continue

            source_row = labels_by_series[key]
            split = source_row["subset"]
            label = source_row["disc_herniation"]

            if level not in boxes:
                missing_rows.append({
                    "patient_id": source_row["patient_id"],
                    "series_name": series_name,
                    "level": level,
                    "split": split,
                    "herniation": label,
                })
                continue

            stack = make_stack(
                volume,
                selected_slice,
                boxes[level]["box"],
                crop_scale=CROP_SCALE,
                posterior_shift=POSTERIOR_SHIFT,
                band=bands[level],
            )

            output_dir = OUTPUT / split
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = (
                output_dir / f"{series_name}_ivd{ivd_label}.npy"
            )
            np.save(output_path, stack)

            manifest_rows.append({
                "patient_id": source_row["patient_id"],
                "series_name": series_name,
                "level": level,
                "split": split,
                "herniation": label,
                "stack_path": str(output_path),
                "band_top": bands[level][0],
                "band_bottom": bands[level][1],
            })
            counts[(split, label)] += 1

    OUTPUT.mkdir(parents=True, exist_ok=True)

    manifest_path = OUTPUT / "manifest.csv"
    with manifest_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "patient_id",
                "series_name",
                "level",
                "split",
                "herniation",
                "stack_path",
                "band_top",
                "band_bottom",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    missing_path = OUTPUT / "missing_levels.csv"
    with missing_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "patient_id",
                "series_name",
                "level",
                "split",
                "herniation",
            ],
        )
        writer.writeheader()
        writer.writerows(missing_rows)

    (OUTPUT / "export_config.json").write_text(json.dumps({
        "crop_scale": CROP_SCALE,
        "posterior_shift": POSTERIOR_SHIFT,
        "vertical_band": "midpoint_between_predicted_disc_centers",
        "source_labels": str(SOURCE),
        "localization_checkpoint": str(LOCALIZATION_CHECKPOINT),
    }, indent=2) + "\n")

    print("Exported:", dict(counts))
    print("Missing predicted levels:", len(missing_rows))
    print(
        "Missing positive levels:",
        sum(int(row["herniation"]) for row in missing_rows),
    )
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
