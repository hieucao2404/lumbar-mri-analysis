"""Find whether a different sagittal slice recovers failed abnormal-disc boxes.

Expert masks are used for this audit only, never for inference.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_spider_abnormal_review_pipeline import box_iou
from scripts.export_spider_disc_localization import letterbox_pair, normalize_slice
from scripts.export_spider_herniation_stacks_v3 import load_volume
from src.inference.herniation_levels import (
    LOCALIZATION_CHECKPOINT, level_bands, load_model,
    load_series_volume, localize_levels, read_localization_records,
)

DATA = Path("data/processed/spider_bulge_herniation_mri_only_geometry_v1")
MASK_VALUES = {
    "L1-L2": 205, "L2-L3": 204, "L3-L4": 203,
    "L4-L5": 202, "L5-S1": 201,
}


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


@torch.inference_mode()
def check_slice(series, index, volume, mask, record, model, device):
    raw = volume[:, :, index]
    image, _ = letterbox_pair(
        normalize_slice(raw),
        np.zeros(raw.shape, dtype=np.uint8),
        256,
    )
    boxes, _ = localize_levels(
        record, volume, model, device, localization_image=image
    )

    try:
        level_bands(boxes, volume.shape[0])
    except ValueError:
        boxes = {}

    references = {}
    for level, value in MASK_VALUES.items():
        y, x = np.where(mask[:, :, index] == value)
        if len(x):
            references[level] = [
                x.min(), y.min(), x.max() + 1, y.max() + 1
            ]

    result = {}
    for level in MASK_VALUES:
        if level not in boxes or level not in references:
            result[level] = (False, 0.0)
            continue

        box = boxes[level]["box"]
        iou = box_iou(box, references[level])
        best_level = max(
            references,
            key=lambda other: box_iou(box, references[other]),
        )
        result[level] = (iou >= 0.5 and best_level == level, iou)

    return result


def main():
    validation = read_csv(DATA / "validation_evaluable.csv")
    audit = {
        row["series_name"]: row
        for row in read_csv(DATA / "slice_audit.csv")
        if row["split"] == "validation"
    }
    by_series = defaultdict(list)
    for row in validation:
        if row["status"] != "normal":
            by_series[row["series_name"]].append(row["level"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    records = read_localization_records()
    model = load_model(LOCALIZATION_CHECKPOINT, device)
    counts = Counter()

    for series, abnormal_levels in sorted(by_series.items()):
        volume = load_series_volume(series)
        mask = load_volume(
            Path("data/raw/spider/masks") / f"{series}.mha"
        )
        chosen = int(audit[series]["selected_slice"])
        window = {
            item["slice"]
            for item in json.loads(audit[series]["candidate_components"])
        }

        cache = {}

        def result_at(index):
            if index not in cache:
                cache[index] = check_slice(
                    series, index, volume, mask, records[series],
                    model, device,
                )
            return cache[index]

        selected = result_at(chosen)
        failed = [
            level for level in abnormal_levels
            if not selected[level][0]
        ]

        for level in abnormal_levels:
            counts["abnormal_total"] += 1
            if selected[level][0]:
                counts["selected_correct"] += 1

        if not failed:
            continue

        for index in range(volume.shape[2]):
            result_at(index)

        for level in failed:
            good_window = [
                index for index in sorted(window)
                if result_at(index)[level][0]
            ]
            good_anywhere = [
                index for index in range(volume.shape[2])
                if result_at(index)[level][0]
            ]

            if good_window:
                category = "recoverable_inside_window"
            elif good_anywhere:
                category = "recoverable_outside_window"
            else:
                category = "not_recovered_on_any_slice"

            counts[category] += 1
            print(
                series, level,
                "selected=", chosen,
                "selected_iou=", round(selected[level][1], 3),
                "category=", category,
                "recovering_slices=", good_anywhere,
            )

    print("\nSUMMARY")
    for key, value in sorted(counts.items()):
        print(key, value)


if __name__ == "__main__":
    main()