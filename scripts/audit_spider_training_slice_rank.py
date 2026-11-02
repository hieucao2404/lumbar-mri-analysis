"""Compare current and confidence-based slice ranking on training scans only."""

from collections import Counter
from pathlib import Path
import argparse
import numpy as np
import torch

from scripts.evaluate_spider_abnormal_review_pipeline import box_iou
from scripts.export_spider_disc_localization import letterbox_pair, normalize_slice
from scripts.export_spider_herniation_stacks_v3 import load_volume
from src.inference.herniation_levels import (
    LOCALIZATION_CHECKPOINT, level_bands, load_model,
    load_series_volume, localize_levels, read_localization_records,
)
from src.inference.mri_slice_selector import select_sagittal_slice

MASK_VALUES = {
    "L1-L2": 205, "L2-L3": 204, "L3-L4": 203,
    "L4-L5": 202, "L5-S1": 201,
}


def correct_boxes(mask_slice, boxes):
    references = {}
    for level, value in MASK_VALUES.items():
        y, x = np.where(mask_slice == value)
        if len(x):
            references[level] = [x.min(), y.min(), x.max() + 1, y.max() + 1]

    correct = 0
    for level, item in boxes.items():
        if level not in references:
            continue
        box = item["box"]
        iou = box_iou(box, references[level])
        best = max(
            references,
            key=lambda other: box_iou(box, references[other]),
        )
        correct += iou >= 0.5 and best == level
    return correct


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=("training", "validation"), required=True
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    records = read_localization_records()
    model = load_model(LOCALIZATION_CHECKPOINT, device)
    totals = Counter()
    changes = []

    for series, record in sorted(records.items()):
        if record["subset"] !=  args.split:
            continue

        volume = load_series_volume(series)
        mask = load_volume(
            Path("data/raw/spider/masks") / f"{series}.mha"
        )
        current_slice, _, selection = select_sagittal_slice(
            record, volume, model, device, middle_window=2
        )

        candidates = []
        for item in selection["candidates"]:
            index = item["slice"]
            raw = volume[:, :, index]
            image, _ = letterbox_pair(
                normalize_slice(raw),
                np.zeros(raw.shape, dtype=np.uint8),
                256,
            )
            boxes, decoder = localize_levels(
                record, volume, model, device,
                localization_image=image,
            )
            try:
                level_bands(boxes, volume.shape[0])
            except ValueError:
                boxes = {}

            minimum_probability = min(
                (box["level_model_probability"] for box in boxes.values()),
                default=0.0,
            )
            candidates.append({
                "slice": index,
                "boxes": boxes,
                "minimum_probability": minimum_probability,
                "correct": correct_boxes(mask[:, :, index], boxes),
                "components": item["components"],
                "area": item["area"],
                "decoder_source": decoder["source"],
            })

        current = next(
            item for item in candidates
            if item["slice"] == current_slice
        )
        confidence = max(
            candidates,
            key=lambda item: (
                len(item["boxes"]) == 5,
                len(item["boxes"]),
                item["minimum_probability"],
                item["components"],
                item["area"],
            ),
        )

        for margin in (0.02, 0.05, 0.10):
            confidence_gain = (
                confidence["minimum_probability"]
                - current["minimum_probability"]
            )
            decoder_ok = (
                confidence["decoder_source"] == "fused_sequence"
                or current["decoder_source"] != "fused_sequence"
            )
            switch = (
                len(confidence["boxes"]) == 5
                and confidence_gain >= margin
                and decoder_ok
            )
            guarded = confidence if switch else current

            totals[f"guarded_{margin}_correct"] += guarded["correct"]
            totals[f"guarded_{margin}_switches"] += switch
            totals[f"guarded_{margin}_improved"] += (
                guarded["correct"] > current["correct"]
            )
            totals[f"guarded_{margin}_worsened"] += (
                guarded["correct"] < current["correct"]
            )

        totals["scans"] += 1
        totals["current_correct"] += current["correct"]
        totals["confidence_correct"] += confidence["correct"]
        if confidence["slice"] != current_slice:
            totals["slice_changed"] += 1
        if confidence["correct"] > current["correct"]:
            totals["improved_scans"] += 1
        elif confidence["correct"] < current["correct"]:
            totals["worsened_scans"] += 1

        if confidence["correct"] != current["correct"]:
            changes.append((
                series, current_slice, current["correct"],
                confidence["slice"], confidence["correct"],
            ))

    for margin in (0.02, 0.05, 0.10):
        print(
            f"Guarded margin {margin}:",
            "correct", totals[f"guarded_{margin}_correct"],
            "switches", totals[f"guarded_{margin}_switches"],
            "improved", totals[f"guarded_{margin}_improved"],
            "worsened", totals[f"guarded_{margin}_worsened"],
        )

    print("Device:", device)
    print("Scans:", totals["scans"])
    print("Current correct boxes:", totals["current_correct"])
    print("Confidence-ranked correct boxes:", totals["confidence_correct"])
    print("Slices changed:", totals["slice_changed"])
    print("Scans improved:", totals["improved_scans"])
    print("Scans worsened:", totals["worsened_scans"])
    print("Changed geometry outcomes:")
    for item in changes:
        print(*item)


if __name__ == "__main__":
    main()