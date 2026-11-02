"""Evaluate the complete SPIDER disc-review pipeline on reused validation cases."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.inference.abnormal_disc_review import (
    DEFAULT_CALIBRATION, DEFAULT_CHECKPOINT, load_review_classifier,
    predict_review_flags,
)
from src.inference.herniation_levels import (
    LEVELS, LOCALIZATION_CHECKPOINT, load_model, read_localization_records,
)

LEVEL_BY_IVD = {"1": "L5-S1", "2": "L4-L5", "3": "L3-L4",
                "4": "L2-L3", "5": "L1-L2"}


def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--validation-manifest", type=Path, default=Path(
        "data/processed/spider_bulge_herniation_banded_v1/validation_natural.csv"))
    parser.add_argument("--stored-predictions", type=Path, default=Path(
        "outputs/spider_abnormal_disc_cnn_fixed_v1/validation_predictions.csv"))
    parser.add_argument("--source-discs", type=Path, default=Path(
        "data/processed/spider_discs.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    manifest = read_csv(args.validation_manifest)
    stored = {(r["series_name"], r["level"]): r
              for r in read_csv(args.stored_predictions)}
    if len(stored) != len(manifest):
        raise ValueError("Stored predictions do not cover validation manifest")
    source = {}
    for row in read_csv(args.source_discs):
        if (row["subset"] == "validation" and row["series_name"].endswith("_t2")
                and row["ivd_label"] in LEVEL_BY_IVD):
            key = (row["series_name"], LEVEL_BY_IVD[row["ivd_label"]])
            source[key] = [int(row[field]) for field in ("x1", "y1", "x2", "y2")]

    records = read_localization_records()
    localizer = load_model(LOCALIZATION_CHECKPOINT, device)
    classifier, threshold = load_review_classifier(args.checkpoint, args.calibration, device)
    by_series = {}
    for row in manifest:
        by_series.setdefault(row["series_name"], []).append(row)
    results = []
    max_score_difference = 0.0
    for series_name, rows in sorted(by_series.items()):
        result = predict_review_flags(series_name, records, localizer,
                                      classifier, threshold, device)
        predicted = {item["level"]: item for item in result["levels"]}
        references = {level: source[(series_name, level)] for level in LEVELS
                      if (series_name, level) in source}
        for row in rows:
            level = row["level"]
            item = predicted[level]
            available = item["status"] == "available"
            score = item["abnormal_disc_score"] if available else None
            if available:
                difference = abs(score - float(stored[(series_name, level)]["abnormal_score"]))
                max_score_difference = max(max_score_difference, difference)
                box = item["disc_bounding_box_original"]
                same_iou = box_iou(box, references[level]) if level in references else None
                best_level = max(references, key=lambda candidate: box_iou(box, references[candidate]))
            else:
                box, same_iou, best_level = None, None, None
            results.append({
                "patient_id": row["patient_id"], "series_name": series_name,
                "level": level, "status": row["status"],
                "abnormal": int(row["status"] != "normal"),
                "available": int(available), "abnormal_score": score,
                "review_recommended": int(item["review_recommended"]) if available else "",
                "disc_box_original": json.dumps(box) if available else "",
                "same_level_box_iou": same_iou,
                "best_matching_source_level": best_level or "",
                "level_assignment_correct": int(best_level == level) if available else "",
            })
        print(f"processed {series_name}", flush=True)
    # Stored scores were produced on CUDA; CPU convolution arithmetic can
    # differ slightly while leaving every review decision unchanged.
    if max_score_difference > 1e-4:
        raise ValueError(f"Runtime and stored crop scores differ by {max_score_difference}")
    truth = np.array([r["abnormal"] == 1 for r in results])
    flags = np.array([r["review_recommended"] == 1 for r in results])
    available = [r for r in results if r["available"] == 1]
    ious = [r["same_level_box_iou"] for r in available
            if r["same_level_box_iou"] is not None]
    report = {
        "series": len(by_series), "source_discs": len(results),
        "localized_discs": len(available), "missing_levels": len(results) - len(available),
        "review_threshold": threshold,
        "true_flags": int((truth & flags).sum()),
        "false_flags": int((~truth & flags).sum()),
        "missed_abnormal_discs": int((truth & ~flags).sum()),
        "normal_discs_without_flag": int((~truth & ~flags).sum()),
        "abnormal_sensitivity_including_missing": float((truth & flags).sum() / truth.sum()),
        "flag_precision": float((truth & flags).sum() / flags.sum()) if flags.any() else 0,
        "level_assignment_correct": sum(r["level_assignment_correct"] == 1 for r in available),
        "level_assignment_evaluated": len(available),
        "disc_box_median_same_level_iou": float(np.median(ious)) if ious else None,
        "disc_boxes_iou_at_least_0_5": sum(value >= 0.5 for value in ious),
        "disc_boxes_evaluated": len(ious),
        "max_stored_score_difference": max_score_difference,
        "box_caveat": "Source disc boxes may be annotated on a different sagittal slice; these are disc-box checks, not herniation lesion localization.",
        "validation_caveat": "SPIDER validation patients have been reused in earlier experiments.",
    }
    args.output.mkdir(parents=True)
    with (args.output / "per_disc.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
