"""Run fixed SPIDER abnormal-disc models on external LSS sagittal T2 series.

LSS provides multi-slice MRI and anatomy/foraminal-stenosis annotations, but no
disc bulging or herniation ground truth. This script reports inference coverage
and score distributions only; it does not compute AP or confusion counts.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pydicom
import torch

from scripts.evaluate_mendeley_multislice_transfer import load_classifiers, score_stack
from src.inference.herniation_levels import (
    LEVELS,
    LOCALIZATION_CHECKPOINT,
    level_bands,
    load_model,
    localize_levels,
    make_stack,
)
from src.inference.mri_slice_selector import select_sagittal_slice


DEFAULT_SOURCE = Path("data/raw/lss_mri_dataset/DICOM")
FIELDS = (
    "patient_id", "level", "status", "series_slices", "selected_slice",
    "guarded_switch", "disc_box_xyxy", "attention_score", "attention_flag",
    "lstm_attention_score", "lstm_attention_flag", "error",
)


def load_volume(folder: Path) -> tuple[np.ndarray, int]:
    slices = []
    skipped_missing_tags = 0
    for filename in sorted(folder.glob("*.dcm")):
        dcm = pydicom.dcmread(filename)
        image = dcm.pixel_array
        if ("ImageOrientationPatient" not in dcm
                or "ImagePositionPatient" not in dcm):
            skipped_missing_tags += 1
            continue
        orientation = np.asarray(dcm.ImageOrientationPatient, dtype=float)
        if image.ndim != 2 or abs(orientation[5]) < 0.9:
            raise ValueError(f"Not an upright sagittal image: {filename}")
        if orientation[5] > 0:
            image = image[::-1]
        slices.append((float(dcm.ImagePositionPatient[0]), image))
    if len(slices) < 8:
        raise ValueError(f"Fewer than eight slices: {folder}")
    slices.sort(key=lambda item: item[0])
    shape = slices[0][1].shape
    if any(image.shape != shape for _, image in slices):
        raise ValueError(f"Mixed image shapes: {folder}")
    if any(b[0] <= a[0] for a, b in zip(slices, slices[1:])):
        raise ValueError(f"Duplicate sagittal positions: {folder}")
    return np.stack([image for _, image in slices], axis=-1), skipped_missing_tags


def summarize(rows: list[dict]) -> dict:
    available = [row for row in rows if row["status"] == "scored"]
    result = {"discs_expected": len(rows), "discs_scored": len(available),
              "disc_coverage": len(available) / len(rows) if rows else None}
    for name in ("attention", "lstm_attention"):
        scores = np.asarray([float(row[f"{name}_score"]) for row in available])
        flags = sum(int(row[f"{name}_flag"]) for row in available)
        result[name] = {
            "scores_n": len(scores), "flags": flags,
            "flag_fraction": flags / len(scores) if len(scores) else None,
            "score_quantiles_05_25_50_75_95": np.quantile(
                scores, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist()
            if len(scores) else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0,
                        help="Pilot with this many patient IDs; 0 means all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.limit < 0:
        raise ValueError("limit must be nonnegative")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)

    folders = sorted(p for p in args.source.iterdir() if p.is_dir())
    if args.limit:
        folders = folders[:args.limit]
    localizer = load_model(LOCALIZATION_CHECKPOINT, device)
    models = load_classifiers(device)
    rows = []
    counters = Counter()
    for position, folder in enumerate(folders, 1):
        patient = folder.name
        common = {"patient_id": patient, "series_slices": "",
                  "selected_slice": "", "guarded_switch": "",
                  "disc_box_xyxy": "", "attention_score": "",
                  "attention_flag": "", "lstm_attention_score": "",
                  "lstm_attention_flag": "", "error": ""}
        try:
            volume, skipped = load_volume(folder)
            counters["slices_skipped_missing_geometry_tags"] += skipped
            record = {"series_name": f"lss_{patient}"}
            selected, image, selection = select_sagittal_slice(
                record, volume, localizer, device,
                middle_window=2, selection_mode="guarded_confidence")
            boxes, decoder = localize_levels(
                record, volume, localizer, device, localization_image=image)
            bands = level_bands(boxes, volume.shape[0])
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            counters["geometry_failed_patients"] += 1
            for level in LEVELS:
                rows.append({**common, "level": level, "status": "geometry_failed",
                             "error": str(error)})
            continue
        common.update(series_slices=volume.shape[-1], selected_slice=selected,
                      guarded_switch=int(selection["guarded_switch"]))
        counters[f"decoder:{decoder['source']}"] += 1
        for level in LEVELS:
            if level not in boxes:
                counters["missing_disc_boxes"] += 1
                rows.append({**common, "level": level, "status": "missing_box"})
                continue
            try:
                stack = make_stack(volume, selected, boxes[level]["box"],
                                   crop_scale=1.5, posterior_shift=0.25,
                                   band=bands[level])
                prediction = score_stack(stack, models, device)
            except (ValueError, RuntimeError) as error:
                counters["crop_or_model_failures"] += 1
                rows.append({**common, "level": level, "status": "scoring_failed",
                             "disc_box_xyxy": json.dumps(boxes[level]["box"]),
                             "error": str(error)})
                continue
            rows.append({**common, "level": level, "status": "scored",
                         "disc_box_xyxy": json.dumps(boxes[level]["box"]),
                         **prediction})
        if position % 25 == 0 or position == len(folders):
            print(f"patients={position}/{len(folders)} "
                  f"scored_discs={sum(r['status'] == 'scored' for r in rows)}",
                  flush=True)

    args.output.mkdir(parents=True)
    with (args.output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "dataset": "LSS-MRI-AISSLab sagittal T2 DICOM",
        "patients": len(folders),
        "patients_with_scored_disc": len({r["patient_id"] for r in rows
                                          if r["status"] == "scored"}),
        "counters": dict(counters),
        "overall": summarize(rows),
        "by_level": {level: summarize([r for r in rows if r["level"] == level])
                     for level in LEVELS},
        "thresholds_from_spider_training_oof": {
            name: threshold for name, (_, threshold) in models.items()},
        "interpretation": (
            "Inference and localization coverage only. LSS has no disc-level "
            "bulging/herniation labels, so AP, TP, FP, sensitivity, and "
            "precision cannot be computed."
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"patients": report["patients"],
                      "overall": report["overall"],
                      "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
