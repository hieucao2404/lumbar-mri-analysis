"""Export binary-review crops using only MRI pixels for slice selection.

The localizer selects the sagittal slice and disc boxes. Expert grades are
read only after inference to attach training/evaluation labels.
"""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.inference.herniation_levels import (
    LEVELS,
    LOCALIZATION_CHECKPOINT,
    level_bands,
    load_model,
    load_series_volume,
    localize_levels,
    make_stack,
    read_localization_records,
)
from src.inference.mri_slice_selector import select_sagittal_slice


IVD_BY_LEVEL = {
    "L5-S1": "1", "L4-L5": "2", "L3-L4": "3",
    "L2-L3": "4", "L1-L2": "5",
}
FIELDS = (
    "patient_id", "series_name", "level", "split", "bulging",
    "herniation", "status", "stack_path", "sample_weight",
)
GRADES = Path("data/raw/spider/radiological_gradings.csv")


def read_grades(path: Path) -> dict[tuple[str, str], dict]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        key = (row["Patient"], row["IVD label"])
        if key in result:
            raise ValueError(f"Duplicate original grading: {key}")
        result[key] = row
    return result


def status(bulging: int, herniation: int) -> str:
    if herniation:
        return "herniation"
    if bulging:
        return "bulging_only"
    return "normal"


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def balanced_validation(rows: list[dict], seed: int) -> list[dict]:
    smallest_class = min(Counter(row["status"] for row in rows).values())
    rng = random.Random(seed)
    selected = []
    for group in ("normal", "bulging_only", "herniation"):
        candidates = [row for row in rows if row["status"] == group]
        rng.shuffle(candidates)
        patients = set()
        for row in candidates:
            if row["patient_id"] not in patients:
                selected.append(row)
                patients.add(row["patient_id"])
            if len(patients) == smallest_class:
                break
        if len(patients) != smallest_class:
            raise ValueError(f"Not enough distinct patients for {group}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--middle-window", type=int, default=2)
    parser.add_argument(
        "--selection-mode",
        choices=("component", "guarded_confidence"),
        default="component",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.middle_window < 0:
        raise ValueError("middle-window must be nonnegative")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)

    records = read_localization_records()
    grades = read_grades(GRADES)
    localizer = load_model(LOCALIZATION_CHECKPOINT, device)
    training, validation, missing, slice_audit = [], [], [], []
    args.output.mkdir(parents=True)

    for series_name, record in sorted(records.items()):
        patient = record["patient_id"]
        split = record["subset"]
        if split not in {"training", "validation"}:
            raise ValueError(f"Unknown split for {series_name}: {split}")
        volume = load_series_volume(series_name)
        chosen, image, selection = select_sagittal_slice(
            record,
            volume,
            localizer,
            device,
            middle_window=args.middle_window,
            selection_mode=args.selection_mode,
        )
        boxes, decoder = localize_levels(
            record, volume, localizer, device, localization_image=image
        )
        geometry_error = ""
        try:
            bands = level_bands(boxes, volume.shape[0])
        except ValueError as error:
            # A contradictory level order cannot define safe crop bands.
            # Preserve the series in the audit and count all five discs as
            # missing instead of silently excluding the patient.
            geometry_error = str(error)
            boxes, bands = {}, {}
        slice_audit.append({
            "patient_id": patient, "series_name": series_name, "split": split,
            "selected_slice": chosen,
            "old_mask_selected_slice": int(record["sagittal_slice"]),
            "candidate_components": json.dumps(selection["candidates"]),
            "available_levels": len(boxes),
            "decoder_source": decoder["source"],
            "geometry_error": geometry_error,
            "slice_selection": args.selection_mode,
        })

        for level in LEVELS:
            grade = grades.get((patient, IVD_BY_LEVEL[level]))
            if grade is None:
                raise ValueError(f"Missing original grading: {series_name} {level}")
            bulging = int(grade["Disc bulging"])
            herniation = int(grade["Disc herniation"])
            if bulging not in (0, 1) or herniation not in (0, 1):
                raise ValueError(f"Invalid grading: {series_name} {level}")
            row = {
                "patient_id": patient, "series_name": series_name,
                "level": level, "split": split, "bulging": bulging,
                "herniation": herniation,
                "status": status(bulging, herniation),
                "stack_path": "", "sample_weight": "",
            }
            if level not in boxes:
                missing.append({key: row[key] for key in FIELDS if key != "sample_weight"})
                continue
            stack = make_stack(
                volume, chosen, boxes[level]["box"], crop_scale=1.5,
                posterior_shift=0.25, band=bands[level]
            )
            output_path = (
                args.output / "stacks" / split
                / f"{series_name}_ivd{IVD_BY_LEVEL[level]}.npy"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, stack)
            row["stack_path"] = str(output_path)
            (training if split == "training" else validation).append(row)
        print(f"{series_name}: slice={chosen} levels={len(boxes)}", flush=True)

    balanced = balanced_validation(validation, seed=42)
    write_csv(args.output / "training.csv", training, FIELDS)
    write_csv(args.output / "validation_natural.csv", validation, FIELDS)
    write_csv(args.output / "validation_balanced.csv", balanced, FIELDS)
    write_csv(
        args.output / "missing_levels.csv", missing,
        tuple(key for key in FIELDS if key != "sample_weight"),
    )
    write_csv(
        args.output / "slice_audit.csv", slice_audit,
        ("patient_id", "series_name", "split", "selected_slice",
         "old_mask_selected_slice", "candidate_components",
         "available_levels", "decoder_source", "geometry_error",
         "slice_selection"),
    )
    report = {
        "task": "normal versus bulging or herniation",
        "slice_selection": args.selection_mode,
        "middle_window": args.middle_window,
        "localization_checkpoint": str(LOCALIZATION_CHECKPOINT),
        "original_grading_source": str(GRADES),
        "training_patients": len({r["patient_id"] for r in training}),
        "validation_patients": len({r["patient_id"] for r in validation}),
        "training": dict(Counter(r["status"] for r in training)),
        "validation": dict(Counter(r["status"] for r in validation)),
        "missing": {
            f"{split}:{label}": count
            for (split, label), count in Counter(
                (r["split"], r["status"]) for r in missing
            ).items()
        },
        "note": "Original grading labels are attached after MRI-only crop generation.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
