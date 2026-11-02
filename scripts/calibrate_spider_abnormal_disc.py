"""Calibrate an axial-review flag from patient-separated SPIDER training scores."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

KEY = ("patient_id", "series_name", "level")


def read_predictions(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = set(KEY) | {"status", "abnormal", "abnormal_score"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"Invalid prediction file: {path}")
    keys = [tuple(row[key] for key in KEY) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate disc in {path}")
    for row in rows:
        if row["status"] not in {"normal", "bulging_only", "herniation"}:
            raise ValueError("Invalid status")
        if row["abnormal"] != str(int(row["status"] != "normal")):
            raise ValueError("Abnormal label does not match source status")
        score = float(row["abnormal_score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid score")
    return rows


def validate(oof, training, validation):
    source = {tuple(row[key] for key in KEY): row for row in training}
    scored = {tuple(row[key] for key in KEY): row for row in oof}
    if len(source) != len(training) or set(source) != set(scored):
        raise ValueError("OOF predictions must cover each training disc exactly once")
    patients = {}
    for row in oof:
        key = tuple(row[field] for field in KEY)
        if row["status"] != source[key]["status"]:
            raise ValueError("OOF status does not match training manifest")
        patient = row["patient_id"]
        fold = row.get("fold")
        if not fold:
            raise ValueError("OOF prediction has no fold")
        if patient in patients and patients[patient] != fold:
            raise ValueError("One patient appears in multiple held-out folds")
        patients[patient] = fold
    if len(set(patients.values())) < 2:
        raise ValueError("Need at least two patient folds")
    if set(patients) & {row["patient_id"] for row in validation}:
        raise ValueError("Training and validation patients overlap")


def choose_threshold(rows, target):
    if not 0 < target <= 1:
        raise ValueError("Target sensitivity must be in (0, 1]")
    positives = sorted((float(row["abnormal_score"]) for row in rows
                        if row["abnormal"] == "1"), reverse=True)
    if not positives:
        raise ValueError("No abnormal training discs")
    return positives[math.ceil(target * len(positives) - 1e-12) - 1]


def summarize(rows, threshold):
    truth = np.array([row["abnormal"] == "1" for row in rows])
    scores = np.array([float(row["abnormal_score"]) for row in rows])
    alerts = scores >= threshold
    tp = int((alerts & truth).sum())
    fp = int((alerts & ~truth).sum())
    fn = int((~alerts & truth).sum())
    tn = int((~alerts & ~truth).sum())
    herniations = [row for row in rows if row["status"] == "herniation"]
    return {
        "discs": len(rows), "abnormal_discs": int(truth.sum()),
        "normal_discs": int((~truth).sum()), "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": tp / (tp + fn), "precision": tp / (tp + fp),
        "specificity": tn / (tn + fp),
        "disc_ap": float(average_precision_score(truth, scores)),
        "herniation_subset_flagged": sum(float(row["abnormal_score"]) >= threshold
                                          for row in herniations),
        "herniation_subset_total": len(herniations),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, default=Path(
        "data/processed/spider_bulge_herniation_banded_v1/training.csv"))
    parser.add_argument("--target-sensitivity", type=float, default=0.91)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    oof = read_predictions(args.oof)
    validation = read_predictions(args.validation)
    with args.training_manifest.open(newline="") as stream:
        training = list(csv.DictReader(stream))
    validate(oof, training, validation)
    threshold = choose_threshold(oof, args.target_sensitivity)
    report = {
        "task": "flag possible bulging or herniation for axial review",
        "target_sensitivity": args.target_sensitivity,
        "threshold": threshold,
        "calibration_source": str(args.oof),
        "calibration": summarize(oof, threshold),
        "validation_source": str(args.validation),
        "validation": summarize(validation, threshold),
        "caveat": "Validation has been reused; this is exploratory, not an independent performance claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
