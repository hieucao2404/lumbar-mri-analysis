"""Evaluate fixed SPIDER abnormal-disc models on original multi-slice Mendeley MRI.

The original DICOM archive supplies actual sagittal T2 slices. LSMA-PQR's
structured notes supply report-derived None/Diffuse Bulge labels for two lower
levels. Localization and crop generation use MRI pixels only.
"""

import argparse
import csv
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pydicom
import torch
from sklearn.metrics import average_precision_score

from src.inference.herniation_levels import (
    LOCALIZATION_CHECKPOINT,
    level_bands,
    load_model,
    localize_levels,
    make_stack,
)
from src.inference.mri_slice_selector import select_sagittal_slice
from src.models.attention_vgg16_disc import AttentionVGG16Disc
from src.models.lstm_attention_vgg16_disc import LSTMAttentionVGG16Disc


DEFAULT_ARCHIVE = Path("data/raw/mendeley_lumbar/MRI_Data.zip")
DEFAULT_NOTES = Path(
    "data/raw/LSMA-PQR Lumbar Spine Multi-view Annotations with/"
    "LSMA-PQR Lumbar Spine Multi-view Annotations with/structured_notes.csv"
)
MODELS = {
    "attention": (
        AttentionVGG16Disc,
        "abnormal_disc_vgg16_slice_attention",
        Path("outputs/spider_abnormal_vgg16_attention_guarded_v1"),
    ),
    "lstm_attention": (
        LSTMAttentionVGG16Disc,
        "abnormal_disc_vgg16_bilstm_attention",
        Path("outputs/spider_abnormal_vgg16_lstm_attention_guarded_v1"),
    ),
}
LEVEL_COLUMNS = {"L4-L5": "L4_L5_Disc_Type", "L5-S1": "L5_S1_Disc_Type"}
ELIGIBLE_LABELS = {"None": 0, "Diffuse Bulge": 1}


def scan_series(archive: zipfile.ZipFile) -> dict[int, list[tuple[str, list[str]]]]:
    grouped = defaultdict(list)
    for filename in archive.namelist():
        parts = filename.split("/")
        if (len(parts) < 5 or not filename.endswith(".ima")
                or not parts[-2].startswith("T2_TSE_SAG")):
            continue
        grouped[(int(parts[1]), "/".join(parts[:-1]))].append(filename)
    by_patient = defaultdict(list)
    for (patient, folder), filenames in grouped.items():
        if len(filenames) >= 8:
            by_patient[patient].append((folder, filenames))
    for studies in by_patient.values():
        studies.sort(key=lambda item: (-len(item[1]), item[0]))
    return dict(by_patient)


def read_volume(archive: zipfile.ZipFile, filenames: list[str]) -> np.ndarray:
    slices = []
    for filename in filenames:
        dcm = pydicom.dcmread(io.BytesIO(archive.read(filename)))
        image = dcm.pixel_array
        orientation = np.asarray(dcm.ImageOrientationPatient, dtype=float)
        if image.ndim != 2 or abs(orientation[5]) < 0.9:
            raise ValueError(f"Not an upright sagittal image: {filename}")
        if orientation[5] > 0:
            image = image[::-1]
        # Sort by patient left-right coordinate. SPIDER's LPS-oriented stack
        # also increases along the left-right (patient x) direction.
        position = float(dcm.ImagePositionPatient[0])
        slices.append((position, image))
    slices.sort(key=lambda item: item[0])
    shape = slices[0][1].shape
    if any(image.shape != shape for _, image in slices):
        raise ValueError("Mixed image shapes within sagittal series")
    if any(b[0] <= a[0] for a, b in zip(slices, slices[1:])):
        raise ValueError("Duplicate or unordered sagittal slice positions")
    return np.stack([image for _, image in slices], axis=-1)


def read_labels(path: Path) -> dict[int, dict]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        patient = int(row["Patient_ID"])
        if patient in result:
            raise ValueError(f"Duplicate patient label {patient}")
        result[patient] = row
    return result


def load_classifiers(device: torch.device):
    loaded = {}
    for name, (model_type, expected_kind, directory) in MODELS.items():
        checkpoint_path = directory / "fixed.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("model_kind") != expected_kind:
            raise ValueError(f"Wrong model kind in {checkpoint_path}")
        calibration = json.loads((directory / "calibration.json").read_text())
        if calibration["calibration_source"] != str(directory / "oof_predictions.csv"):
            raise ValueError(f"Calibration does not match {directory}")
        model = model_type(pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        loaded[name] = (model, float(calibration["threshold"]))
    return loaded


@torch.inference_mode()
def score_stack(stack: np.ndarray, models: dict, device: torch.device) -> dict:
    tensor = torch.from_numpy(stack[None].astype(np.float32) / 255.0).to(device)
    result = {}
    for name, (model, threshold) in models.items():
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            score = float(model(tensor).sigmoid().float().item())
        result[f"{name}_score"] = score
        result[f"{name}_flag"] = int(score >= threshold)
    return result


def summarize(rows: list[dict], score_column: str, flag_column: str) -> dict:
    truth = np.array([int(row["truth"]) for row in rows], dtype=bool)
    scores = np.array([float(row[score_column]) for row in rows])
    flags = np.array([int(row[flag_column]) for row in rows], dtype=bool)
    tp = int((truth & flags).sum())
    fp = int((~truth & flags).sum())
    fn = int((truth & ~flags).sum())
    tn = int((~truth & ~flags).sum())
    return {
        "discs": len(rows), "abnormal": int(truth.sum()),
        "normal": int((~truth).sum()),
        "ap": float(average_precision_score(truth, scores)) if truth.any() else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--notes", type=Path, default=DEFAULT_NOTES)
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

    labels = read_labels(args.notes)
    models = load_classifiers(device)
    localizer = load_model(LOCALIZATION_CHECKPOINT, device)
    rows, failures = [], []
    counters = Counter()
    args.output.mkdir(parents=True)
    with zipfile.ZipFile(args.archive) as archive:
        series = scan_series(archive)
        patients = sorted(set(labels) & set(series))
        if args.limit:
            patients = patients[:args.limit]
        source_label_counts = Counter(
            labels[patient][column]
            for patient in patients for column in LEVEL_COLUMNS.values()
        )
        total_eligible_discs = sum(
            source_label_counts[label] for label in ELIGIBLE_LABELS
        )
        for position, patient in enumerate(patients, 1):
            eligible = {level: (labels[patient][column], ELIGIBLE_LABELS.get(
                labels[patient][column]))
                for level, column in LEVEL_COLUMNS.items()}
            if all(truth is None for _, truth in eligible.values()):
                counters["no_eligible_levels"] += 1
                continue
            folder, filenames = series[patient][0]
            if len(series[patient]) > 1:
                counters["multiple_t2_series"] += 1
            try:
                volume = read_volume(archive, filenames)
                record = {"series_name": f"mendeley_{patient:04d}"}
                selected, image, selection = select_sagittal_slice(
                    record, volume, localizer, device,
                    middle_window=2, selection_mode="guarded_confidence")
                boxes, _ = localize_levels(
                    record, volume, localizer, device, localization_image=image)
                bands = level_bands(boxes, volume.shape[0])
            except (ValueError, OSError, KeyError) as error:
                failures.append({"patient_id": patient, "stage": "geometry",
                                 "error": str(error)})
                counters["geometry_failed_patients"] += 1
                continue
            for level, (label, truth) in eligible.items():
                if truth is None:
                    counters[f"excluded_label:{label}"] += 1
                    continue
                counters["eligible_discs"] += 1
                if level not in boxes:
                    counters["missing_disc_boxes"] += 1
                    failures.append({"patient_id": patient, "stage": level,
                                     "error": "No MRI-only predicted disc box"})
                    continue
                try:
                    stack = make_stack(volume, selected, boxes[level]["box"],
                                       crop_scale=1.5, posterior_shift=0.25,
                                       band=bands[level])
                    prediction = score_stack(stack, models, device)
                except (ValueError, RuntimeError) as error:
                    failures.append({"patient_id": patient, "stage": level,
                                     "error": str(error)})
                    counters["crop_or_model_failures"] += 1
                    continue
                rows.append({
                    "patient_id": patient, "level": level, "source_label": label,
                    "truth": truth, "series_folder": folder,
                    "series_slices": len(filenames), "selected_slice": selected,
                    "guarded_switch": int(selection["guarded_switch"]),
                    "disc_box_xyxy": json.dumps(boxes[level]["box"]),
                    **prediction,
                })
            if position % 25 == 0 or position == len(patients):
                print(f"patients={position}/{len(patients)} scored_discs={len(rows)} "
                      f"failed_geometry={counters['geometry_failed_patients']}",
                      flush=True)
    if not rows:
        raise RuntimeError("No discs could be scored; inspect localization failures")
    with (args.output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "failures.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("patient_id", "stage", "error"))
        writer.writeheader()
        writer.writerows(failures)
    report = {
        "dataset": "Original Mendeley 515-patient multi-slice sagittal T2",
        "label_source": "LSMA-PQR structured reports for the same patients",
        "task": "None versus Diffuse Bulge at L4-L5 and L5-S1",
        "patient_ids_considered": len(patients),
        "patients_scored": len({row["patient_id"] for row in rows}),
        "source_label_counts": dict(source_label_counts),
        "total_eligible_discs": total_eligible_discs,
        "scored_eligible_discs": len(rows),
        "unscored_eligible_discs": total_eligible_discs - len(rows),
        "eligible_disc_coverage": len(rows) / total_eligible_discs,
        "counters": dict(counters),
        "models": {
            name: {"checkpoint": str(directory / "fixed.pt"),
                   "threshold": models[name][1],
                   "overall": summarize(rows, f"{name}_score", f"{name}_flag"),
                   "by_level": {
                       level: summarize([r for r in rows if r["level"] == level],
                                        f"{name}_score", f"{name}_flag")
                       for level in LEVEL_COLUMNS
                   }}
            for name, (_, _, directory) in MODELS.items()
        },
        "limitations": [
            "Report-derived labels are not SPIDER expert bulging/herniation grades.",
            "Central Protrusion and ambiguous labels are excluded from AP.",
            "Missing localization boxes are counted separately and excluded from AP.",
            "This cohort was previously tested using LSMA-PQR single-slice images.",
            "No checkpoint or threshold was fitted on this external dataset.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
