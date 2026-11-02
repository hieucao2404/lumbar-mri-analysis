"""Evaluate frozen sagittal-to-axial decisions using LSMA report-derived labels.

Original Mendeley DICOM supplies the multi-slice inputs. Predictions are made
without reference labels; labels are attached afterwards. Failed cases remain
indeterminate, with both coverage and full-cohort detection sensitivity reported.
"""

import argparse
import csv
import json
import threading
import time
import zipfile
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score

from scripts.evaluate_mendeley_multislice_transfer import (
    DEFAULT_NOTES, LEVEL_COLUMNS, read_labels, read_volume, scan_series,
)
from scripts.run_paired_sagittal_axial import (
    ARCHIVE, CALIBRATION, CHECKPOINT, LEVELS, LOCALIZATION_CHECKPOINT,
    analyze_axial_level, axial_candidates, dicom_from_zip, fuse_level,
    level_bands, load_classifier, load_model, load_models, localize_levels,
    make_stack, predict_stacks, sagittal_disc_point, select_axial_slices,
    select_sagittal_slice,
)


LABEL_MAP = {"None": 0, "Diffuse Bulge": 1, "Central Protrusion": 1}
EVALUATED_LEVELS = tuple(LEVEL_COLUMNS)


def binary_metrics(truth, predictions):
    tp = sum(y == 1 and p == 1 for y, p in zip(truth, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(truth, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(truth, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(truth, predictions))
    ratio = lambda n, d: n / d if d else None
    return {
        "discs": len(truth), "abnormal": sum(truth),
        "normal": len(truth) - sum(truth),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": ratio(tp, tp + fp),
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "accuracy": ratio(tp + tn, len(truth)),
    }


def summarize(rows):
    labelled = [r for r in rows if r["truth"] is not None]
    decided = [r for r in labelled if r["final_label"] in ("normal", "abnormal")]
    unresolved = [r for r in labelled if r["final_label"] == "indeterminate"]
    scored = [r for r in labelled if r.get("sagittal_abnormal_score") is not None]
    abnormal = sum(r["truth"] for r in labelled)
    detected = sum(r["truth"] == 1 and r["final_label"] == "abnormal" for r in labelled)
    final = binary_metrics(
        [r["truth"] for r in decided],
        [int(r["final_label"] == "abnormal") for r in decided],
    )
    sagittal = binary_metrics(
        [r["truth"] for r in scored], [int(r["sagittal_flag"]) for r in scored],
    )
    sagittal["ap"] = (
        float(average_precision_score(
            [r["truth"] for r in scored],
            [r["sagittal_abnormal_score"] for r in scored],
        )) if scored and any(r["truth"] for r in scored) else None
    )
    paired = [r for r in decided if r.get("sagittal_flag") is not None]
    sagittal_paired = binary_metrics(
        [r["truth"] for r in paired], [int(r["sagittal_flag"]) for r in paired],
    )
    return {
        "labelled_discs": len(labelled), "abnormal": abnormal,
        "normal": len(labelled) - abnormal,
        "source_labels": dict(Counter(r["source_label"] for r in labelled)),
        "decided_discs": len(decided), "indeterminate_discs": len(unresolved),
        "decision_coverage": len(decided) / len(labelled) if labelled else None,
        "indeterminate_abnormal": sum(r["truth"] for r in unresolved),
        "indeterminate_normal": sum(r["truth"] == 0 for r in unresolved),
        "final_on_decided_discs": final,
        "detected_abnormal_sensitivity_all_labelled_discs": (
            detected / abnormal if abnormal else None
        ),
        "sagittal_on_scored_discs": sagittal,
        "sagittal_on_decided_discs": sagittal_paired,
        "axial_effect_on_decided_discs": {
            "false_positive_alerts_corrected": sum(
                r["truth"] == 0 and r["sagittal_flag"] and r["final_label"] == "normal"
                for r in paired
            ),
            "true_positive_alerts_lost": sum(
                r["truth"] == 1 and r["sagittal_flag"] and r["final_label"] == "normal"
                for r in paired
            ),
        },
        "status_counts": dict(Counter(r["status"] for r in labelled)),
        "decision_reasons": dict(Counter(r["decision_reason"] for r in labelled)),
        "final_ap": None,
    }


def predict_patient(patient, series, archive, localizer, classifier, threshold,
                    axial_models, device, output):
    """Same sagittal geometry and fusion as the pilot; grade only the two test levels."""
    if patient not in series:
        raise ValueError("No qualifying multi-slice sagittal T2 series")
    folder, filenames = series[patient][0]
    sagittal_dcms = [dicom_from_zip(archive, name) for name in filenames]
    sagittal_dcms.sort(key=lambda d: float(d.ImagePositionPatient[0]))
    volume = read_volume(archive, filenames)
    record = {"series_name": f"mendeley_{patient:04d}"}
    selected, image, selection = select_sagittal_slice(
        record, volume, localizer, device, middle_window=2,
        selection_mode="guarded_confidence",
    )
    boxes, decoder = localize_levels(
        record, volume, localizer, device, localization_image=image,
    )
    bands = level_bands(boxes, volume.shape[0])
    stacks = {
        level: make_stack(
            volume, selected, boxes[level]["box"], crop_scale=1.5,
            posterior_shift=0.25, band=bands[level],
        ) for level in LEVELS if level in boxes
    }
    scores = predict_stacks(classifier, stacks, device)
    study_uid = str(sagittal_dcms[selected].StudyInstanceUID)
    candidates = axial_candidates(archive, patient, study_uid)
    items = []
    for level in EVALUATED_LEVELS:
        item = {"level": level}
        if level not in boxes:
            item["status"] = "sagittal_level_unavailable"
        else:
            score = float(scores[level])
            flagged = score >= threshold
            item.update({
                "sagittal_disc_box_xyxy": boxes[level]["box"],
                "sagittal_abnormal_score": score, "sagittal_flag": flagged,
            })
            if not flagged:
                item["status"] = "not_routed_to_axial"
            else:
                point = sagittal_disc_point(sagittal_dcms[selected], boxes[level]["box"])
                matched = select_axial_slices(point, candidates)
                item["axial_slices"] = []
                if not matched:
                    item["status"] = "no_confident_axial_match"
                else:
                    analyze_axial_level(
                        item, matched, axial_models, output, threading.Lock(),
                        save_overlays=False,
                    )
        item.update(fuse_level(item))
        items.append(item)
    return {
        "patient_id": patient, "sagittal_series": folder, "study_uid": study_uid,
        "selected_sagittal_slice": selected, "slice_selection": selection,
        "sagittal_decoder": decoder, "threshold": threshold,
        "axial_series_slices": len(candidates), "levels": items,
    }


def attach_labels(patient, prediction, label_row):
    rows = []
    for item in prediction["levels"]:
        label = label_row[LEVEL_COLUMNS[item["level"]]]
        rows.append({
            "patient_id": patient, "level": item["level"],
            "source_label": label, "truth": LABEL_MAP.get(label),
            "sagittal_abnormal_score": item.get("sagittal_abnormal_score"),
            "sagittal_flag": item.get("sagittal_flag"),
            "status": item["status"], "final_label": item["final_label"],
            "final_grade": item["final_grade"],
            "decision_reason": item["decision_reason"],
            "axial_slices": len(item.get("axial_slices", [])),
            "axial_slices_ok": sum(s["status"] == "ok" for s in item.get("axial_slices", [])),
            "error": prediction.get("error", ""),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--notes", type=Path, default=DEFAULT_NOTES)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.limit < 0:
        raise ValueError("limit must be nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    labels = read_labels(args.notes)
    patients = sorted(labels)[:args.limit or None]
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    localizer = load_model(LOCALIZATION_CHECKPOINT, device)
    classifier, threshold = load_classifier(CHECKPOINT, CALIBRATION, device)
    axial_models = (*load_models(device), device)
    rows, failures = [], []
    with zipfile.ZipFile(args.archive) as archive:
        series = scan_series(archive)
        for position, patient in enumerate(patients, 1):
            output = args.output / "patients" / f"patient_{patient:04d}"
            output.mkdir(parents=True)
            patient_started = time.perf_counter()
            try:
                prediction = predict_patient(
                    patient, series, archive, localizer, classifier, threshold,
                    axial_models, device, output,
                )
            except (ValueError, OSError, KeyError, RuntimeError) as error:
                if isinstance(error, RuntimeError) and "out of memory" in str(error).lower():
                    raise
                failure = {"patient_id": patient, "error": str(error)}
                failures.append(failure)
                items = []
                for level in EVALUATED_LEVELS:
                    item = {"level": level, "status": "patient_processing_failed"}
                    item.update(fuse_level(item))
                    items.append(item)
                prediction = {**failure, "levels": items}
            prediction["processing_seconds"] = round(time.perf_counter() - patient_started, 4)
            (output / "result.json").write_text(json.dumps(prediction, indent=2) + "\n")
            rows.extend(attach_labels(patient, prediction, labels[patient]))
            if position % 25 == 0 or position == len(patients):
                print(json.dumps({
                    "patients": position, "total": len(patients),
                    "discs": len(rows), "patient_failures": len(failures),
                    "final_labels": dict(Counter(r["final_label"] for r in rows)),
                    "elapsed_seconds": round(time.perf_counter() - started, 1),
                }), flush=True)
    with (args.output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparable = [r for r in rows if r["source_label"] in ("None", "Diffuse Bulge")]
    report = {
        "task": "frozen sagittal-to-axial normal versus bulge or protrusion",
        "patients": len(patients), "evaluated_levels": EVALUATED_LEVELS,
        "label_mapping": LABEL_MAP, "threshold": threshold,
        "classifier": str(CHECKPOINT), "localizer": str(LOCALIZATION_CHECKPOINT),
        "calibration": str(CALIBRATION), "archive": str(args.archive),
        "notes": str(args.notes), "device": str(device),
        "execution": "persistent models; sequential flagged-level routing; no overlays",
        "overall": summarize(rows),
        "by_level": {level: summarize([r for r in rows if r["level"] == level])
                     for level in EVALUATED_LEVELS},
        "none_vs_diffuse_bulge_subset": summarize(comparable),
        "patient_failures": failures,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "limitations": [
            "Report-derived labels have not been independently adjudicated for this evaluation.",
            "Central Protrusion is mapped to abnormal; the historical None/Diffuse Bulge subset is also reported.",
            "Only L4-L5 and L5-S1 have disc-type reference labels in this CSV.",
            "Indeterminate cases are excluded from decided-disc F1 and reported as failures with full-cohort sensitivity.",
            "The final decision has no defined continuous score, so fused AP is not calculated.",
            "Axial geometric grades 0-3 are not validated against matching expert grades by this experiment.",
            "Axial training scripts name AISSLab; checkpoint training provenance is not independently verified.",
            "This external cohort has been inspected in earlier experiments; the evaluation is exploratory.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
