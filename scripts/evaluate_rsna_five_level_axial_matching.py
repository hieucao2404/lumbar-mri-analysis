"""Audit sagittal-to-axial level matching on paired RSNA DICOM studies.

RSNA axial coordinate annotations are used only for evaluation, never to pick
the predicted axial series or slices. Stenosis labels are not disc pathology
labels, so this script does not calculate abnormal-disc accuracy or F1.
"""

import argparse
import csv
import io
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pydicom
import torch

from scripts.run_paired_sagittal_axial import (
    distance_to_axial_plane,
    sagittal_disc_point,
    select_axial_slices,
)
from src.inference.attention_vgg16_stack import load_classifier, predict_stacks
from src.inference.herniation_levels import (
    LEVELS,
    LOCALIZATION_CHECKPOINT,
    level_bands,
    load_model,
    localize_levels,
    make_stack,
)
from src.inference.mri_slice_selector import select_sagittal_slice


ARCHIVE = Path("data/raw/rsna-2024-lsdd.zip")
ROOT = "rsna-2024-lumbar-spine-degenerative-classification/"
CHECKPOINT = Path("outputs/spider_abnormal_vgg16_attention_guarded_v1/fixed.pt")
CALIBRATION = Path("outputs/spider_abnormal_vgg16_attention_guarded_v1/calibration.json")
AXIAL_CONDITIONS = {"Left Subarticular Stenosis", "Right Subarticular Stenosis"}


def read_csv(archive, filename):
    with archive.open(ROOT + filename) as stream:
        return list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig")))


def index_images(archive):
    result = defaultdict(list)
    prefix = ROOT + "train_images/"
    for filename in archive.namelist():
        if not filename.startswith(prefix) or not filename.endswith(".dcm"):
            continue
        parts = filename[len(prefix):].split("/")
        if len(parts) == 3 and parts[2][:-4].isdigit():
            result[(parts[0], parts[1])].append(filename)
    return result


def dicom(archive, filename, *, pixels=True):
    return pydicom.dcmread(
        io.BytesIO(archive.read(filename)), stop_before_pixels=not pixels
    )


def sagittal_volume(archive, filenames):
    slices = []
    for filename in filenames:
        ds = dicom(archive, filename)
        image = ds.pixel_array
        direction = np.asarray(ds.ImageOrientationPatient, dtype=float)
        if image.ndim != 2 or abs(direction[5]) < 0.9:
            raise ValueError("Sagittal series contains a non-upright 2D slice")
        if direction[5] > 0:
            image = image[::-1]
        slices.append((float(ds.ImagePositionPatient[0]), image, ds))
    if len(slices) < 8:
        raise ValueError("Sagittal series has fewer than eight slices")
    slices.sort(key=lambda item: item[0])
    if len({item[1].shape for item in slices}) != 1:
        raise ValueError("Sagittal image shapes differ within a series")
    if any(right[0] <= left[0] for left, right in zip(slices, slices[1:])):
        raise ValueError("Sagittal slice positions repeat")
    return np.stack([item[1] for item in slices], axis=-1), [item[2] for item in slices]


def summarize(rows):
    comparable = [row for row in rows if row.get("reference_in_selected_series")]
    selected = [row for row in rows if row.get("predicted_instances")]
    result = {
        "expected_levels": len(rows),
        "localized_levels": sum(row.get("disc_box_xyxy") is not None for row in rows),
        "matched_levels": len(selected),
        "reference_in_selected_series": len(comparable),
        "top1_exact_count": sum(row.get("top1_exact", False) for row in comparable),
        "top3_exact_count": sum(row.get("top3_exact", False) for row in comparable),
        "top3_adjacent_count": sum(row.get("top3_adjacent", False) for row in comparable),
    }
    denominator = len(comparable)
    for key in ("top1_exact", "top3_exact", "top3_adjacent"):
        result[key + "_rate"] = (
            result[key + "_count"] / denominator if denominator else None
        )
    result["top3_exact_full_cohort_rate"] = (
        result["top3_exact_count"] / len(rows) if rows else None
    )
    result["status_counts"] = dict(Counter(row["status"] for row in rows))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--studies", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.studies <= 0:
        raise ValueError("--studies must be positive")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)

    with zipfile.ZipFile(args.archive) as archive:
        descriptions = read_csv(archive, "train_series_descriptions.csv")
        coordinates = read_csv(archive, "train_label_coordinates.csv")
        images = index_images(archive)
        series = defaultdict(lambda: defaultdict(list))
        for item in descriptions:
            series[item["study_id"]][item["series_description"]].append(item["series_id"])

        axial_ids = {
            (study, series_id)
            for study, kinds in series.items()
            for series_id in kinds.get("Axial T2", [])
        }
        references = defaultdict(lambda: defaultdict(set))
        study_levels = defaultdict(set)
        for row in coordinates:
            key = (row["study_id"], row["series_id"])
            if key not in axial_ids or row["condition"] not in AXIAL_CONDITIONS:
                continue
            level = row["level"].replace("/", "-")
            number = int(row["instance_number"])
            references[key][level].add(number)
            study_levels[row["study_id"]].add(level)

        eligible = sorted(
            study for study, kinds in series.items()
            if kinds.get("Sagittal T2/STIR") and kinds.get("Axial T2")
            and set(LEVELS) <= study_levels[study]
        )
        chosen = random.Random(args.seed).sample(
            eligible, min(args.studies, len(eligible))
        )
        localizer = load_model(LOCALIZATION_CHECKPOINT, device)
        classifier, threshold = load_classifier(CHECKPOINT, CALIBRATION, device)
        rows = []
        failures = []

        for position, study in enumerate(chosen, 1):
            kinds = series[study]
            sagittal_id = max(
                kinds["Sagittal T2/STIR"],
                key=lambda identifier: len(images[(study, identifier)]),
            )
            axial_id = max(
                kinds["Axial T2"],
                key=lambda identifier: len(images[(study, identifier)]),
            )
            common = {
                "study_id": study,
                "sagittal_series_id": sagittal_id,
                "axial_series_id": axial_id,
            }
            try:
                volume, sagittal_dcms = sagittal_volume(
                    archive, images[(study, sagittal_id)]
                )
                record = {"series_name": f"rsna_{study}"}
                selected, image, selection = select_sagittal_slice(
                    record, volume, localizer, device,
                    middle_window=2, selection_mode="guarded_confidence",
                )
                boxes, decoder = localize_levels(
                    record, volume, localizer, device, localization_image=image
                )
                bands = level_bands(boxes, volume.shape[0])
                stacks = {
                    level: make_stack(
                        volume, selected, boxes[level]["box"],
                        crop_scale=1.5, posterior_shift=0.25, band=bands[level],
                    )
                    for level in LEVELS if level in boxes
                }
                scores = predict_stacks(classifier, stacks, device) if stacks else {}
                axial = [
                    (filename, dicom(archive, filename, pixels=False))
                    for filename in images[(study, axial_id)]
                ]
                axial_by_number = {
                    int(Path(filename).stem): ds for filename, ds in axial
                }
            except (ValueError, RuntimeError, OSError, KeyError, AttributeError) as error:
                failures.append({**common, "error": str(error)})
                for level in LEVELS:
                    rows.append({**common, "level": level, "status": "study_failed"})
                continue

            for level in LEVELS:
                truth = references[(study, axial_id)].get(level, set())
                item = {
                    **common, "level": level,
                    "selected_sagittal_instance": int(
                        sagittal_dcms[selected].InstanceNumber
                    ),
                    "disc_box_xyxy": boxes[level]["box"] if level in boxes else None,
                    "abnormal_score": scores.get(level),
                    "abnormal_flag": scores[level] >= threshold if level in scores else None,
                    "reference_in_selected_series": bool(truth),
                    "reference_instances": sorted(truth),
                    "predicted_instances": [],
                    "status": "missing_disc_box" if level not in boxes else "no_axial_match",
                }
                if level in boxes:
                    point = sagittal_disc_point(
                        sagittal_dcms[selected], boxes[level]["box"]
                    )
                    matched = select_axial_slices(point, axial)
                    predicted = [int(Path(filename).stem) for _, filename, _ in matched]
                    item["predicted_instances"] = predicted
                    if predicted:
                        item["status"] = "matched"
                    if truth:
                        item["top1_exact"] = bool(predicted and predicted[0] in truth)
                        item["top3_exact"] = any(number in truth for number in predicted)
                        item["top3_adjacent"] = any(
                            abs(number - reference) <= 1
                            for number in predicted for reference in truth
                        )
                        reference_distances = [
                            distance_to_axial_plane(point, axial_by_number[number])[0]
                            for number in truth if number in axial_by_number
                        ]
                        item["reference_plane_distance_mm"] = (
                            min(reference_distances) if reference_distances else None
                        )
                        item["predicted_plane_distance_mm"] = (
                            matched[0][0] if matched else None
                        )
                rows.append(item)

            if position % 10 == 0 or position == len(chosen):
                summary = summarize(rows)
                print(
                    f"studies={position}/{len(chosen)} "
                    f"matched={summary['matched_levels']} "
                    f"top3_exact={summary['top3_exact_count']}",
                    flush=True,
                )

    args.output.mkdir(parents=True)
    report = {
        "task": "five-level sagittal-to-axial instance matching",
        "reference": "RSNA left/right subarticular axial coordinate instances",
        "selection_uses_reference_labels": False,
        "eligible_studies": len(eligible),
        "sampled_studies": len(chosen),
        "seed": args.seed,
        "device": str(device),
        "sagittal_abnormal_threshold": threshold,
        "overall": summarize(rows),
        "by_level": {
            level: summarize([row for row in rows if row["level"] == level])
            for level in LEVELS
        },
        "failed_studies": failures,
        "caveat": "RSNA stenosis labels cannot measure bulge/herniation classification F1.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "level_predictions.csv").open("w", newline="") as stream:
        fields = (
            "study_id", "sagittal_series_id", "axial_series_id", "level",
            "status", "selected_sagittal_instance", "disc_box_xyxy",
            "abnormal_score", "abnormal_flag", "reference_in_selected_series",
            "reference_instances", "predicted_instances", "top1_exact",
            "top3_exact", "top3_adjacent", "reference_plane_distance_mm",
            "predicted_plane_distance_mm",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    print(json.dumps({
        "sampled_studies": len(chosen), "overall": report["overall"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
