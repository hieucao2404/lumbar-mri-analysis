"""Pilot: sagittal abnormal-disc flags followed by matched axial geometry.

Input: one patient from the original Mendeley MRI_Data.zip archive.
Output: per-level JSON plus axial overlay images.
"""

import argparse
import io
import json
import threading
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pydicom
import torch
from PIL import Image

from scripts.evaluate_mendeley_multislice_transfer import (
    read_volume,
    scan_series,
)
from src.inference.attention_vgg16_stack import (
    load_classifier,
    predict_stacks,
)
from src.inference.herniation_levels import (
    LEVELS,
    LOCALIZATION_CHECKPOINT,
    level_bands,
    load_model,
    localize_levels,
    make_stack,
)
from src.inference.mri_slice_selector import select_sagittal_slice

from MRI_Lumbar.geometric_ldh.landmark_ldh_engine import LandmarkLdhEngine
from MRI_Lumbar.geometric_ldh.run_ldh_landmark import (
    analyze,
    load_models,
    render,
)

ARCHIVE = Path("data/raw/mendeley_lumbar/MRI_Data.zip")
CHECKPOINT = Path("outputs/spider_abnormal_vgg16_attention_guarded_v1/fixed.pt")
CALIBRATION = Path("outputs/spider_abnormal_vgg16_attention_guarded_v1/calibration.json")

# Start where the axial project has level-based examples.
AXIAL_LEVELS = {"L3-L4", "L4-L5", "L5-S1"}


def fuse_level(item):
    """Return one disc decision while retaining stage results for audit."""
    flag = item.get("sagittal_flag")
    if flag is False:
        return {
            "final_label": "normal",
            "final_grade": None,
            "decision_reason": "sagittal_screen_negative",
        }
    if flag is not True or item.get("status") != "axial_analyzed":
        return {
            "final_label": "indeterminate",
            "final_grade": None,
            "decision_reason": item.get("status", "no_result"),
        }

    grade = item.get("provisional_max_grade")
    if not isinstance(grade, int) or grade not in range(4):
        raise ValueError(f"Invalid axial grade for {item['level']}: {grade}")
    return {
        "final_label": "normal" if grade == 0 else "abnormal",
        "final_grade": grade,
        "decision_reason": "axial_grade_zero" if grade == 0 else "axial_grade_positive",
    }


def append_fused_level(result, item):
    item.update(fuse_level(item))
    result["levels"].append(item)


def dicom_from_zip(archive, filename):
    return pydicom.dcmread(io.BytesIO(archive.read(filename)))


def orientation(dcm):
    values = np.asarray(dcm.ImageOrientationPatient, dtype=float)
    across = values[:3]  # direction of increasing image column
    down = values[3:]    # direction of increasing image row
    normal = np.cross(across, down)
    return across, down, normal


def sagittal_disc_point(dcm, box):
    """Original sagittal box centre -> 3D patient coordinates, in millimetres."""
    x1, y1, x2, y2 = box
    column = (x1 + x2) / 2.0
    row = (y1 + y2) / 2.0

    # read_volume() flips rows when the sagittal row direction points upward.
    if float(dcm.ImageOrientationPatient[5]) > 0:
        row = dcm.Rows - 1 - row

    across, down, _ = orientation(dcm)
    row_spacing, column_spacing = map(float, dcm.PixelSpacing)
    origin = np.asarray(dcm.ImagePositionPatient, dtype=float)

    return (
        origin
        + column * column_spacing * across
        + row * row_spacing * down
    )


def axial_candidates(archive, patient_id, study_uid):
    """Find actual axial T2 images; exclude scout/localizer images."""
    patient_folder = f"/{patient_id:04d}/"
    grouped = {}

    for filename in archive.namelist():
        if not filename.endswith(".ima") or patient_folder not in filename:
            continue
        series_folder = filename.split("/")[-2].lower()
        if (
            "t2" not in series_folder
            or "tra" not in series_folder
        ):
            continue

        dcm = dicom_from_zip(archive, filename)
        if str(dcm.StudyInstanceUID) != study_uid:
            continue

        description = str(getattr(dcm, "SeriesDescription", "")).lower()
        if "t2" not in description or "tra" not in description:
            continue

        _, _, normal = orientation(dcm)
        if abs(normal[2]) < 0.70:
            continue  # not an axial or near-axial image

        uid = str(dcm.SeriesInstanceUID)
        grouped.setdefault(uid, []).append((filename, dcm))

    if not grouped:
        return []

    # Pilot choice: use the axial T2 series containing the most slices.
    best_uid = max(grouped, key=lambda uid: len(grouped[uid]))
    return grouped[best_uid]


def distance_to_axial_plane(point, dcm):
    across, down, normal = orientation(dcm)
    origin = np.asarray(dcm.ImagePositionPatient, dtype=float)
    delta = point - origin

    plane_distance = abs(float(np.dot(delta, normal)))
    row_spacing, column_spacing = map(float, dcm.PixelSpacing)
    column = float(np.dot(delta, across) / column_spacing)
    row = float(np.dot(delta, down) / row_spacing)

    inside_image = (
        0 <= row < dcm.Rows
        and 0 <= column < dcm.Columns
    )
    return plane_distance, inside_image


def select_axial_slices(point, candidates, maximum_distance_mm=8.0):
    ranked = []

    for filename, dcm in candidates:
        distance, inside = distance_to_axial_plane(point, dcm)
        if inside and distance <= maximum_distance_mm:
            ranked.append((distance, filename, dcm))

    ranked.sort(key=lambda item: item[0])
    return ranked[:3]


def axial_png(dcm, output_path):
    """Create the 8-bit input expected by the existing axial runner."""
    image = dcm.pixel_array.astype(np.float32)
    image = image * float(getattr(dcm, "RescaleSlope", 1))
    image = image + float(getattr(dcm, "RescaleIntercept", 0))

    valid = image[np.isfinite(image) & (image > 0)]
    if valid.size == 0:
        raise ValueError("Axial image has no usable positive pixels")

    low, high = np.percentile(valid, (1, 99))
    if high <= low:
        raise ValueError("Axial image has no usable intensity range")

    image = np.clip((image - low) / (high - low), 0, 1)
    image[~np.isfinite(image)] = 0
    image = (image * 255).astype(np.uint8)

    # The axial geometry model expects image-left to mean patient-right
    # and image-down to mean patient-posterior. Reject unexpected orientations.
    across, down, _ = orientation(dcm)
    if across[0] < 0.70 or down[1] < 0.70:
        raise ValueError("Axial orientation needs manual reorientation review")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)


def effective_spacing_mm(dcm):
    """Millimetres per pixel after the axial runner resizes to 256 × 256."""
    row_spacing, column_spacing = map(float, dcm.PixelSpacing)
    row_mm = row_spacing * dcm.Rows / 256
    column_mm = column_spacing * dcm.Columns / 256

    # The existing geometry engine accepts one scalar spacing.
    if abs(row_mm - column_mm) / max(row_mm, column_mm) > 0.05:
        raise ValueError("Resized axial pixels are anisotropic")

    return (row_mm + column_mm) / 2


def analyze_axial_level(
    item, matched, axial_models, output, render_lock, *, save_overlays=True,
):
    """Analyze one level independently, sharing read-only inference models."""
    level = item["level"]
    item["axial_slices"] = []
    for index, (distance, filename, dcm) in enumerate(matched):
        slice_item = {
            "dicom_path_in_archive": filename,
            "sop_instance_uid": str(dcm.SOPInstanceUID),
            "distance_to_disc_mm": round(distance, 2),
        }
        try:
            spacing = effective_spacing_mm(dcm)
            stem = f"{level}_{index}_{dcm.SOPInstanceUID}"
            png = output / "axial_inputs" / f"{stem}.png"
            overlay = output / "overlays" / f"{stem}.png"
            axial_png(dcm, png)
            engine = LandmarkLdhEngine(pixel_spacing_mm=spacing)
            image, mask, grade, landmark_method, reason = (
                analyze(png, axial_models, engine, det_size=256)
            )
            if grade is None:
                slice_item.update({
                    "status": "axial_geometry_failed", "reason": reason,
                })
            else:
                if save_overlays:
                    overlay.parent.mkdir(parents=True, exist_ok=True)
                    # Matplotlib's pyplot state is shared across threads.
                    with render_lock:
                        render(engine, image, mask, grade, overlay)
                slice_item.update({
                    "status": "ok",
                    "grade": int(grade.grade_num),
                    "zone": grade.zone_name,
                    "zone_code": grade.zone_code,
                    "side": grade.lateralization,
                    "ab_mm": float(grade.ab_mm),
                    "gap_mm": float(grade.gap_mm),
                    "cd_mm": float(grade.cd_mm),
                    "index_pct": float(grade.index_pct),
                    "landmark_method": landmark_method,
                    "input_png": str(png),
                    "overlay": str(overlay) if save_overlays else None,
                })
        except (ValueError, RuntimeError) as error:
            slice_item.update({
                "status": "axial_input_failed", "reason": str(error),
            })
        item["axial_slices"].append(slice_item)

    valid = [row for row in item["axial_slices"] if row["status"] == "ok"]
    if valid:
        representative = max(
            valid, key=lambda row: (row["grade"], -row["distance_to_disc_mm"]),
        )
        item["status"] = "axial_analyzed"
        item["provisional_max_grade"] = representative["grade"]
        item["provisional_zone"] = representative["zone"]
        item["provisional_side"] = representative["side"]
        item["grade_disagreement"] = len({row["grade"] for row in valid}) > 1
    else:
        item["status"] = "axial_analysis_failed"
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--serial",
        action="store_true",
        help="Run axial analysis only at sagittal-flagged levels",
    )
    mode_group.add_argument(
        "--concurrent",
        action="store_true",
        help="Run flagged axial levels concurrently after sagittal routing",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    localizer = load_model(LOCALIZATION_CHECKPOINT, device)
    classifier, threshold = load_classifier(
        CHECKPOINT, CALIBRATION, device
    )
    seg_model, det_model = load_models(device)
    axial_models = (seg_model, det_model, device)

    with zipfile.ZipFile(ARCHIVE) as archive:
        series = scan_series(archive)
        if args.patient_id not in series:
            raise ValueError(f"No sagittal T2 for patient {args.patient_id}")

        folder, sagittal_filenames = series[args.patient_id][0]
        sagittal_dcms = [
            dicom_from_zip(archive, name)
            for name in sagittal_filenames
        ]
        sagittal_dcms.sort(
            key=lambda d: float(d.ImagePositionPatient[0])
        )

        volume = read_volume(archive, sagittal_filenames)
        record = {"series_name": f"mendeley_{args.patient_id:04d}"}

        selected, localization_image, selection_info = (
            select_sagittal_slice(
                record,
                volume,
                localizer,
                device,
                middle_window=2,
                selection_mode="guarded_confidence",
            )
        )
        boxes, decoder = localize_levels(
            record,
            volume,
            localizer,
            device,
            localization_image=localization_image,
        )
        bands = level_bands(boxes, volume.shape[0])

        stacks = {
            level: make_stack(
                volume,
                selected,
                boxes[level]["box"],
                crop_scale=1.5,
                posterior_shift=0.25,
                band=bands[level],
            )
            for level in LEVELS
            if level in boxes
        }
        scores = predict_stacks(classifier, stacks, device)

        study_uid = str(sagittal_dcms[selected].StudyInstanceUID)
        axial_images = axial_candidates(
            archive, args.patient_id, study_uid
        )

        result = {
            "patient_id": args.patient_id,
            "study_uid": study_uid,
            "sagittal_series": folder,
            "selected_sagittal_slice": selected,
            "sagittal_decoder": decoder,
            "slice_selection": selection_info,
            "threshold": threshold,
            "mode": (
                "concurrent" if args.concurrent else
                "serial" if args.serial else "all_levels_pilot"
            ),
            "decision_rule": (
                "sagittal negative: normal; sagittal positive with axial grade 0: "
                "normal; sagittal positive with axial grade 1-3: abnormal; "
                "missing axial result: indeterminate"
            ),
            "axial_series_slices": len(axial_images),
            "levels": [],
        }

        items = []
        pending = []
        render_lock = threading.Lock()
        for level in LEVELS:
            item = {"level": level}
            items.append(item)

            if level not in boxes:
                item["status"] = "sagittal_level_unavailable"
                continue

            box = boxes[level]["box"]
            score = float(scores[level])
            flagged = score >= threshold
            item.update({
                "sagittal_disc_box_xyxy": box,
                "sagittal_abnormal_score": score,
                "sagittal_flag": flagged,
            })

            if level not in AXIAL_LEVELS:
                item["status"] = "axial_level_not_yet_validated"
                continue

            if (args.serial or args.concurrent) and not flagged:
                item["status"] = "not_routed_to_axial"
                continue

            point = sagittal_disc_point(sagittal_dcms[selected], box)
            matched = select_axial_slices(point, axial_images)
            item["axial_slices"] = []
            if not matched:
                item["status"] = "no_confident_axial_match"
                continue
            pending.append((item, matched))

        if args.concurrent and pending:
            with ThreadPoolExecutor(max_workers=min(3, len(pending))) as pool:
                futures = [
                    pool.submit(
                        analyze_axial_level, item, matched, axial_models,
                        args.output, render_lock,
                    )
                    for item, matched in pending
                ]
                for future in futures:
                    future.result()
        else:
            for item, matched in pending:
                analyze_axial_level(
                    item, matched, axial_models, args.output, render_lock
                )

        for item in items:
            append_fused_level(result, item)

    output_json = args.output / "result.json"
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "patient_id": args.patient_id,
        "mode": result["mode"],
        "final_abnormal_levels": [
            row["level"] for row in result["levels"]
            if row["final_label"] == "abnormal"
        ],
        "final_indeterminate_levels": [
            row["level"] for row in result["levels"]
            if row["final_label"] == "indeterminate"
        ],
        "flagged_levels": [
            row["level"] for row in result["levels"]
            if row.get("sagittal_flag")
        ],
        "axial_analyzed_levels": [
            row["level"] for row in result["levels"]
            if row["status"] == "axial_analyzed"
        ],
        "output": str(output_json),
    }, indent=2))


if __name__ == "__main__":
    main()
