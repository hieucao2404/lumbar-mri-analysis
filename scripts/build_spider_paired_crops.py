import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import SimpleITK as sitk


def normalize_slice(image: np.ndarray) -> np.ndarray:
    valid = image[np.isfinite(image) & (image > 0)]

    if valid.size == 0:
        raise ValueError("MRI slice has no positive finite pixels")

    low, high = np.percentile(valid, (1, 99))

    if high <= low:
        raise ValueError("MRI slice has no useful intensity range")

    normalized = np.clip(
        (image.astype(np.float32) - low) / (high - low),
        0.0,
        1.0,
    )

    normalized[~np.isfinite(normalized)] = 0.0
    return normalized.astype(np.float32)


def mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)

    if len(xs) == 0:
        raise ValueError("Disc ID is absent from selected slice")

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def predicted_box_to_original(
    row: dict,
    original_height: int,
    original_width: int,
    canvas_size: int,
) -> tuple[int, int, int, int]:
    """Invert the geometry in letterbox_pair()."""
    scale = min(
        canvas_size / original_width,
        canvas_size / original_height,
    )

    new_width = max(1, round(original_width * scale))
    new_height = max(1, round(original_height * scale))

    offset_x = (canvas_size - new_width) // 2
    offset_y = (canvas_size - new_height) // 2

    # Use the actual rounded resize dimensions.
    scale_x = new_width / original_width
    scale_y = new_height / original_height

    x1 = math.floor(
        (int(row["x1"]) - offset_x) / scale_x
    )
    y1 = math.floor(
        (int(row["y1"]) - offset_y) / scale_y
    )
    x2 = math.ceil(
        (int(row["x2"]) - offset_x) / scale_x
    )
    y2 = math.ceil(
        (int(row["y2"]) - offset_y) / scale_y
    )

    x1 = max(0, min(original_width, x1))
    x2 = max(0, min(original_width, x2))
    y1 = max(0, min(original_height, y1))
    y2 = max(0, min(original_height, y2))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid mapped box for {row['series_name']}: "
            f"{(x1, y1, x2, y2)}"
        )

    return x1, y1, x2, y2


def make_crop(
    normalized_image: np.ndarray,
    box: tuple[int, int, int, int],
    output_size: int,
) -> np.ndarray:
    """Match the context padding in your Pfirrmann crop exporter."""
    x1, y1, x2, y2 = box
    image_height, image_width = normalized_image.shape

    box_width = x2 - x1
    box_height = y2 - y1

    if box_width <= 0 or box_height <= 0:
        raise ValueError(f"Invalid disc box: {box}")

    pad_x = max(8, round(box_width * 1.0))
    pad_y = max(8, round(box_height * 1.5))

    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(image_width, x2 + pad_x)
    bottom = min(image_height, y2 + pad_y)

    crop = normalized_image[
        top:bottom,
        left:right,
    ].astype(np.float32)

    crop_height, crop_width = crop.shape

    scale = min(
        output_size / crop_width,
        output_size / crop_height,
    )

    resized_width = max(1, round(crop_width * scale))
    resized_height = max(1, round(crop_height * scale))

    resized = np.asarray(
        Image.fromarray(crop).resize(
            (resized_width, resized_height),
            Image.Resampling.BILINEAR,
        ),
        dtype=np.float32,
    )

    canvas = np.zeros(
        (output_size, output_size),
        dtype=np.float32,
    )

    offset_x = (output_size - resized_width) // 2
    offset_y = (output_size - resized_height) // 2

    canvas[
        offset_y:offset_y + resized_height,
        offset_x:offset_x + resized_width,
    ] = resized

    return canvas


def load_grades(
    path: Path,
) -> tuple[dict[tuple[str, int], int], set[str]]:
    grades = {}
    patients_with_zero_label = set()

    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            patient_id = row["Patient"].strip()
            ivd_label = int(row["IVD label"])
            grade = int(row["Pfirrman grade"])

            if ivd_label == 0:
                patients_with_zero_label.add(patient_id)

            key = (patient_id, ivd_label)

            if key in grades:
                raise ValueError(f"Duplicate grading row: {key}")

            grades[key] = grade

    return grades, patients_with_zero_label


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/spider"),
    )
    parser.add_argument(
        "--matches",
        type=Path,
        default=Path(
            "data/processed/spider_disc_localization/"
            "validation_matched_discs.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/processed/spider_paired_disc_crops"
        ),
    )
    parser.add_argument("--canvas-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N matches for a smoke test",
    )

    args = parser.parse_args()

    with args.matches.open(newline="") as file:
        match_rows = list(csv.DictReader(file))

    if args.limit is not None:
        match_rows = match_rows[:args.limit]

    if not match_rows:
        raise ValueError("Matched-disc CSV is empty")

    grades, patients_with_zero_label = load_grades(
        args.root / "radiological_gradings.csv"
    )

    # A missing expected grade also makes that patient's mapping unsafe.
    excluded_patients = set(patients_with_zero_label)

    for row in match_rows:
        patient_id = row["patient_id"]
        ivd_label = int(row["disc_id"]) - 200

        if (patient_id, ivd_label) not in grades:
            excluded_patients.add(patient_id)

    excluded_rows = [
        row for row in match_rows
        if row["patient_id"] in excluded_patients
    ]

    match_rows = [
        row for row in match_rows
        if row["patient_id"] not in excluded_patients
    ]

    print("Excluded patients with uncertain disc numbering:",
        sorted(excluded_patients))
    print("Excluded matched discs:", len(excluded_rows))

    if not match_rows:
        raise ValueError("No safely labelled matches remain")

    grouped = defaultdict(list)

    for row in match_rows:
        grouped[row["series_name"]].append(row)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_rows = []
    

    for series_name, series_rows in grouped.items():
        image_path = (
            args.root / "images" / f"{series_name}.mha"
        )
        mask_path = (
            args.root / "masks" / f"{series_name}.mha"
        )

        image_source = sitk.DICOMOrient(
            sitk.ReadImage(str(image_path)),
            "LPS",
        )
        mask_source = sitk.DICOMOrient(
            sitk.ReadImage(str(mask_path)),
            "LPS",
        )

        image_volume = np.flip(
            sitk.GetArrayFromImage(image_source),
            axis=0,
        )
        mask_volume = np.flip(
            sitk.GetArrayFromImage(mask_source),
            axis=0,
        )

        if image_volume.shape != mask_volume.shape:
            raise ValueError(
                f"{series_name}: image and mask shapes differ"
            )

        for row in series_rows:
            slice_index = int(row["sagittal_slice"])

            image = image_volume[:, :, slice_index]
            mask = mask_volume[:, :, slice_index]
            height, width = image.shape

            if (
                height != int(row["original_height"])
                or width != int(row["original_width"])
            ):
                raise ValueError(
                    f"{series_name}: original image size "
                    "does not match matched-disc CSV"
                )

            disc_id = int(row["disc_id"])
            ivd_label = disc_id - 200
            patient_id = row["patient_id"]

            grade_key = (patient_id, ivd_label)

            if grade_key not in grades:
                raise ValueError(
                    f"No Pfirrmann grade for {grade_key}"
                )

            expert_box = mask_box(mask == disc_id)

            predicted_box = predicted_box_to_original(
                row,
                height,
                width,
                args.canvas_size,
            )

            normalized = normalize_slice(image)

            expert_crop = make_crop(
                normalized,
                expert_box,
                args.crop_size,
            )
            predicted_crop = make_crop(
                normalized,
                predicted_box,
                args.crop_size,
            )

            prefix = (
                f"{series_name}_ivd{ivd_label:02d}_"
                f"component{int(row['component_id']):02d}"
            )

            expert_path = (
                args.output_dir / f"{prefix}_expert.npy"
            )
            predicted_path = (
                args.output_dir / f"{prefix}_predicted.npy"
            )

            np.save(expert_path, expert_crop)
            np.save(predicted_path, predicted_crop)

            output_rows.append({
                "patient_id": patient_id,
                "series_name": series_name,
                "subset": row["subset"],
                "sagittal_slice": slice_index,
                "disc_id": disc_id,
                "ivd_label": ivd_label,
                "pfirrmann_grade": grades[grade_key],
                "mask_iou": row["mask_iou"],
                "expert_crop_path": str(expert_path),
                "predicted_crop_path": str(predicted_path),
                "expert_box_original": str(expert_box),
                "predicted_box_original": str(predicted_box),
            })

    manifest_path = args.output_dir / "paired_crops.csv"

    with manifest_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(output_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print("Input matches:", len(match_rows))
    print("Paired crops saved:", len(output_rows))
    print("Output shape:", expert_crop.shape)
    print("Manifest:", manifest_path)
    print("First record:", output_rows[0])


if __name__ == "__main__":
    main()