import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import SimpleITK as sitk


def make_crop(
    image: np.ndarray,
    row: dict,
    output_size: int,
    row_spacing_mm: float,
    col_spacing_mm: float,
    fov_mm: float = 70.0,
) -> np.ndarray:
    x1, y1, x2, y2 = (
        int(row[key]) for key in ("x1", "y1", "x2", "y2")
    )

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid box in {row['series_name']}")
    if row_spacing_mm <= 0 or col_spacing_mm <= 0:
        raise ValueError(f"Invalid spacing in {row['series_name']}")

    # The annotated disc determines the center. Every output uses the
    # same physical field of view around that center.
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    crop_width = max(1, round(fov_mm / col_spacing_mm))
    crop_height = max(1, round(fov_mm / row_spacing_mm))

    left = round(center_x - crop_width / 2)
    top = round(center_y - crop_height / 2)
    right = left + crop_width
    bottom = top + crop_height

    valid = image[np.isfinite(image) & (image > 0)]
    if valid.size == 0:
        raise ValueError(f"Empty MRI slice in {row['series_name']}")

    low, high = np.percentile(valid, (1, 99))
    if high <= low:
        raise ValueError(f"No intensity range in {row['series_name']}")

    clean_image = np.nan_to_num(
        image, nan=low, posinf=high, neginf=low
    )
    normalized = np.clip(
        (clean_image - low) / (high - low), 0, 1
    ).astype(np.float32)

    # Copy the overlapping image area into a fixed-size crop.
    # Zeros are used only where the field of view crosses an image edge.
    crop = np.zeros(
        (crop_height, crop_width), dtype=np.float32
    )

    src_left = max(0, left)
    src_top = max(0, top)
    src_right = min(image.shape[1], right)
    src_bottom = min(image.shape[0], bottom)

    if src_left >= src_right or src_top >= src_bottom:
        raise ValueError(
            f"Crop misses the image in {row['series_name']}"
        )

    dst_left = src_left - left
    dst_top = src_top - top

    crop[
        dst_top:dst_top + (src_bottom - src_top),
        dst_left:dst_left + (src_right - src_left),
    ] = normalized[
        src_top:src_bottom,
        src_left:src_right,
    ]

    # Both axes cover fov_mm millimeters. Resizing to a square therefore
    # also corrects for different source pixel spacing on the two axes.
    resized = Image.fromarray(crop).resize(
        (output_size, output_size),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(resized, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/spider_discs.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/spider_crops"),
    )
    parser.add_argument("--size", type=int, default=160)
    args = parser.parse_args()

    with args.manifest.open(newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Manifest is empty")

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["series_name"]].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = []

    for series_name, series_rows in grouped.items():
        source = sitk.ReadImage(series_rows[0]["image_path"])
        oriented = sitk.DICOMOrient(source, "LPS")
        spacing = oriented.GetSpacing()
        row_spacing_mm = spacing[2]
        col_spacing_mm = spacing[1]
        volume = np.flip(
            sitk.GetArrayFromImage(oriented),
            axis=0,
        )

        for row in series_rows:
            axis = int(row["sagittal_axis"])
            if axis != 2:
                raise ValueError(
                    f"{series_name}: expected canonical axis 2, got {axis}. "
                    "Rebuild the manifest with LPS orientation."
                )

            slice_index = int(row["sagittal_slice"])
            image = np.take(volume, slice_index, axis=axis)
            crop = make_crop(
                image,
                row,
                args.size,
                row_spacing_mm=row_spacing_mm,
                col_spacing_mm=col_spacing_mm,
            )

            crop_path = (
                args.output_dir
                / f"{series_name}_ivd{int(row['ivd_label']):02d}.npy"
            )
            np.save(crop_path, crop)

            result.append({**row, "crop_path": str(crop_path)})

        print(f"Processed {series_name}: {len(series_rows)} discs")

    output_manifest = args.output_dir / "crops.csv"
    with output_manifest.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=result[0].keys())
        writer.writeheader()
        writer.writerows(result)

    print(f"Saved {len(result)} crops")
    print("Crop manifest:", output_manifest)


if __name__ == "__main__":
    main()