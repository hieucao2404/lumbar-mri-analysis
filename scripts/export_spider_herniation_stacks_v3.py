import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image


SOURCE_CSV = Path("data/processed/spider_discs.csv")
LOCALIZATION_MANIFEST = Path(
    "data/processed/spider_disc_localization/manifest.csv"
)
OUTPUT = Path("data/processed/spider_herniation_stacks_v3_fixed")
CROP_SIZE = 160


def load_volume(path):
    image = sitk.DICOMOrient(sitk.ReadImage(str(path)), "LPS")
    return np.flip(sitk.GetArrayFromImage(image), axis=0)


def normalize_slice(image):
    image = image.astype(np.float32)
    valid = image[np.isfinite(image) & (image > 0)]

    if valid.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)

    low, high = np.percentile(valid, (1, 99))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)

    return (
        np.clip((image - low) / (high - low), 0, 1) * 255
    ).astype(np.uint8)


def make_fixed_crop(
    image, box, *, crop_scale=1.5, posterior_shift=0.25
):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    cx = (x1 + x2) / 2 + posterior_shift * width
    cy = (y1 + y2) / 2
    half_side = max(48.0, crop_scale * width, 2.5 * height)

    # Keep the rest of the existing function.

    rectangle = (
        round(cx - half_side),
        round(cy - half_side),
        round(cx + half_side),
        round(cy + half_side),
    )

    mri = (
        Image.fromarray(normalize_slice(image))
        .crop(rectangle)
        .resize((CROP_SIZE, CROP_SIZE), Image.Resampling.BILINEAR)
    )

    # The MRI-only classifier receives a zero mask channel.
    result = np.zeros((2, CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    result[0] = np.asarray(mri, dtype=np.uint8)
    return result


def main():
    with SOURCE_CSV.open(newline="") as f:
        disc_rows = list(csv.DictReader(f))

    with LOCALIZATION_MANIFEST.open(newline="") as f:
        localization_rows = list(csv.DictReader(f))

    selected_slice_by_series = {
        row["series_name"]: int(row["sagittal_slice"])
        for row in localization_rows
    }

    grouped = {}
    for row in disc_rows:
        if not row["series_name"].endswith("_t2"):
            continue
        if row["subset"] not in {"training", "validation"}:
            continue
        if row["ivd_label"] not in {"1", "2", "3", "4", "5"}:
            continue
        if row["disc_herniation"] not in {"0", "1"}:
            continue

        grouped.setdefault(row["series_name"], []).append(row)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    counts = Counter()
    skipped = []

    for series, rows in sorted(grouped.items()):
        if series not in selected_slice_by_series:
            raise ValueError(f"No localization manifest row: {series}")

        image = load_volume(rows[0]["image_path"])
        mask = load_volume(rows[0]["mask_path"])

        if image.shape != mask.shape:
            raise ValueError(f"{series}: image/mask shapes differ")

        selected = selected_slice_by_series[series]
        if not 0 <= selected < image.shape[2]:
            raise ValueError(f"{series}: invalid selected slice {selected}")

        indices = np.clip(
            np.arange(selected - 3, selected + 5),
            0,
            image.shape[2] - 1,
        ).astype(int)

        for row in rows:
            value = int(row["mask_value"])
            ys, xs = np.where(mask[:, :, selected] == value)

            # Do not fabricate an expert anchor for a disc absent
            # from the pipeline-selected slice.
            if len(xs) < 25:
                skipped.append((
                    series, row["ivd_label"], row["subset"],
                    row["disc_herniation"], len(xs)
                ))
                continue

            anchor_box = [
                int(xs.min()),
                int(ys.min()),
                int(xs.max()) + 1,
                int(ys.max()) + 1,
            ]

            stack = np.stack(
                [
                    make_fixed_crop(image[:, :, index], anchor_box)
                    for index in indices
                ],
                axis=0,
            )

            split = row["subset"]
            output_dir = OUTPUT / split
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = (
                output_dir
                / f"{series}_ivd{row['ivd_label']}.npy"
            )
            np.save(output_file, stack)

            manifest.append({
                "patient_id": row["patient_id"],
                "series_name": series,
                "ivd_label": row["ivd_label"],
                "split": split,
                "herniation": row["disc_herniation"],
                "stack_path": str(output_file),
                "slice_indices": json.dumps(indices.tolist()),
                "selected_slice": selected,
                "anchor_box": json.dumps(anchor_box),
            })
            counts[(split, row["disc_herniation"])] += 1

        print("Processed:", series)

    manifest_path = OUTPUT / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient_id", "series_name", "ivd_label",
                "split", "herniation", "stack_path",
                "slice_indices", "selected_slice", "anchor_box",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest)

    print("Exported counts:", dict(counts))
    print("Skipped because target mask was absent on selected slice:")
    print("  total:", len(skipped))
    print("  positives:", sum(item[3] == "1" for item in skipped))
    for item in skipped[:15]:
        print(" ", item)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()