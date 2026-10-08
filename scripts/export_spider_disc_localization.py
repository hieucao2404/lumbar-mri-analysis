import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import SimpleITK as sitk


def normalize_slice(image: np.ndarray) -> np.ndarray:
    valid = image[np.isfinite(image) & (image > 0)]

    if valid.size == 0:
        raise ValueError("MRI slice contains no positive finite pixels")

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


def select_slice(mask_volume: np.ndarray) -> tuple[int, int]:
    """
    mask_volume shape after LPS orientation and flip: [z, y, x].
    Axis 2 is the sagittal slice axis in this project.
    """
    best_index = -1
    best_score = (-1, -1)
    best_count = 0

    for slice_index in range(mask_volume.shape[2]):
        mask_slice = mask_volume[:, :, slice_index]

        # SPIDER disc instance IDs are in the 200 range.
        disc_pixels = (
            (mask_slice >= 201) & (mask_slice < 300)
        )

        labels, counts = np.unique(
            mask_slice[disc_pixels],
            return_counts=True,
        )

        # A few stray pixels should not count as a visible disc.
        visible_count = int(np.count_nonzero(counts >= 30))
        disc_area = int(disc_pixels.sum())

        score = (visible_count, disc_area)

        if score > best_score:
            best_score = score
            best_index = slice_index
            best_count = visible_count

    if best_index < 0 or best_count == 0:
        raise ValueError("No disc instances visible in any sagittal slice")

    return best_index, best_count


def letterbox_pair(
    image: np.ndarray,
    mask: np.ndarray,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape

    scale = min(
        output_size / width,
        output_size / height,
    )

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    resized_image = np.asarray(
        Image.fromarray(image, mode="F").resize(
            (new_width, new_height),
            Image.Resampling.BILINEAR,
        ),
        dtype=np.float32,
    )

    resized_mask = np.asarray(
        Image.fromarray(mask.astype(np.uint8), mode="L").resize(
            (new_width, new_height),
            Image.Resampling.NEAREST,
        ),
        dtype=np.uint8,
    )

    image_canvas = np.zeros(
        (output_size, output_size),
        dtype=np.float32,
    )

    mask_canvas = np.zeros(
        (output_size, output_size),
        dtype=np.uint8,
    )

    offset_x = (output_size - new_width) // 2
    offset_y = (output_size - new_height) // 2

    image_canvas[
        offset_y:offset_y + new_height,
        offset_x:offset_x + new_width,
    ] = resized_image

    mask_canvas[
        offset_y:offset_y + new_height,
        offset_x:offset_x + new_width,
    ] = resized_mask

    return image_canvas, mask_canvas

def create_level_mask(
    instance_mask: np.ndarray,
) -> np.ndarray:
    """
    Convert SPIDER instance IDs into lumbar-level classes.

    Classes:
        0 = Background
        1 = L1-L2
        2 = L2-L3
        3 = L3-L4
        4 = L4-L5
        5 = L5-S1
      255 = Ignore upper non-lumbar discs
    """

    level_mask = np.zeros(
        instance_mask.shape,
        dtype=np.uint8,
    )

    # Discs above L1-L2 are not target lumbar levels.
    # Ignore these pixels during training instead of treating
    # them as background.
    upper_disc_pixels = (
        (instance_mask >= 206)
        & (instance_mask < 300)
    )

    level_mask[upper_disc_pixels] = 255

    raw_to_class = {
        205: 1,  # L1-L2
        204: 2,  # L2-L3
        203: 3,  # L3-L4
        202: 4,  # L4-L5
        201: 5,  # L5-S1
    }

    for raw_value, class_id in raw_to_class.items():
        level_mask[
            instance_mask == raw_value
        ] = class_id

    return level_mask

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/spider"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/spider_disc_localization"),
    )

    parser.add_argument(
        "--size",
        type=int,
        default=256,
    )

    args = parser.parse_args()

    overview_path = args.root / "overview.csv"

    with overview_path.open(newline="") as file:
        overview_rows = list(csv.DictReader(file))

    # Preserve SPIDER's patient-level training/validation split.
    t2_rows = [
        row
        for row in overview_rows
        if row["new_file_name"].endswith("_t2")
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_rows = []
    skipped = []

    for row in t2_rows:
        series_name = row["new_file_name"]
        patient_id = series_name.rsplit("_", 1)[0]

        image_path = args.root / "images" / f"{series_name}.mha"
        mask_path = args.root / "masks" / f"{series_name}.mha"

        if not image_path.is_file() or not mask_path.is_file():
            skipped.append((series_name, "missing image or mask"))
            continue

        try:
            image_sitk = sitk.DICOMOrient(
                sitk.ReadImage(str(image_path)),
                "LPS",
            )

            mask_sitk = sitk.DICOMOrient(
                sitk.ReadImage(str(mask_path)),
                "LPS",
            )

            if image_sitk.GetSize() != mask_sitk.GetSize():
                raise ValueError("image and mask sizes differ")

            # Match your existing SPIDER crop orientation.
            image_volume = np.flip(
                sitk.GetArrayFromImage(image_sitk),
                axis=0,
            )

            mask_volume = np.flip(
                sitk.GetArrayFromImage(mask_sitk),
                axis=0,
            )

            if image_volume.shape != mask_volume.shape:
                raise ValueError("image and mask arrays differ")

            slice_index, visible_discs = select_slice(mask_volume)

            image_slice = image_volume[:, :, slice_index]
            mask_slice = mask_volume[:, :, slice_index]

            disc_mask = (
                (mask_slice >= 201) & (mask_slice < 300)
            ).astype(np.uint8)

            level_mask = create_level_mask(mask_slice)

            normalized = normalize_slice(image_slice)

            image_output, mask_output = letterbox_pair(
                normalized,
                disc_mask,
                args.size,
            )

            _, level_mask_output = letterbox_pair(
                normalized,
                level_mask,
                args.size,
            )

            if mask_output.sum() == 0:
                raise ValueError("disc mask vanished after resizing")

            output_path = args.output_dir / f"{series_name}.npz"

            np.savez_compressed(
                output_path,
                image=image_output,
                mask=mask_output,
                level_mask=level_mask_output,
            )

            output_rows.append({
                "patient_id": patient_id,
                "series_name": series_name,
                "subset": row["subset"],
                "sagittal_slice": slice_index,
                "visible_discs": visible_discs,
                "height": image_slice.shape[0],
                "width": image_slice.shape[1],
                "example_path": str(output_path),
            })

        except (ValueError, RuntimeError) as error:
            skipped.append((series_name, str(error)))

    if not output_rows:
        raise RuntimeError("No localization examples were exported")

    manifest_path = args.output_dir / "manifest.csv"

    with manifest_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(output_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print("T2 series found:", len(t2_rows))
    print("Examples saved:", len(output_rows))
    print(
        "Examples by subset:",
        dict(Counter(row["subset"] for row in output_rows)),
    )
    print(
        "Visible-disc counts:",
        dict(Counter(row["visible_discs"] for row in output_rows)),
    )
    print("Skipped:", len(skipped))

    for series_name, reason in skipped[:10]:
        print(" ", series_name, reason)

    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()