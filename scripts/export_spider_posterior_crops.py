import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import SimpleITK as sitk


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def load_oriented_volume(path: Path) -> np.ndarray:
    image = sitk.DICOMOrient(
        sitk.ReadImage(str(path)),
        "LPS",
    )

    # Match the orientation used by the existing SPIDER exporters.
    return np.flip(
        sitk.GetArrayFromImage(image),
        axis=0,
    )


def normalize_slice(image: np.ndarray) -> np.ndarray:
    valid = image[np.isfinite(image) & (image > 0)]

    if valid.size == 0:
        raise ValueError("Slice contains no positive finite pixels")

    low, high = np.percentile(valid, (1, 99))

    if high <= low:
        raise ValueError("Slice has no useful intensity range")

    result = np.clip(
        (image.astype(np.float32) - low) / (high - low),
        0.0,
        1.0,
    )

    result[~np.isfinite(result)] = 0.0
    return result.astype(np.float32)


def mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)

    if len(xs) == 0:
        raise ValueError("Disc mask is empty")

    # Exclusive maximum coordinates.
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def make_context_crop(
    normalized_image: np.ndarray,
    box: tuple[int, int, int, int],
    output_size: int,
) -> np.ndarray:
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1

    if box_width <= 0 or box_height <= 0:
        raise ValueError(f"Invalid disc box: {box}")

    # Include the disc, adjacent vertebral tissue, and a wider area
    # behind the disc. The reviewed LPS images have posterior tissue
    # toward increasing image x.
    center_x = (x1 + x2) / 2 + 0.35 * box_width
    center_y = (y1 + y2) / 2

    half_side = int(
        np.ceil(
            max(
                2.0 * box_width,
                2.5 * box_height,
                48,
            )
        )
    )

    left = int(round(center_x)) - half_side
    top = int(round(center_y)) - half_side
    right = left + 2 * half_side
    bottom = top + 2 * half_side

    height, width = normalized_image.shape

    source_left = max(0, left)
    source_top = max(0, top)
    source_right = min(width, right)
    source_bottom = min(height, bottom)

    canvas = np.zeros(
        (2 * half_side, 2 * half_side),
        dtype=np.float32,
    )

    if (
        source_right > source_left
        and source_bottom > source_top
    ):
        target_left = source_left - left
        target_top = source_top - top

        canvas[
            target_top:target_top + source_bottom - source_top,
            target_left:target_left + source_right - source_left,
        ] = normalized_image[
            source_top:source_bottom,
            source_left:source_right,
        ]

    resized = Image.fromarray(canvas).resize(
        (output_size, output_size),
        Image.Resampling.BILINEAR,
    )

    return np.asarray(resized, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--disc-manifest",
        type=Path,
        default=Path("data/processed/spider_crops/crops.csv"),
    )
    parser.add_argument(
        "--localization-manifest",
        type=Path,
        default=Path(
            "data/processed/spider_disc_localization/manifest.csv"
        ),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/spider"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/spider_posterior_crops"),
    )
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--minimum-mask-area", type=int, default=25)

    args = parser.parse_args()

    if args.size <= 0:
        raise ValueError("--size must be positive")

    disc_rows = read_csv(args.disc_manifest)
    localization_rows = read_csv(args.localization_manifest)

    localization_by_series = {
        row["series_name"]: row
        for row in localization_rows
    }

    if len(localization_by_series) != len(localization_rows):
        raise ValueError(
            "Localization manifest contains duplicate series names"
        )

    grouped = defaultdict(list)

    for row in disc_rows:
        series_name = row["series_name"]
        ivd_label = int(row["ivd_label"])

        # SPIDER IVD 1..5 correspond to the five lumbar levels
        # established in the earlier disc-order audit.
        if (
            series_name.endswith("_t2")
            and 1 <= ivd_label <= 5
        ):
            grouped[series_name].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    exported = []
    skipped = []
    source_slice_counts = Counter()

    for series_name, rows in sorted(grouped.items()):
        localization = localization_by_series.get(series_name)

        if localization is None:
            skipped.extend(
                (series_name, row["ivd_label"], "no localization row")
                for row in rows
            )
            continue

        image_path = (
            args.raw_root / "images" / f"{series_name}.mha"
        )
        mask_path = (
            args.raw_root / "masks" / f"{series_name}.mha"
        )

        if not image_path.is_file() or not mask_path.is_file():
            skipped.extend(
                (series_name, row["ivd_label"], "missing image or mask")
                for row in rows
            )
            continue

        image_volume = load_oriented_volume(image_path)
        mask_volume = load_oriented_volume(mask_path)

        if image_volume.shape != mask_volume.shape:
            raise ValueError(
                f"{series_name}: image/mask shape mismatch: "
                f"{image_volume.shape} vs {mask_volume.shape}"
            )

        selected_slice = int(localization["sagittal_slice"])

        if not 0 <= selected_slice < image_volume.shape[2]:
            raise ValueError(
                f"{series_name}: invalid selected slice "
                f"{selected_slice}"
            )

        image_slice = image_volume[:, :, selected_slice]
        mask_slice = mask_volume[:, :, selected_slice]
        normalized = normalize_slice(image_slice)

        for row in rows:
            ivd_label = int(row["ivd_label"])
            mask_value = int(row["mask_value"])

            disc_mask = mask_slice == mask_value
            area = int(disc_mask.sum())

            if area < args.minimum_mask_area:
                skipped.append(
                    (
                        series_name,
                        ivd_label,
                        f"mask area {area} below "
                        f"{args.minimum_mask_area}",
                    )
                )
                continue

            box = mask_box(disc_mask)
            crop = make_context_crop(
                normalized,
                box,
                args.size,
            )

            crop_path = (
                args.output_dir
                / f"{series_name}_ivd{ivd_label:02d}.npy"
            )
            np.save(crop_path, crop)

            original_slice = int(row["sagittal_slice"])
            same_slice = original_slice == selected_slice

            source_slice_counts[
                (row["subset"], same_slice)
            ] += 1

            exported.append({
                **row,
                "original_crop_path": row["crop_path"],
                "crop_path": str(crop_path),
                "original_crop_slice": original_slice,
                "selected_slice": selected_slice,
                "same_slice_as_original_crop": int(same_slice),
                "selected_mask_area": area,
                "selected_x1": box[0],
                "selected_y1": box[1],
                "selected_x2": box[2],
                "selected_y2": box[3],
            })

        print(
            f"{series_name}: selected slice {selected_slice}, "
            f"exported {sum(r['series_name'] == series_name for r in exported)}"
        )

    if not exported:
        raise RuntimeError("No posterior crops were exported")

    manifest_path = args.output_dir / "crops.csv"

    with manifest_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(exported[0].keys()),
        )
        writer.writeheader()
        writer.writerows(exported)

    print("\nExport complete")
    print("Crops:", len(exported))
    print(
        "By subset:",
        dict(Counter(row["subset"] for row in exported)),
    )
    print(
        "Herniation positives by subset:",
        dict(Counter(
            row["subset"]
            for row in exported
            if int(row["disc_herniation"]) == 1
        )),
    )
    print(
        "Original and selected slice comparison:",
        dict(source_slice_counts),
    )
    print("Skipped discs:", len(skipped))

    for item in skipped[:15]:
        print("  Skipped:", item)

    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()