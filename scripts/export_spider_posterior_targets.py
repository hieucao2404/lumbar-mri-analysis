import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.export_spider_posterior_crops import (
    load_oriented_volume,
    make_context_crop,
)


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/processed/spider_posterior_crops/crops.csv"
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

    args = parser.parse_args()

    rows = read_manifest(args.manifest)

    if not rows:
        raise ValueError("Posterior crop manifest is empty")

    mask_dir = args.output_dir / "target_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    output_rows = []
    current_series = None
    mask_volume = None

    for row in rows:
        series_name = row["series_name"]
        ivd_label = int(row["ivd_label"])

        if series_name != current_series:
            mask_path = (
                args.raw_root
                / "masks"
                / f"{series_name}.mha"
            )

            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)

            mask_volume = load_oriented_volume(mask_path)
            current_series = series_name

        selected_slice = int(row["selected_slice"])

        if not 0 <= selected_slice < mask_volume.shape[2]:
            raise ValueError(
                f"{series_name}: invalid selected slice "
                f"{selected_slice}"
            )

        mask_slice = mask_volume[:, :, selected_slice]
        mask_value = int(row["mask_value"])

        expert_disc_mask = (
            mask_slice == mask_value
        ).astype(np.float32)

        if expert_disc_mask.sum() == 0:
            raise ValueError(
                f"{series_name} IVD {ivd_label}: "
                "expert disc mask is empty"
            )

        box = (
            int(row["selected_x1"]),
            int(row["selected_y1"]),
            int(row["selected_x2"]),
            int(row["selected_y2"]),
        )

        image = np.load(
            row["crop_path"],
            allow_pickle=False,
        )

        if image.shape != (160, 160):
            raise ValueError(
                f"{row['crop_path']}: expected (160, 160), "
                f"got {image.shape}"
            )

        # Reuse the exact crop geometry used for the MRI channel.
        # Threshold after resizing to obtain a binary indicator.
        resized_mask = make_context_crop(
            expert_disc_mask,
            box,
            output_size=160,
        )

        target_mask = (
            resized_mask >= 0.5
        ).astype(np.float32)

        if target_mask.sum() == 0:
            raise ValueError(
                f"{series_name} IVD {ivd_label}: "
                "target mask vanished during resizing"
            )

        if not np.isfinite(image).all():
            raise ValueError(
                f"{row['crop_path']}: non-finite image values"
            )

        target_path = (
            mask_dir
            / f"{series_name}_ivd{ivd_label:02d}_target.npy"
        )

        np.save(target_path, target_mask)

        output_rows.append({
            **row,
            "target_mask_path": str(target_path),
            "target_mask_pixels": int(target_mask.sum()),
        })

    output_manifest = (
        args.output_dir / "crops_with_targets.csv"
    )

    with output_manifest.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(output_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print("Records:", len(output_rows))
    print(
        "By subset:",
        dict(Counter(row["subset"] for row in output_rows)),
    )
    print(
        "Herniation positives:",
        dict(Counter(
            row["subset"]
            for row in output_rows
            if int(row["disc_herniation"]) == 1
        )),
    )
    print("First image:", output_rows[0]["crop_path"])
    print(
        "First target mask:",
        output_rows[0]["target_mask_path"],
    )
    print("Manifest:", output_manifest)


if __name__ == "__main__":
    main()