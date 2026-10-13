import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LEVEL_NAMES = {
    1: "L5-S1",
    2: "L4-L5",
    3: "L3-L4",
    4: "L2-L3",
    5: "L1-L2",
}

COMBINATIONS = (
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/spider_crops/crops.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/spider_posterior_label_review"),
    )
    parser.add_argument("--per-group", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.per_group < 1:
        raise ValueError("--per-group must be positive")

    with args.manifest.open(newline="") as file:
        rows = list(csv.DictReader(file))

    groups = defaultdict(list)

    for row in rows:
        if not row["series_name"].endswith("_t2"):
            continue

        ivd_label = int(row["ivd_label"])

        if ivd_label not in range(1, 6):
            continue

        subset = row["subset"]
        bulge = int(row["disc_bulging"])
        herniation = int(row["disc_herniation"])

        if subset not in {"training", "validation"}:
            continue

        if (bulge, herniation) not in COMBINATIONS:
            raise ValueError(f"Unexpected labels: {row}")

        groups[(subset, bulge, herniation)].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    for subset in ("training", "validation"):
        fig, axes = plt.subplots(
            len(COMBINATIONS),
            args.per_group,
            figsize=(3.2 * args.per_group, 12),
            squeeze=False,
        )

        for row_index, (bulge, herniation) in enumerate(
            COMBINATIONS
        ):
            candidates = groups[
                (subset, bulge, herniation)
            ]

            # Avoid showing several discs from one patient within
            # the same label group when possible.
            rng.shuffle(candidates)

            selected = []
            selected_patients = set()

            for record in candidates:
                patient = record["patient_id"]

                if patient in selected_patients:
                    continue

                selected.append(record)
                selected_patients.add(patient)

                if len(selected) == args.per_group:
                    break

            for column_index, axis in enumerate(
                axes[row_index]
            ):
                axis.axis("off")

                if column_index >= len(selected):
                    continue

                record = selected[column_index]
                path = Path(record["crop_path"])
                crop = np.load(path, allow_pickle=False)

                if crop.shape != (160, 160):
                    raise ValueError(
                        f"{path}: expected (160, 160), "
                        f"got {crop.shape}"
                    )

                axis.imshow(
                    crop,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )

                level = LEVEL_NAMES[
                    int(record["ivd_label"])
                ]

                axis.set_title(
                    f"Patient {record['patient_id']}\n"
                    f"{level} | B={bulge}, H={herniation}",
                    fontsize=9,
                )

            axes[row_index, 0].set_ylabel(
                f"Bulge={bulge}\nHerniation={herniation}\n"
                f"n={len(candidates)}",
                rotation=0,
                labelpad=55,
                va="center",
            )

        fig.suptitle(
            f"SPIDER lumbar sagittal T2 crops — {subset}",
            fontsize=15,
        )
        fig.tight_layout(
            rect=(0.04, 0, 1, 0.96)
        )

        output_path = (
            args.output_dir
            / f"{subset}_posterior_crops.png"
        )

        fig.savefig(
            output_path,
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)

        print("Saved:", output_path)


if __name__ == "__main__":
    main()