import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SpiderPosteriorDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        subset: str,
    ) -> None:
        if subset not in {"training", "validation"}:
            raise ValueError(f"Unknown subset: {subset}")

        with Path(manifest_path).open(newline="") as file:
            rows = list(csv.DictReader(file))

        self.rows = [
            row for row in rows
            if row["subset"] == subset
        ]

        if not self.rows:
            raise ValueError(f"No rows for subset {subset}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]

        image = np.load(
            row["crop_path"],
            allow_pickle=False,
        ).astype(np.float32)

        target_mask = np.load(
            row["target_mask_path"],
            allow_pickle=False,
        ).astype(np.float32)

        if image.shape != (160, 160):
            raise ValueError(
                f"{row['crop_path']}: image shape {image.shape}"
            )

        if target_mask.shape != image.shape:
            raise ValueError(
                f"{row['target_mask_path']}: mask shape "
                f"{target_mask.shape}"
            )

        if not (
            np.isfinite(image).all()
            and np.isfinite(target_mask).all()
        ):
            raise ValueError(
                f"Non-finite values in {row['series_name']}"
            )

        if not 0.0 <= float(image.min()) <= float(image.max()) <= 1.0:
            raise ValueError("MRI crop is outside [0, 1]")

        if not set(np.unique(target_mask)).issubset(
            {0.0, 1.0}
        ):
            raise ValueError("Target mask must be binary")

        if target_mask.sum() == 0:
            raise ValueError("Target-disc mask is empty")

        # Channel 0: MRI image.
        # Channel 1: location of the disc being classified.
        inputs = np.stack(
            [image, target_mask],
            axis=0,
        )

        labels = np.array(
            [
                int(row["disc_bulging"]),
                int(row["disc_herniation"]),
            ],
            dtype=np.float32,
        )

        if not set(labels).issubset({0.0, 1.0}):
            raise ValueError(
                f"Invalid posterior labels: {labels}"
            )

        return (
            torch.from_numpy(inputs),
            torch.from_numpy(labels),
        )