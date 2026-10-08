import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SpiderDiscLocalizationDataset(Dataset):
    def __init__(self, manifest_path: str | Path, subset: str):
        self.manifest_path = Path(manifest_path)

        with self.manifest_path.open(newline="") as file:
            rows = list(csv.DictReader(file))

        self.rows = [
            row for row in rows
            if row["subset"] == subset
        ]

        if not self.rows:
            raise ValueError(
                f"No examples for subset {subset!r} "
                f"in {self.manifest_path}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]

        with np.load(row["example_path"]) as data:
            image = data["image"].astype(np.float32)
            mask = data["mask"].astype(np.float32)

        if image.shape != mask.shape:
            raise ValueError(
                f"Image/mask shape mismatch: {row['series_name']}"
            )

        if image.ndim != 2:
            raise ValueError(
                f"Expected a 2D image: {row['series_name']}"
            )

        if not np.isin(mask, [0.0, 1.0]).all():
            raise ValueError(
                f"Expected a binary mask: {row['series_name']}"
            )

        # [1, height, width] for a one-channel CNN.
        image_tensor = torch.from_numpy(
            image[None, :, :].copy()
        )

        mask_tensor = torch.from_numpy(
            mask[None, :, :].copy()
        )

        return image_tensor, mask_tensor