import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SpiderPfirrmannDataset(Dataset):
    def __init__(self, manifest_path: str | Path, subset: str):
        if subset not in {"training", "validation"}:
            raise ValueError(f"Unknown subset: {subset}")

        with Path(manifest_path).open(newline="") as file:
            all_rows = list(csv.DictReader(file))

        self.rows = [
            row for row in all_rows if row["subset"] == subset
        ]

        if not self.rows:
            raise ValueError(f"No samples for subset {subset}")

        self.subset = subset

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]

        image = np.load(row["crop_path"], allow_pickle=False)
        if image.shape != (160, 160):
            raise ValueError(
                f"{row['crop_path']}: expected (160, 160), "
                f"got {image.shape}"
            )
        if not np.isfinite(image).all():
            raise ValueError(f"Non-finite pixels: {row['crop_path']}")

        # [H, W] -> [C, H, W], with one grayscale channel.
        image_tensor = torch.from_numpy(
            image.astype(np.float32, copy=False)
        ).unsqueeze(0)

        grade = int(row["pfirrmann_grade"])
        if grade not in range(1, 6):
            raise ValueError(f"Unexpected Pfirrmann grade: {grade}")

        # Cross-entropy expects class IDs 0–4.
        target = grade - 1

        return image_tensor, target