import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SpiderDiscLevelDataset(Dataset):
    """
    Load sagittal T2 MRI slices and level-aware segmentation masks.

    Image:
        Float tensor [1, 256, 256], range approximately [0, 1]

    Target:
        Integer tensor [256, 256]

        0 = Background
        1 = L1-L2
        2 = L2-L3
        3 = L3-L4
        4 = L4-L5
        5 = L5-S1
      255 = Ignore
    """

    VALID_SUBSETS = {
        "training",
        "validation",
    }

    VALID_TARGET_VALUES = {
        0,
        1,
        2,
        3,
        4,
        5,
        255,
    }

    def __init__(
        self,
        manifest_path: str | Path,
        subset: str,
    ) -> None:
        if subset not in self.VALID_SUBSETS:
            raise ValueError(
                f"Unknown subset: {subset}. "
                f"Expected one of {sorted(self.VALID_SUBSETS)}"
            )

        manifest_path = Path(manifest_path)

        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)

        with manifest_path.open(newline="") as file:
            all_rows = list(csv.DictReader(file))

        self.rows = [
            row
            for row in all_rows
            if row["subset"] == subset
        ]

        if not self.rows:
            raise ValueError(
                f"No examples found for subset {subset}"
            )

        self.subset = subset

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        example_path = Path(row["example_path"])

        if not example_path.is_file():
            raise FileNotFoundError(example_path)

        with np.load(
            example_path,
            allow_pickle=False,
        ) as data:
            if "image" not in data.files:
                raise KeyError(
                    f"{example_path} does not contain image"
                )

            if "level_mask" not in data.files:
                raise KeyError(
                    f"{example_path} does not contain level_mask. "
                    "Re-run export_spider_disc_localization."
                )

            image = data["image"].astype(
                np.float32,
                copy=False,
            )

            target = data["level_mask"].astype(
                np.int64,
                copy=False,
            )

        if image.shape != (256, 256):
            raise ValueError(
                f"{example_path}: expected image shape "
                f"(256, 256), got {image.shape}"
            )

        if target.shape != (256, 256):
            raise ValueError(
                f"{example_path}: expected target shape "
                f"(256, 256), got {target.shape}"
            )

        if not np.isfinite(image).all():
            raise ValueError(
                f"{example_path}: image contains "
                "non-finite pixels"
            )

        image_minimum = float(image.min())
        image_maximum = float(image.max())

        if image_minimum < 0.0 or image_maximum > 1.0:
            raise ValueError(
                f"{example_path}: image range is "
                f"[{image_minimum}, {image_maximum}]"
            )

        unique_targets = {
            int(value)
            for value in np.unique(target)
        }

        unexpected_targets = (
            unique_targets
            - self.VALID_TARGET_VALUES
        )

        if unexpected_targets:
            raise ValueError(
                f"{example_path}: unexpected target values "
                f"{sorted(unexpected_targets)}"
            )

        # Copy into contiguous arrays before converting to tensors.
        image = np.ascontiguousarray(image)
        target = np.ascontiguousarray(target)

        image_tensor = torch.from_numpy(
            image
        ).unsqueeze(0)

        target_tensor = torch.from_numpy(target)

        return image_tensor, target_tensor