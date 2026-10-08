import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

class SpiderDiscMultitaskDataset(Dataset):
    """Return:
    
    image:
        Float tensor [1, 256, 256]
        
    binary_target:
        Float tensor [1, 256, 256]
        0 = background
        1 = any annotated disc
    
    level_target:
        Long tensor [256, 256]
        0   = background
        1   = L1-L2
        2   = L2-L3
        3   = L3-L4
        4   = L4-L5
        5   = L5-S1
        255 = annotated upper disc
        """

    def __init__(
        self,
        manifest_path: str | Path,
        subset: str,
    ) -> None:
        if subset not in {"training", "validation"}:
            raise ValueError(f"Unknown subset: {subset}")

        with Path(manifest_path).open(newline="") as file:
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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]: 
        row = self.rows[index]

        with np.load(
            row["example_path"],
            allow_pickle=False,
        ) as data:
            required_arrays = {
                "image",
                "mask",
                "level_mask",
            }

            if not required_arrays.issubset(data.files):
                raise ValueError(
                    f"{row['example_path']} contains"
                    f"{data.files}, expected {required_arrays}"
                )

            image = data["image"].astype(
                np.float32,
                copy=False,
            )

            binary_mask = data["mask"].astype(
                np.float32,
                copy=False
            )

            level_mask = data["level_mask"].astype(
                np.int64,
                copy=False,
            )
        expected_shape = (256, 256)

        if image.shape != expected_shape:
            raise ValueError(
                f"Unexpected image shape in "
                f"{row['series_name']}: {image.shape}"
            )

        if binary_mask.shape != expected_shape:
            raise ValueError(
                f"Unexpected binary-mask shape in "
                f"{row['series_name']}: "
                f"{binary_mask.shape}"
            )

        if level_mask.shape != expected_shape:
            raise ValueError(
                f"Unexpected level-mask shape in "
                f"{row['series_name']}: "
                f"{level_mask.shape}"
            )

        if not np.isfinite(image).all():
            raise ValueError(
                f"Non-finite image pixels in "
                f"{row['series_name']}"
            )

        binary_values = set(
            np.unique(binary_mask).tolist()
        )

        if not binary_values.issubset({0.0, 1.0}):
            raise ValueError(
                f"Unexpected binary-mask values: "
                f"{sorted(binary_values)}"
            )

        level_values = set(
            np.unique(level_mask).tolist()
        )

        valid_level_values = {
            0,
            1,
            2,
            3,
            4,
            5,
            255,
        }

        if not level_values.issubset(
            valid_level_values
        ):
            raise ValueError(
                f"Unexpected level-mask values: "
                f"{sorted(level_values)}"
            )

        image_tensor = torch.from_numpy(
            image,
        ).unsqueeze(0)

        binary_tensor = torch.from_numpy(
            binary_mask,
        ).unsqueeze(0)

        level_tensor = torch.from_numpy(
            level_mask,
        )

        return (
            image_tensor,
            binary_tensor,
            level_tensor,
        )