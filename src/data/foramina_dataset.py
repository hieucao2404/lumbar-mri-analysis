import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


SEVERITY_NAMES = ("Normal", "Mild", "Moderate", "Severe")


def load_foramina_manifest(manifest_path):
    manifest_path = Path(manifest_path).resolve()

    with manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    processed_rows = []

    for line_number, row in enumerate(rows, start=2):
        image_path = (
            manifest_path.parent / row["image_path"]
        ).resolve()

        if not image_path.is_file():
            raise FileNotFoundError(
                f"CSV line {line_number}: image missing: {image_path}"
            )

        processed_row = {
            "patient_id": row["patient_id"],
            "image_path": image_path,
            "box": (
                int(row["x1"]),
                int(row["y1"]),
                int(row["x2"]),
                int(row["y2"]),
            ),
            "level": row["level"],
            "side": row["side"],
            "severity": int(row["severity"]),
            "split": row["split"],
        }

        if processed_row["severity"] not in range(4):
            raise ValueError(
                f"CSV line {line_number}: invalid severity"
            )

        processed_rows.append(processed_row)

    return processed_rows


def expand_box(box, image_size, margin_ratio):
    x1, y1, x2, y2 = box
    image_width, image_height = image_size

    width = x2 - x1
    height = y2 - y1

    margin_x = round(width * margin_ratio)
    margin_y = round(height * margin_ratio)

    return (
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(image_width, x2 + margin_x),
        min(image_height, y2 + margin_y),
    )


class ForaminaDataset(Dataset):
    def __init__(
        self,
        rows,
        image_size=128,
        margin_ratio=0.20,
    ):
        self.rows = rows
        self.image_size = image_size
        self.margin_ratio = margin_ratio

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        with Image.open(row["image_path"]) as source:
            image = source.convert("L")

            expanded_box = expand_box(
                row["box"],
                image.size,
                self.margin_ratio,
            )

            crop = image.crop(expanded_box)

            # Preserve aspect ratio and pad to a square.
            crop = ImageOps.pad(
                crop,
                (self.image_size, self.image_size),
                method=Image.Resampling.BILINEAR,
                color=0,
            )

        pixels = np.asarray(crop, dtype=np.float32) / 255.0

        # Shape changes from [H, W] to [1, H, W].
        image_tensor = torch.from_numpy(pixels).unsqueeze(0)

        severity = torch.tensor(
            row["severity"],
            dtype=torch.long,
        )

        return {
            "image": image_tensor,
            "severity": severity,
            "patient_id": row["patient_id"],
            "level": row["level"],
            "side": row["side"],
        }