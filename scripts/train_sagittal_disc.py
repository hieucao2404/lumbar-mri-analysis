import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.data.disc_dataset import DiscDataset
from src.models.sagittal_disc_cnn import SagittalDiscCNN
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


MANIFEST = Path("disc.csv")
IMAGE_SIZE = 128
LEVELS = {"L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"}


def load_rows(manifest_path):
    rows = []
    patient_splits = defaultdict(set)

    with manifest_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for line_number, raw in enumerate(reader, start=2):
            patient_id = raw["patient_id"].strip()
            split = raw["split"].strip()
            level = raw["level"].strip()

            if split not in {"train", "val", "test"}:
                raise ValueError(f"Line {line_number}: invalid split")
            if level not in LEVELS:
                raise ValueError(f"Line {line_number}: invalid level")

            image_path = manifest_path.parent / raw["image_path"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)

            box = tuple(
                int(raw[key]) for key in ("x1", "y1", "x2", "y2")
            )
            x1, y1, x2, y2 = box

            with Image.open(image_path) as image:
                width, height = image.size

            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(
                    f"Line {line_number}: box {box} outside image"
                )

            pfirrmann = int(raw["pfirrmann"])
            bulge = int(raw["bulge"])
            herniation = int(raw["herniation"])

            if pfirrmann not in {-1, 1, 2, 3, 4, 5}:
                raise ValueError(f"Line {line_number}: invalid Pfirrman grade")
            if bulge not in {-1, 0 , 1}:
                raise ValueError(f"Line {line_number}: invalid bulge label")
            if herniation not in {-1, 0, 1}:
                raise ValueError(f"Line {line_number}: invalid herniation label")

            rows.append(
            {    "patient_id": patient_id,
                "image_path": image_path,
                "level": level,
                "box": box,
                "pfirrmann": pfirrmann,
                "bulge": bulge,
                "herniation": herniation,
                "split": split,
            })
            patient_splits[patient_id].add(split)
    leaked = [
        patient for patient, splits in patient_splits.items()
        if len(splits) > 1
    ]
    if leaked:
        raise ValueError(f"Patients appear in multiple splits: {leaked[:5]}")

    for split in ("train", "val", "test"):
        subset = [row for row in rows if row["split"] == split]
        print(f"{split}: {len(subset)} disc regions")

    return rows

def compute_loss(predictions, grades, bulges, herniations):
        losses = []

        known_grades = grades >= 0 
        if known_grades.any():
            losses.append(
                nn.functional.cross_entropy(
                    predictions["pfirrmann"][known_grades],
                    grades[known_grades],
                )
            )

        known_bulges = bulges >= 0
        if known_bulges.any():
            losses.append(
                nn.functional.binary_cross_entropy_with_logits(
                    predictions["bulge"][known_bulges],
                    bulges[known_bulges],
                )
            )

        known_herniations = herniations >= 0
        if known_herniations.any():
            losses.append(
                nn.functional.binary_cross_entropy_with_logits(
                    predictions["herniation"][known_herniations],
                    herniations[known_herniations],
                )
            )

        if not losses:
            raise ValueError("Batch has no usable labels")

        return sum(losses) / len(losses)


def run_epoch(model, loader, optimizer, device):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0

    for images, grades, bulges, herniations in loader:
        images = images.to(device)
        grades = grades.to(device)
        bulges = bulges.to(device)
        herniations = herniations.to(device)

        with torch.set_grad_enabled(training):
            predictions = model(images)
            loss = compute_loss(
                predictions, grades, bulges, herniations
            )

            if training: 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def main(): 
    torch.manual_seed(42)
    rows = load_rows(MANIFEST)

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]

    if not train_rows or not val_rows:
        raise ValueError("Need both train and val patient splits")

    train_loader = DataLoader(
        DiscDataset(train_rows, IMAGE_SIZE),
        batch_size=16,
        shuffle=True,
    )
    val_loader = DataLoader(
        DiscDataset(val_rows, IMAGE_SIZE),
        batch_size=16,
        shuffle=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SagittalDiscCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")

    for epoch in range(1, 21):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        val_loss = run_epoch(model, val_loader, None, device)

        print(
            f"Epoch {epoch:02d}: "
            f"train={train_loss:.4f}, val={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "image_size": IMAGE_SIZE,
                },
                "best_sagittal_disc.pt",
            )

    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()