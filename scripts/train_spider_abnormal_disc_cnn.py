"""Train a binary disc-review CNN on SPIDER bulging OR herniation labels.

The output means axial review is recommended. It does not classify the
abnormality as a herniation, and it does not diagnose from sagittal MRI.
"""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.models.abnormal_disc_cnn import AbnormalDiscCNN


def abnormal(row):
    return int(row["status"] != "normal")


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def check_rows(training, natural, balanced):
    def keys(rows):
        result = [(row["series_name"], row["level"]) for row in rows]
        if len(result) != len(set(result)):
            raise ValueError("Duplicate series and level in a split")
        return set(result)

    train_keys, val_keys, balanced_keys = map(keys, (training, natural, balanced))
    if train_keys & val_keys:
        raise ValueError("Training and validation share discs")
    train_patients = {row["patient_id"] for row in training}
    val_patients = {row["patient_id"] for row in natural}
    if train_patients & val_patients:
        raise ValueError("Training and validation share patients")
    if not balanced_keys <= val_keys:
        raise ValueError("Balanced diagnostic must be a validation subset")
    for rows in (training, natural, balanced):
        for row in rows:
            expected = (
                "herniation" if row["herniation"] == "1" else
                "bulging_only" if row["bulging"] == "1" else "normal"
            )
            if row["status"] != expected:
                raise ValueError(f"Inconsistent labels: {row['series_name']} {row['level']}")
            if not Path(row["stack_path"]).is_file():
                raise FileNotFoundError(row["stack_path"])
    if not training or not natural:
        raise ValueError("Training and natural validation must not be empty")
    if sum(row["herniation"] == "1" for row in natural) == 0:
        raise ValueError("Natural validation has no positive discs")


class AbnormalStacks(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        stack = np.load(row["stack_path"], allow_pickle=False)
        if stack.shape != (8, 1, 160, 160):
            raise ValueError(f"Unexpected stack: {row['stack_path']}")
        if stack.dtype != np.uint8:
            raise ValueError(f"Expected uint8: {row['stack_path']}")
        image = torch.from_numpy(stack.astype(np.float32) / 255.0)
        return image, torch.tensor(float(abnormal(self.rows[index])), dtype=torch.float32)


def patient_folds(rows, count, seed):
    labels = np.array([abnormal(row) for row in rows])
    patients = np.array([row["patient_id"] for row in rows])
    result = np.full(len(rows), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed)
    for fold, (train, held_out) in enumerate(splitter.split(np.zeros(len(rows)), labels, patients)):
        if set(patients[train]) & set(patients[held_out]):
            raise ValueError("Patient overlap between training and held-out fold")
        if not labels[train].any() or not labels[held_out].any():
            raise ValueError("Fold without abnormal training or held-out discs")
        result[held_out] = fold
    if np.any(result < 0):
        raise ValueError("Some discs have no held-out fold")
    return result


def sampling_weights(rows):
    counts = Counter((abnormal(row), row["patient_id"]) for row in rows)
    patients_per_class = Counter(label for label, _ in counts)
    if set(patients_per_class) != {0, 1}:
        raise ValueError("Training needs normal and abnormal discs")
    return torch.tensor([
        0.5 / patients_per_class[abnormal(row)]
        / counts[(abnormal(row), row["patient_id"])]
        for row in rows
    ], dtype=torch.double)


def train_model(rows, epochs, seed, device, label):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    weights = sampling_weights(rows)
    sampler = WeightedRandomSampler(
        weights, num_samples=len(rows), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(AbnormalStacks(rows), batch_size=8, sampler=sampler)
    model = AbnormalDiscCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        for images, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
        print(f"{label} epoch={epoch + 1}/{epochs}", flush=True)
    return model


@torch.inference_mode()
def score_model(model, rows, device):
    model.eval()
    loader = DataLoader(AbnormalStacks(rows), batch_size=16)
    return np.concatenate([
        model(images.to(device)).sigmoid().cpu().numpy()
        for images, _ in loader
    ])


def save_predictions(path, rows, scores, folds=None):
    fields = ["patient_id", "series_name", "level", "status", "abnormal", "abnormal_score"]
    if folds is not None:
        fields.append("fold")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            record = {key: row[key] for key in fields[:4]}
            record.update(abnormal=abnormal(row), abnormal_score=float(scores[i]))
            if folds is not None:
                record["fold"] = int(folds[i])
            writer.writerow(record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(
        "data/processed/spider_bulge_herniation_banded_v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("oof", "full"), required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.folds < 2 or args.epochs < 1:
        raise ValueError("Need at least two folds and positive epochs")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    training = read_rows(args.data / "training.csv")
    validation = read_rows(args.data / "validation_natural.csv")
    balanced = read_rows(args.data / "validation_balanced.csv")
    check_rows(training, validation, balanced)
    args.output.mkdir(parents=True)
    if args.mode == "oof":
        assignments = patient_folds(training, args.folds, args.seed)
        scores = np.zeros(len(training), dtype=np.float32)
        for fold in range(args.folds):
            train = [row for i, row in enumerate(training) if assignments[i] != fold]
            held_out = [row for i, row in enumerate(training) if assignments[i] == fold]
            model = train_model(train, args.epochs, args.seed + fold, device,
                                f"fold={fold + 1}/{args.folds}")
            scores[assignments == fold] = score_model(model, held_out, device)
            del model
        rows = training
        save_predictions(args.output / "oof_predictions.csv", rows, scores, assignments)
    else:
        model = train_model(training, args.epochs, args.seed, device, "full")
        torch.save({
            "model_state_dict": model.cpu().state_dict(),
            "model_kind": "abnormal_disc_cnn", "epochs": args.epochs,
            "seed": args.seed, "selection": "fixed epochs; no validation selection",
            "positive_class": "bulging or herniation; axial review recommended",
            "level_balanced": False,
            "level_mix": 0.0,
            "normal_share": 0.5,
            "encoder_checkpoint": None,
            "natural_sampling": False,
        }, args.output / "fixed.pt")
        model.to(device)
        rows = validation
        scores = score_model(model, rows, device)
        save_predictions(args.output / "validation_predictions.csv", rows, scores)
    truth = np.array([abnormal(row) for row in rows])
    (args.output / "metrics.json").write_text(json.dumps({
        "mode": args.mode, "epochs": args.epochs, "seed": args.seed,
        "discs": len(rows), "abnormal_discs": int(truth.sum()),
        "abnormal_disc_ap": float(average_precision_score(truth, scores)),
        "sampler": "normal_share=0.5; level_mix=0.0; encoder_checkpoint=None; "
                   "base=equal binary classes; patient balanced within groups",
        "note": "SPIDER validation has been reused; exploratory only.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
