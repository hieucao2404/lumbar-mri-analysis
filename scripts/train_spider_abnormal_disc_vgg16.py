"""Patient-separated frozen-VGG16 experiment for abnormal disc review."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from scripts.train_spider_abnormal_disc_cnn import (
    AbnormalStacks,
    abnormal,
    patient_folds,
    read_rows,
    sampling_weights,
    save_predictions,
)
from src.models.abnormal_disc_vgg16 import VGG16AbnormalDisc


@torch.no_grad()
def encode_rows(model, rows, device):
    model.eval()
    loader = DataLoader(AbnormalStacks(rows), batch_size=2, shuffle=False)
    parts = [
        model.encode(images.to(device)).cpu()
        for images, _ in loader
    ]
    return torch.cat(parts, dim=0)


def train_head(model, features, rows, epochs, seed, device, label):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model.reset_head()
    model.slice_head.to(device)
    weights = sampling_weights(rows)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(rows),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    targets = torch.tensor(
        [float(abnormal(row)) for row in rows],
        dtype=torch.float32,
    )
    loader = DataLoader(
        TensorDataset(features, targets),
        batch_size=32,
        sampler=sampler,
    )
    optimizer = torch.optim.AdamW(
        model.slice_head.parameters(), lr=1e-3, weight_decay=1e-4
    )
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.slice_head.train()
        for batch_features, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model.logits_from_features(
                batch_features.to(device)
            )
            loss = loss_fn(logits, batch_targets.to(device))
            loss.backward()
            optimizer.step()
        print(f"{label} epoch={epoch + 1}/{epochs}", flush=True)


@torch.no_grad()
def score_features(model, features, device):
    model.slice_head.eval()
    parts = []
    for start in range(0, len(features), 128):
        logits = model.logits_from_features(
            features[start:start + 128].to(device)
        )
        parts.append(logits.sigmoid().cpu().numpy())
    return np.concatenate(parts)


def check_manifests(training, validation):
    if not training or not validation:
        raise ValueError("Training and validation must be nonempty")

    train_patients = {row["patient_id"] for row in training}
    val_patients = {row["patient_id"] for row in validation}
    if train_patients & val_patients:
        raise ValueError("Patient overlap between splits")

    for rows in (training, validation):
        keys = [(row["series_name"], row["level"]) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate disc")
        for row in rows:
            if row["status"] not in {
                "normal", "bulging_only", "herniation"
            }:
                raise ValueError(f"Invalid status: {row}")
            if not Path(row["stack_path"]).is_file():
                raise FileNotFoundError(row["stack_path"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("oof", "full"), required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)

    training = read_rows(args.data / "training.csv")
    validation = read_rows(args.validation_manifest)
    check_manifests(training, validation)

    # Torchvision downloads and caches ImageNet weights on first use.
    model = VGG16AbnormalDisc(pretrained=True).to(device)
    train_features = encode_rows(model, training, device)

    if args.mode == "oof":
        assignments = patient_folds(
            training, args.folds, args.seed
        )
        scores = np.zeros(len(training), dtype=np.float32)
        for fold in range(args.folds):
            train_indices = np.flatnonzero(assignments != fold)
            held_indices = np.flatnonzero(assignments == fold)
            fold_rows = [training[i] for i in train_indices]

            train_head(
                model,
                train_features[train_indices],
                fold_rows,
                args.epochs,
                args.seed + fold,
                device,
                f"fold={fold + 1}/{args.folds}",
            )
            scores[held_indices] = score_features(
                model, train_features[held_indices], device
            )
        result_rows = training

    else:
        train_head(
            model, train_features, training,
            args.epochs, args.seed, device, "full"
        )
        val_features = encode_rows(model, validation, device)
        scores = score_features(model, val_features, device)
        result_rows = validation

    truth = np.array([abnormal(row) for row in result_rows])
    ap = float(average_precision_score(truth, scores))

    args.output.mkdir(parents=True)
    if args.mode == "oof":
        save_predictions(
            args.output / "oof_predictions.csv",
            result_rows, scores, assignments,
        )
    else:
        state = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }
        torch.save({
            "model_kind": "abnormal_disc_vgg16_frozen",
            "model_state_dict": state,
            "epochs": args.epochs,
            "seed": args.seed,
            "pretrained_weights": "VGG16_Weights.IMAGENET1K_V1",
            "positive_class": "bulging or herniation",
            "slice_pooling": "mean of top two logits",
        }, args.output / "fixed.pt")
        save_predictions(
            args.output / "validation_predictions.csv",
            result_rows, scores,
        )

    report = {
        "mode": args.mode,
        "discs": len(result_rows),
        "abnormal_discs": int(truth.sum()),
        "disc_ap": ap,
        "note": "Reused SPIDER validation is exploratory, not an independent test.",
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()