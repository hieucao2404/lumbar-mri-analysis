"""Fine-tune all VGG16 convolutional layers for disc status."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from scripts.train_spider_abnormal_disc_cnn import AbnormalStacks, read_rows
from scripts.train_spider_abnormal_disc_vgg16 import check_manifests
from scripts.train_spider_vgg16_disc_status import (
    CLASS_ID,
    choose_threshold,
    patient_folds,
    sample_weights,
    summarize,
    write_predictions,
)
from src.models.trainable_vgg16_disc import TrainableVGG16Disc


class TaskDataset(AbnormalStacks):
    def __init__(self, rows, task):
        super().__init__(rows)
        self.task = task

    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        row = self.rows[index]
        if self.task == "binary_herniation":
            target = torch.tensor(
                float(row["status"] == "herniation"),
                dtype=torch.float32,
            )
        else:
            target = torch.tensor(
                CLASS_ID[row["status"]],
                dtype=torch.long,
            )
        return image, target


def train_model(rows, task, epochs, seed, device, label):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    outputs = 1 if task == "binary_herniation" else 3
    model = TrainableVGG16Disc(outputs, pretrained=True).to(device)
    sampler = WeightedRandomSampler(
        sample_weights(rows),
        num_samples=len(rows),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        TaskDataset(rows, task),
        batch_size=8,
        sampler=sampler,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": model.slice_encoder.parameters(), "lr": 3e-5},
            {"params": model.slice_head.parameters(), "lr": 3e-4},
        ],
        weight_decay=1e-4,
    )
    loss_fn = (
        nn.BCEWithLogitsLoss()
        if outputs == 1
        else nn.CrossEntropyLoss()
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda"
    )

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = loss_fn(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())

        print(
            f"{label} epoch={epoch + 1}/{epochs} "
            f"loss={running / len(loader):.4f}",
            flush=True,
        )

    return model


@torch.no_grad()
def score_model(model, rows, task, device):
    model.eval()
    loader = DataLoader(
        TaskDataset(rows, task),
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )
    parts = []
    for images, _ in loader:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(images.to(device))
        probabilities = (
            logits.sigmoid()
            if task == "binary_herniation"
            else logits.softmax(dim=1)
        )
        parts.append(probabilities.float().cpu().numpy())
    return np.concatenate(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=("binary_herniation", "three_class"),
        required=True,
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    if args.epochs < 1 or args.folds < 2:
        raise ValueError("Need positive epochs and at least two folds")

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
    assignments = patient_folds(training, args.folds, args.seed)

    shape = (
        (len(training),)
        if args.task == "binary_herniation"
        else (len(training), 3)
    )
    oof = np.zeros(shape, dtype=np.float32)

    for fold in range(args.folds):
        train_ids = np.flatnonzero(assignments != fold)
        held_ids = np.flatnonzero(assignments == fold)
        fold_rows = [training[i] for i in train_ids]
        held_rows = [training[i] for i in held_ids]

        model = train_model(
            fold_rows, args.task, args.epochs,
            args.seed + fold, device,
            f"{args.task} fold={fold + 1}/{args.folds}",
        )
        oof[held_ids] = score_model(
            model, held_rows, args.task, device
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    threshold = choose_threshold(training, oof, args.task)
    model = train_model(
        training, args.task, args.epochs,
        args.seed, device, f"{args.task} full",
    )
    validation_scores = score_model(
        model, validation, args.task, device
    )

    args.output.mkdir(parents=True)
    write_predictions(
        args.output / "oof_predictions.csv",
        training, oof, args.task, assignments,
    )
    write_predictions(
        args.output / "validation_predictions.csv",
        validation, validation_scores, args.task,
    )
    state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    torch.save(
        {
            "model_kind": f"spider_vgg16_finetuned_{args.task}",
            "model_state_dict": state,
            "herniation_threshold": threshold,
            "epochs": args.epochs,
            "seed": args.seed,
            "encoder": "ImageNet VGG16 features, all convolutional layers trained",
        },
        args.output / "fixed.pt",
    )

    report = {
        "task": args.task,
        "threshold_from_training_oof": threshold,
        "training_oof": summarize(
            training, oof, args.task, threshold
        ),
        "reused_validation": summarize(
            validation, validation_scores, args.task, threshold
        ),
        "note": "Reused SPIDER validation; exploratory only.",
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()