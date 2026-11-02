"""Patient-separated VGG16 slice-attention experiment: normal vs abnormal."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from scripts.train_spider_abnormal_disc_cnn import (
    AbnormalStacks,
    abnormal,
    patient_folds,
    read_rows,
    sampling_weights,
    save_predictions,
)
from scripts.train_spider_abnormal_disc_vgg16 import check_manifests
from src.models.attention_vgg16_disc import AttentionVGG16Disc


def train_model(rows, epochs, seed, device, label):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = AttentionVGG16Disc(pretrained=True).to(device)

    sampler = WeightedRandomSampler(
        sampling_weights(rows),
        num_samples=len(rows),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        AbnormalStacks(rows),
        batch_size=8,
        sampler=sampler,
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": model.slice_encoder.parameters(), "lr": 3e-5},
            {"params": model.slice_head.parameters(), "lr": 3e-4},
            {"params": model.slice_attention.parameters(), "lr": 3e-4},
        ],
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda"
    )
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

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
            total_loss += float(loss.detach())

        print(
            f"{label} epoch={epoch + 1}/{epochs} "
            f"loss={total_loss / len(loader):.4f}",
            flush=True,
        )

    return model


@torch.no_grad()
def score_model(model, rows, device):
    model.eval()
    loader = DataLoader(
        AbnormalStacks(rows), batch_size=8, shuffle=False
    )
    scores = []

    for images, _ in loader:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(images.to(device))
        scores.append(logits.sigmoid().float().cpu().numpy())

    return np.concatenate(scores)


def main():
    parser = argparse.ArgumentParser()
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
    folds = patient_folds(training, args.folds, args.seed)

    oof_scores = np.zeros(len(training), dtype=np.float32)

    for fold in range(args.folds):
        train_ids = np.flatnonzero(folds != fold)
        held_ids = np.flatnonzero(folds == fold)

        fold_train = [training[i] for i in train_ids]
        fold_held = [training[i] for i in held_ids]

        model = train_model(
            fold_train,
            args.epochs,
            args.seed + fold,
            device,
            f"fold={fold + 1}/{args.folds}",
        )
        oof_scores[held_ids] = score_model(
            model, fold_held, device
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model = train_model(
        training, args.epochs, args.seed, device, "full"
    )
    validation_scores = score_model(model, validation, device)

    args.output.mkdir(parents=True)

    save_predictions(
        args.output / "oof_predictions.csv",
        training,
        oof_scores,
        folds,
    )
    save_predictions(
        args.output / "validation_predictions.csv",
        validation,
        validation_scores,
    )

    state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    torch.save(
        {
            "model_kind": "abnormal_disc_vgg16_slice_attention",
            "model_state_dict": state,
            "epochs": args.epochs,
            "seed": args.seed,
            "positive_class": "bulging or herniation",
        },
        args.output / "fixed.pt",
    )

    report = {
        "task": "normal versus bulging or herniation",
        "aggregation": "learned attention over eight slice logits",
        "training_oof_ap": float(average_precision_score(
            [abnormal(row) for row in training], oof_scores
        )),
        "reused_validation_ap": float(average_precision_score(
            [abnormal(row) for row in validation],
            validation_scores,
        )),
        "training_discs": len(training),
        "validation_discs": len(validation),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()