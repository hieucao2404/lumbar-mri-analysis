"""Fine-tune VGG16 with slice attention for three exclusive disc classes."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from scripts.train_spider_abnormal_disc_cnn import read_rows
from scripts.train_spider_abnormal_disc_vgg16 import check_manifests
from scripts.train_spider_vgg16_disc_status import (
    CLASSES,
    CLASS_ID,
    SAMPLE_SHARE,
    choose_threshold,
    patient_folds,
    sample_weights,
    summarize,
    write_predictions,
)
from scripts.train_spider_vgg16_finetune import TaskDataset, score_model
from src.models.attention_vgg16_disc import AttentionVGG16Disc


def train_model(rows, epochs, seed, device, label):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = AttentionVGG16Disc(pretrained=True, num_classes=3).to(device)
    sampler = WeightedRandomSampler(
        sample_weights(rows),
        num_samples=len(rows),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        TaskDataset(rows, "three_class"),
        batch_size=8,
        sampler=sampler,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.slice_encoder.parameters(), "lr": 3e-5},
            {"params": model.slice_head.parameters(), "lr": 3e-4},
            {"params": model.slice_attention.parameters(), "lr": 3e-4},
        ],
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()

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


def summarize_three_classes(rows, probabilities, threshold):
    result = summarize(rows, probabilities, "three_class", threshold)
    labels = np.array([CLASS_ID[row["status"]] for row in rows])
    result["class_counts"] = {
        name: int((labels == class_id).sum())
        for class_id, name in enumerate(CLASSES)
    }
    result["per_class_ap"] = {
        name: float(average_precision_score(
            labels == class_id, probabilities[:, class_id]
        )) for class_id, name in enumerate(CLASSES)
    }
    result["abnormal_ap_from_one_minus_p_normal"] = float(
        average_precision_score(labels != CLASS_ID["normal"],
                                1 - probabilities[:, CLASS_ID["normal"]])
    )
    result["argmax_accuracy"] = float(
        np.mean(probabilities.argmax(axis=1) == labels)
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
    folds = patient_folds(training, args.folds, args.seed)

    oof = np.zeros((len(training), len(CLASSES)), dtype=np.float32)
    for fold in range(args.folds):
        train_ids = np.flatnonzero(folds != fold)
        held_ids = np.flatnonzero(folds == fold)
        fold_rows = [training[i] for i in train_ids]
        held_rows = [training[i] for i in held_ids]
        model = train_model(
            fold_rows, args.epochs, args.seed + fold, device,
            f"three_class fold={fold + 1}/{args.folds}",
        )
        oof[held_ids] = score_model(model, held_rows, "three_class", device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    threshold = choose_threshold(training, oof, "three_class")
    model = train_model(training, args.epochs, args.seed, device,
                        "three_class full")
    validation_scores = score_model(model, validation, "three_class", device)

    args.output.mkdir(parents=True)
    write_predictions(args.output / "oof_predictions.csv", training, oof,
                      "three_class", folds)
    write_predictions(args.output / "validation_predictions.csv", validation,
                      validation_scores, "three_class")
    state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    torch.save({
        "model_kind": "spider_vgg16_attention_three_class",
        "model_state_dict": state,
        "classes": list(CLASSES),
        "herniation_threshold": threshold,
        "sampling_shares": SAMPLE_SHARE,
        "epochs": args.epochs,
        "seed": args.seed,
        "encoder": "ImageNet VGG16 features, all convolutional layers trained",
        "aggregation": "shared learned attention over eight slice logits",
    }, args.output / "fixed.pt")

    report = {
        "task": "three_class",
        "class_order": list(CLASSES),
        "sampling_shares": SAMPLE_SHARE,
        "threshold_from_training_oof_for_herniation_sensitivity_0_90": threshold,
        "training_oof": summarize_three_classes(training, oof, threshold),
        "reused_validation": summarize_three_classes(
            validation, validation_scores, threshold
        ),
        "note": "Reused SPIDER validation; exploratory only.",
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
