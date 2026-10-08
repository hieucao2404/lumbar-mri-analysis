import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.spider_disc_localization_dataset import (
    SpiderDiscLocalizationDataset,
)
from src.models.spider_disc_unet import SpiderDiscUNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    dimensions = (1, 2, 3)

    intersection = (probabilities * targets).sum(dim=dimensions)
    total = (
        probabilities.sum(dim=dimensions)
        + targets.sum(dim=dimensions)
    )

    dice = (2 * intersection + 1.0) / (total + 1.0)
    return 1.0 - dice.mean()


def segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce: nn.Module,
) -> torch.Tensor:
    return bce(logits, targets) + dice_loss(logits, targets)


@torch.no_grad()
def batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    predictions = torch.sigmoid(logits) >= 0.5
    truth = targets >= 0.5
    dimensions = (1, 2, 3)

    true_positive = (predictions & truth).sum(
        dim=dimensions
    ).float()

    predicted_positive = predictions.sum(
        dim=dimensions
    ).float()

    actual_positive = truth.sum(
        dim=dimensions
    ).float()

    dice = (
        (2 * true_positive + 1.0)
        / (predicted_positive + actual_positive + 1.0)
    )

    precision = (
        true_positive / predicted_positive.clamp_min(1)
    )

    recall = (
        true_positive / actual_positive.clamp_min(1)
    )

    probabilities = torch.sigmoid(logits)
    soft_intersection = (probabilities * targets).sum(dim=dimensions)
    soft_total = (
        probabilities.sum(dim=dimensions)
        + targets.sum(dim=dimensions)
    )
    soft_dice = (
        (2 * soft_intersection + 1.0)
        / (soft_total + 1.0)
    )

    return {
        "dice": dice.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "predicted_fraction": predictions.float().mean().item(),
        "target_fraction": truth.float().mean().item(),
        "soft_dice": soft_dice.mean().item(),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    bce: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    totals = {
        "loss": 0.0,
        "dice": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "predicted_fraction": 0.0,
        "target_fraction": 0.0,
        "soft_dice": 0.0,
    }

    total_examples = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = segmentation_loss(logits, targets, bce)

            if training:
                loss.backward()
                optimizer.step()

        metrics = batch_metrics(logits.detach(), targets)

        batch_size = images.shape[0]
        total_examples += batch_size

        totals["loss"] += loss.item() * batch_size

        for key, value in metrics.items():
            totals[key] += value * batch_size

    return {
        key: value / total_examples
        for key, value in totals.items()
    }


def overfit_one_batch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    bce: nn.Module,
    device: torch.device,
    steps: int = 200,
) -> None:
    # Load once. Every update below uses these exact same images and masks.
    images, targets = next(iter(loader))
    images = images.to(device)
    targets = targets.to(device)

    positive_pixels = (
        targets.flatten(start_dim=1).sum(dim=1).tolist()
    )

    print("\nOne-batch diagnostic")
    print("Batch shape:", tuple(images.shape))
    print("Positive pixels per image:", positive_pixels)

    if any(count == 0 for count in positive_pixels):
        print(
            "Warning: at least one image has an empty disc mask. "
            "Inspect the exported data."
        )

    for step in range(steps + 1):
        model.train()

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = segmentation_loss(logits, targets, bce)
        loss.backward()
        optimizer.step()

        if step % 25 == 0 or step == steps:
            # Evaluate the same fixed batch after the update.
            model.eval()

            with torch.no_grad():
                updated_logits = model(images)
                probabilities = torch.sigmoid(updated_logits)
                metrics = batch_metrics(updated_logits, targets)
                updated_loss = segmentation_loss(
                    updated_logits,
                    targets,
                    bce,
                )

            print(
                f"step={step:3d} "
                f"loss={updated_loss.item():.4f} "
                f"dice={metrics['dice']:.4f} "
                f"precision={metrics['precision']:.4f} "
                f"recall={metrics['recall']:.4f} "
                f"pred_fraction="
                f"{metrics['predicted_fraction']:.4f} "
                f"target_fraction="
                f"{metrics['target_fraction']:.4f} "
                f"prob_max={probabilities.max().item():.4f}"
            )

    print("\nDiagnostic complete. No checkpoint was saved.")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/processed/spider_disc_localization/manifest.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/spider_disc_localization_baseline"
        ),
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--overfit-one-batch",
        action="store_true",
        help=(
            "Train repeatedly on one fixed batch to diagnose "
            "whether the model can learn its masks."
        ),
    )

    args = parser.parse_args()
    set_seed(args.seed)

    train_dataset = SpiderDiscLocalizationDataset(
        args.manifest,
        subset="training",
    )

    validation_dataset = SpiderDiscLocalizationDataset(
        args.manifest,
        subset="validation",
    )

    train_patients = {
        row["patient_id"] for row in train_dataset.rows
    }

    validation_patients = {
        row["patient_id"] for row in validation_dataset.rows
    }

    if train_patients & validation_patients:
        raise ValueError(
            "Training and validation sets share patients"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = SpiderDiscUNet().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([20.0], device=device)
    )

    print("Training images:", len(train_dataset))
    print("Validation images:", len(validation_dataset))
    print("Patient overlap:", 0)
    print("Device:", device)

    if args.overfit_one_batch:
        overfit_one_batch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            bce=bce,
            device=device,
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_soft_dice = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            bce,
            device,
        )

        validation_metrics = run_epoch(
            model,
            validation_loader,
            None,
            bce,
            device,
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }

        history.append(record)

        print(f"\nEpoch {epoch}/{args.epochs}")

        for name, metrics in (
            ("Train", train_metrics),
            ("Validation", validation_metrics),
        ):
            print(
                f"{name}: "
                f"loss={metrics['loss']:.4f} "
                f"dice={metrics['dice']:.4f} "
                f"precision={metrics['precision']:.4f} "
                f"recall={metrics['recall']:.4f} "
                f"pred_fraction="
                f"{metrics['predicted_fraction']:.4f} "
                f"target_fraction="
                f"{metrics['target_fraction']:.4f}"
                f"soft_dice{metrics['soft_dice']:.4f}"
            )

        if validation_metrics["dice"] > best_soft_dice:
            best_soft_dice = validation_metrics["dice"]
            epochs_without_improvement = 0

            checkpoint_path = (
                args.output_dir / "best_model.pt"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_metrics": validation_metrics,
                    "config": vars(args),
                },
                checkpoint_path,
            )

            print("Saved best checkpoint:", checkpoint_path)

        else:
            epochs_without_improvement += 1

            print(
                "Epochs without improvement:",
                epochs_without_improvement,
            )

        history_path = args.output_dir / "history.json"

        with history_path.open("w") as file:
            json.dump(history, file, indent=2)

        if epochs_without_improvement >= args.patience:
            print("Early stopping")
            break

    print(f"\nBest validation Dice: {best_soft_dice:.4f}")
    print(
        "Best checkpoint:",
        args.output_dir / "best_model.pt",
    )


if __name__ == "__main__":
    main()