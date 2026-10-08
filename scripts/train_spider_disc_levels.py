import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.data.spider_disc_level_dataset import (
    SpiderDiscLevelDataset,
)
from src.models.spider_disc_unet import SpiderDiscUNet


LEVEL_NAMES = [
    "Background",
    "L1-L2",
    "L2-L3",
    "L3-L4",
    "L4-L5",
    "L5-S1",
]

NUMBER_OF_CLASSES = len(LEVEL_NAMES)
IGNORE_INDEX = 255


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def multiclass_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate differentiable Dice loss for the five disc-level
    classes.

    Background class 0 is excluded because it occupies almost the
    complete MRI image.

    Pixels with value 255 are excluded because they represent upper
    discs that are not part of L1-L2 through L5-S1.
    """
    probabilities = torch.softmax(logits, dim=1)

    valid_mask = targets != IGNORE_INDEX

    safe_targets = targets.clone()
    safe_targets[~valid_mask] = 0

    one_hot_targets = F.one_hot(
        safe_targets,
        num_classes=NUMBER_OF_CLASSES,
    )

    # [B, H, W, C] -> [B, C, H, W]
    one_hot_targets = one_hot_targets.permute(
        0,
        3,
        1,
        2,
    ).float()

    valid_mask = valid_mask.unsqueeze(1).float()

    probabilities = probabilities * valid_mask
    one_hot_targets = one_hot_targets * valid_mask

    # Exclude background and retain classes 1 through 5.
    foreground_probabilities = probabilities[:, 1:]
    foreground_targets = one_hot_targets[:, 1:]

    dimensions = (0, 2, 3)

    intersection = (
        foreground_probabilities * foreground_targets
    ).sum(dim=dimensions)

    predicted_total = foreground_probabilities.sum(
        dim=dimensions
    )

    target_total = foreground_targets.sum(
        dim=dimensions
    )

    soft_dice = (
        2.0 * intersection + 1.0
    ) / (
        predicted_total + target_total + 1.0
    )

    return 1.0 - soft_dice.mean()


def segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    cross_entropy: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combine weighted cross-entropy with multiclass Dice loss.
    """
    ce_loss = cross_entropy(logits, targets)
    dice_loss = multiclass_dice_loss(logits, targets)

    total_loss = ce_loss + dice_loss

    return total_loss, ce_loss, dice_loss


def update_confusion_matrix(
    confusion: np.ndarray,
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """
    Add one batch to the global pixel-level confusion matrix.

    Matrix rows are true classes.
    Matrix columns are predicted classes.
    """
    predictions = predictions.detach().cpu().numpy().reshape(-1)
    targets = targets.detach().cpu().numpy().reshape(-1)

    valid = targets != IGNORE_INDEX

    predictions = predictions[valid]
    targets = targets[valid]

    encoded = (
        targets * NUMBER_OF_CLASSES + predictions
    )

    counts = np.bincount(
        encoded,
        minlength=NUMBER_OF_CLASSES * NUMBER_OF_CLASSES,
    )

    confusion += counts.reshape(
        NUMBER_OF_CLASSES,
        NUMBER_OF_CLASSES,
    )


def metrics_from_confusion(
    confusion: np.ndarray,
) -> dict:
    per_class = []

    dice_values = []
    iou_values = []

    for class_id, class_name in enumerate(LEVEL_NAMES):
        true_positive = int(confusion[class_id, class_id])

        false_positive = int(
            confusion[:, class_id].sum() - true_positive
        )

        false_negative = int(
            confusion[class_id, :].sum() - true_positive
        )

        support = int(confusion[class_id, :].sum())

        precision_denominator = (
            true_positive + false_positive
        )

        recall_denominator = (
            true_positive + false_negative
        )

        dice_denominator = (
            2 * true_positive
            + false_positive
            + false_negative
        )

        iou_denominator = (
            true_positive
            + false_positive
            + false_negative
        )

        precision = (
            true_positive / precision_denominator
            if precision_denominator > 0
            else 0.0
        )

        recall = (
            true_positive / recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        dice = (
            2 * true_positive / dice_denominator
            if dice_denominator > 0
            else 0.0
        )

        iou = (
            true_positive / iou_denominator
            if iou_denominator > 0
            else 0.0
        )

        per_class.append({
            "class_id": class_id,
            "class_name": class_name,
            "support": support,
            "precision": precision,
            "recall": recall,
            "dice": dice,
            "iou": iou,
        })

        # Background is not part of the macro disc-level metrics.
        if class_id != 0:
            dice_values.append(dice)
            iou_values.append(iou)

    total_pixels = int(confusion.sum())
    correct_pixels = int(np.trace(confusion))

    pixel_accuracy = (
        correct_pixels / total_pixels
        if total_pixels > 0
        else 0.0
    )

    return {
        "pixel_accuracy": pixel_accuracy,
        "macro_level_dice": float(np.mean(dice_values)),
        "macro_level_iou": float(np.mean(iou_values)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    cross_entropy: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict:
    training = optimizer is not None
    model.train(training)

    confusion = np.zeros(
        (NUMBER_OF_CLASSES, NUMBER_OF_CLASSES),
        dtype=np.int64,
    )

    total_examples = 0
    total_loss = 0.0
    total_ce_loss = 0.0
    total_dice_loss = 0.0

    for images, targets in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)

            if logits.shape[1] != NUMBER_OF_CLASSES:
                raise ValueError(
                    "Expected six output channels, "
                    f"received {logits.shape}"
                )

            loss, ce_loss, dice_loss = segmentation_loss(
                logits,
                targets,
                cross_entropy,
            )

            if training:
                loss.backward()

                # Protect training against unusually large gradients.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

        predictions = logits.detach().argmax(dim=1)

        update_confusion_matrix(
            confusion,
            predictions,
            targets,
        )

        batch_size = images.shape[0]
        total_examples += batch_size

        total_loss += float(loss.item()) * batch_size
        total_ce_loss += float(ce_loss.item()) * batch_size
        total_dice_loss += (
            float(dice_loss.item()) * batch_size
        )

    if total_examples == 0:
        raise RuntimeError("Data loader produced no examples")

    result = metrics_from_confusion(confusion)

    result["loss"] = total_loss / total_examples
    result["cross_entropy_loss"] = (
        total_ce_loss / total_examples
    )
    result["dice_loss"] = (
        total_dice_loss / total_examples
    )

    return result


def print_metrics(
    name: str,
    metrics: dict,
) -> None:
    print(
        f"{name}: "
        f"loss={metrics['loss']:.4f} "
        f"ce={metrics['cross_entropy_loss']:.4f} "
        f"dice_loss={metrics['dice_loss']:.4f} "
        f"pixel_accuracy={metrics['pixel_accuracy']:.4f} "
        f"macro_dice={metrics['macro_level_dice']:.4f} "
        f"macro_iou={metrics['macro_level_iou']:.4f}"
    )

    for class_result in metrics["per_class"]:
        if class_result["class_id"] == 0:
            continue

        print(
            f"  {class_result['class_name']:<6} "
            f"precision={class_result['precision']:.3f} "
            f"recall={class_result['recall']:.3f} "
            f"dice={class_result['dice']:.3f} "
            f"iou={class_result['iou']:.3f} "
            f"support={class_result['support']}"
        )

    print(
        "  Confusion matrix "
        "[true class row, predicted class column]:"
    )

    for row in metrics["confusion_matrix"]:
        print("   ", row)


def extract_state_dict(
    checkpoint: object,
) -> dict[str, torch.Tensor]:
    """
    Support either:

    1. A plain model state dictionary.
    2. A structured checkpoint containing model_state_dict.
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "The binary checkpoint must be a dictionary"
        )

    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint

    if not isinstance(state, dict):
        raise TypeError(
            "Could not find a valid model state dictionary"
        )

    return state


def load_binary_pretraining(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    """
    Load the encoder and decoder from the previously trained binary
    disc segmentation model.

    The old output layer has one channel, while the new output layer
    has six channels. Therefore, output_layer weights are deliberately
    excluded.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Binary localization checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    source_state = extract_state_dict(checkpoint)

    transferred_state = {
        key: value
        for key, value in source_state.items()
        if not key.startswith("output_layer.")
    }

    incompatible = model.load_state_dict(
        transferred_state,
        strict=False,
    )

    expected_missing = {
        "output_layer.weight",
        "output_layer.bias",
    }

    actual_missing = set(incompatible.missing_keys)

    if actual_missing != expected_missing:
        raise ValueError(
            "Unexpected missing parameters while loading binary "
            f"pretraining: {sorted(actual_missing)}"
        )

    if incompatible.unexpected_keys:
        raise ValueError(
            "Unexpected binary checkpoint parameters: "
            f"{incompatible.unexpected_keys}"
        )

    print("Loaded binary localization pretraining")
    print("Source checkpoint:", checkpoint_path)
    print("Transferred tensors:", len(transferred_state))
    print(
        "New randomly initialized tensors:",
        sorted(actual_missing),
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
    validation_metrics: dict,
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "validation_metrics": validation_metrics,
        "config": {
            "manifest": str(args.manifest),
            "output_dir": str(args.output_dir),
            "binary_checkpoint": str(
                args.binary_checkpoint
            ),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "patience": args.patience,
            "seed": args.seed,
            "number_of_classes": NUMBER_OF_CLASSES,
            "ignore_index": IGNORE_INDEX,
        },
        "class_names": LEVEL_NAMES,
    }

    torch.save(checkpoint, path)

    # Immediately verify that the checkpoint is readable.
    verification = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if verification["epoch"] != epoch:
        raise RuntimeError(
            "Saved checkpoint failed epoch verification"
        )

    if "model_state_dict" not in verification:
        raise RuntimeError(
            "Saved checkpoint has no model_state_dict"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/processed/spider_disc_localization/"
            "manifest.csv"
        ),
    )

    parser.add_argument(
        "--binary-checkpoint",
        type=Path,
        default=Path(
            "outputs/spider_disc_localization_softdice_v1/"
            "best_model.pt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/spider_disc_levels_v1"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--no-pretraining",
        action="store_true",
        help=(
            "Train from random initialization instead of loading "
            "the binary disc-localization checkpoint"
        ),
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    if args.lr <= 0:
        raise ValueError("--lr must be positive")

    set_seed(args.seed)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_dataset = SpiderDiscLevelDataset(
        args.manifest,
        subset="training",
    )

    validation_dataset = SpiderDiscLevelDataset(
        args.manifest,
        subset="validation",
    )

    train_patients = {
        row["patient_id"]
        for row in train_dataset.rows
    }

    validation_patients = {
        row["patient_id"]
        for row in validation_dataset.rows
    }

    overlap = train_patients & validation_patients

    if overlap:
        raise ValueError(
            "Training and validation share patients: "
            f"{sorted(overlap)}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
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

    model = SpiderDiscUNet(
        output_channels=NUMBER_OF_CLASSES,
    ).to(device)

    if not args.no_pretraining:
        load_binary_pretraining(
            model,
            args.binary_checkpoint,
            device,
        )
    else:
        print("Training without binary-model pretraining")

    # Background is down-weighted because it occupies nearly the
    # complete image. Foreground weights are kept close together
    # because the five lumbar levels have similar pixel counts.
    class_weights = torch.tensor(
        [
            0.05,  # Background
            1.15,  # L1-L2
            1.06,  # L2-L3
            1.00,  # L3-L4
            0.97,  # L4-L5
            0.99,  # L5-S1
        ],
        dtype=torch.float32,
        device=device,
    )

    cross_entropy = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=IGNORE_INDEX,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    best_macro_dice = -1.0
    best_macro_iou = -1.0
    epochs_without_improvement = 0

    history = []

    checkpoint_path = (
        args.output_dir / "best_model.pt"
    )

    print()
    print("Training images:", len(train_dataset))
    print(
        "Validation images:",
        len(validation_dataset),
    )
    print("Patient overlap:", len(overlap))
    print("Device:", device)
    print(
        "Trainable parameters:",
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )
    print(
        "Class weights:",
        class_weights.detach().cpu().tolist(),
    )

    for epoch in range(1, args.epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            cross_entropy=cross_entropy,
            device=device,
            optimizer=optimizer,
        )

        validation_metrics = run_epoch(
            model=model,
            loader=validation_loader,
            cross_entropy=cross_entropy,
            device=device,
            optimizer=None,
        )

        current_macro_dice = validation_metrics[
            "macro_level_dice"
        ]

        current_macro_iou = validation_metrics[
            "macro_level_iou"
        ]

        scheduler.step(current_macro_dice)

        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train": train_metrics,
            "validation": validation_metrics,
        }

        history.append(record)

        print()
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Learning rate: {learning_rate:.8f}")

        print_metrics(
            "Train",
            train_metrics,
        )

        print_metrics(
            "Validation",
            validation_metrics,
        )

        improved = (
            current_macro_dice > best_macro_dice + 1e-6
            or (
                abs(
                    current_macro_dice
                    - best_macro_dice
                ) <= 1e-6
                and current_macro_iou > best_macro_iou
            )
        )

        if improved:
            best_macro_dice = current_macro_dice
            best_macro_iou = current_macro_iou
            epochs_without_improvement = 0

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                validation_metrics=validation_metrics,
                args=args,
            )

            with (
                args.output_dir
                / "best_validation_metrics.json"
            ).open("w") as file:
                json.dump(
                    {
                        "epoch": epoch,
                        **validation_metrics,
                    },
                    file,
                    indent=2,
                )

            print(
                "Saved best checkpoint:",
                checkpoint_path,
            )

        else:
            epochs_without_improvement += 1

            print(
                "Epochs without improvement:",
                epochs_without_improvement,
            )

        with (
            args.output_dir / "history.json"
        ).open("w") as file:
            json.dump(
                history,
                file,
                indent=2,
            )

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print("Early stopping")
            break

    print()
    print(
        "Best validation macro level Dice:",
        round(best_macro_dice, 4),
    )
    print(
        "Best validation macro level IoU:",
        round(best_macro_iou, 4),
    )
    print("Best checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()