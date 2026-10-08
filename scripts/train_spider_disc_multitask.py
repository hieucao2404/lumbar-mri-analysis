import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.data.spider_disc_multitask_dataset import (
    SpiderDiscMultitaskDataset,
)
from src.models.spider_disc_multitask_unet import (
    SpiderDiscMultitaskUNet,
    load_binary_pretraining,
)


LEVEL_NAMES = [
    "L1-L2",
    "L2-L3",
    "L3-L4",
    "L4-L5",
    "L5-S1",
    "Upper",
]

NUMBER_OF_LEVEL_CLASSES = len(LEVEL_NAMES)
LEVEL_IGNORE_INDEX = -100


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_level_targets(
    level_masks: torch.Tensor,
) -> torch.Tensor:
    """
    Stored masks:
        0   = background
        1   = L1-L2
        2   = L2-L3
        3   = L3-L4
        4   = L4-L5
        5   = L5-S1
        255 = upper/non-lumbar disc

    Training targets:
        -100 = ignored background
        0-4  = lumbar levels
        5    = upper/non-lumbar disc
    """
    targets = torch.full_like(
        level_masks,
        fill_value=LEVEL_IGNORE_INDEX,
    )

    lumbar_pixels = (
        (level_masks >= 1)
        & (level_masks <= 5)
    )

    targets[lumbar_pixels] = (
        level_masks[lumbar_pixels] - 1
    )

    targets[level_masks == 255] = 5

    return targets


def binary_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)

    dimensions = (1, 2, 3)

    intersection = (
        probabilities * targets
    ).sum(dim=dimensions)

    total = (
        probabilities.sum(dim=dimensions)
        + targets.sum(dim=dimensions)
    )

    dice = (
        2.0 * intersection + 1.0
    ) / (
        total + 1.0
    )

    return 1.0 - dice.mean()


def calculate_loss(
    binary_logits: torch.Tensor,
    level_logits: torch.Tensor,
    binary_targets: torch.Tensor,
    level_targets: torch.Tensor,
    binary_bce: nn.Module,
) -> dict[str, torch.Tensor]:
    binary_bce_value = binary_bce(
        binary_logits,
        binary_targets,
    )

    binary_dice_value = binary_dice_loss(
        binary_logits,
        binary_targets,
    )

    binary_loss = (
        binary_bce_value
        + binary_dice_value
    )

    level_loss = F.cross_entropy(
        level_logits,
        level_targets,
        ignore_index=LEVEL_IGNORE_INDEX,
    )

    total_loss = binary_loss + level_loss

    return {
        "total": total_loss,
        "binary": binary_loss,
        "binary_bce": binary_bce_value,
        "binary_dice_loss": binary_dice_value,
        "level": level_loss,
    }


def update_level_confusion(
    confusion: np.ndarray,
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    predictions = (
        predictions.detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    targets = (
        targets.detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    valid = targets != LEVEL_IGNORE_INDEX

    predictions = predictions[valid]
    targets = targets[valid]

    encoded = (
        targets * NUMBER_OF_LEVEL_CLASSES
        + predictions
    )

    counts = np.bincount(
        encoded,
        minlength=(
            NUMBER_OF_LEVEL_CLASSES
            * NUMBER_OF_LEVEL_CLASSES
        ),
    )

    confusion += counts.reshape(
        NUMBER_OF_LEVEL_CLASSES,
        NUMBER_OF_LEVEL_CLASSES,
    )


def level_metrics_from_confusion(
    confusion: np.ndarray,
) -> dict:
    per_class = []
    f1_values = []

    for class_id, name in enumerate(LEVEL_NAMES):
        true_positive = int(
            confusion[class_id, class_id]
        )

        false_positive = int(
            confusion[:, class_id].sum()
            - true_positive
        )

        false_negative = int(
            confusion[class_id, :].sum()
            - true_positive
        )

        support = int(
            confusion[class_id, :].sum()
        )

        precision_denominator = (
            true_positive + false_positive
        )

        recall_denominator = (
            true_positive + false_negative
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

        f1 = (
            2.0 * precision * recall
            / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        per_class.append({
            "class_id": class_id,
            "class_name": name,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

        f1_values.append(f1)

    total = int(confusion.sum())
    correct = int(np.trace(confusion))

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def binary_metrics_from_counts(
    true_positive: int,
    false_positive: int,
    false_negative: int,
    true_negative: int,
) -> dict[str, float]:
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

    total = (
        true_positive
        + false_positive
        + false_negative
        + true_negative
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

    accuracy = (
        (true_positive + true_negative) / total
        if total > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
        "pixel_accuracy": accuracy,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    binary_bce: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict:
    training = optimizer is not None
    model.train(training)

    level_confusion = np.zeros(
        (
            NUMBER_OF_LEVEL_CLASSES,
            NUMBER_OF_LEVEL_CLASSES,
        ),
        dtype=np.int64,
    )

    binary_true_positive = 0
    binary_false_positive = 0
    binary_false_negative = 0
    binary_true_negative = 0

    loss_totals = {
        "total": 0.0,
        "binary": 0.0,
        "binary_bce": 0.0,
        "binary_dice_loss": 0.0,
        "level": 0.0,
    }

    total_examples = 0

    for (
        images,
        binary_targets,
        stored_level_masks,
    ) in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        binary_targets = binary_targets.to(
            device,
            non_blocking=True,
        )

        stored_level_masks = stored_level_masks.to(
            device,
            non_blocking=True,
        )

        level_targets = create_level_targets(
            stored_level_masks
        )

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            binary_logits, level_logits = model(
                images
            )

            losses = calculate_loss(
                binary_logits=binary_logits,
                level_logits=level_logits,
                binary_targets=binary_targets,
                level_targets=level_targets,
                binary_bce=binary_bce,
            )

            if training:
                losses["total"].backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

        binary_predictions = (
            torch.sigmoid(binary_logits.detach())
            >= 0.5
        )

        binary_truth = binary_targets >= 0.5

        binary_true_positive += int(
            (binary_predictions & binary_truth)
            .sum()
            .item()
        )

        binary_false_positive += int(
            (binary_predictions & ~binary_truth)
            .sum()
            .item()
        )

        binary_false_negative += int(
            (~binary_predictions & binary_truth)
            .sum()
            .item()
        )

        binary_true_negative += int(
            (~binary_predictions & ~binary_truth)
            .sum()
            .item()
        )

        level_predictions = (
            level_logits.detach().argmax(dim=1)
        )

        update_level_confusion(
            level_confusion,
            level_predictions,
            level_targets,
        )

        batch_size = images.shape[0]
        total_examples += batch_size

        for name, loss in losses.items():
            loss_totals[name] += (
                float(loss.item()) * batch_size
            )

    if total_examples == 0:
        raise RuntimeError(
            "Data loader produced no examples"
        )

    binary_metrics = binary_metrics_from_counts(
        true_positive=binary_true_positive,
        false_positive=binary_false_positive,
        false_negative=binary_false_negative,
        true_negative=binary_true_negative,
    )

    level_metrics = level_metrics_from_confusion(
        level_confusion
    )

    result = {
        "loss": (
            loss_totals["total"]
            / total_examples
        ),
        "binary_loss": (
            loss_totals["binary"]
            / total_examples
        ),
        "binary_bce_loss": (
            loss_totals["binary_bce"]
            / total_examples
        ),
        "binary_dice_loss": (
            loss_totals["binary_dice_loss"]
            / total_examples
        ),
        "level_loss": (
            loss_totals["level"]
            / total_examples
        ),
        "binary": binary_metrics,
        "level": level_metrics,
    }

    return result


def print_metrics(
    name: str,
    result: dict,
) -> None:
    binary = result["binary"]
    level = result["level"]

    print(
        f"{name}: "
        f"loss={result['loss']:.4f} "
        f"binary_loss={result['binary_loss']:.4f} "
        f"level_loss={result['level_loss']:.4f}"
    )

    print(
        "  Binary: "
        f"dice={binary['dice']:.4f} "
        f"iou={binary['iou']:.4f} "
        f"precision={binary['precision']:.4f} "
        f"recall={binary['recall']:.4f}"
    )

    print(
        "  Level: "
        f"accuracy={level['accuracy']:.4f} "
        f"macro_f1={level['macro_f1']:.4f}"
    )

    for class_result in level["per_class"]:
        print(
            f"    {class_result['class_name']:<6} "
            f"precision={class_result['precision']:.3f} "
            f"recall={class_result['recall']:.3f} "
            f"f1={class_result['f1']:.3f} "
            f"support={class_result['support']}"
        )

    print(
        "  Level confusion matrix "
        "[true row, predicted column]:"
    )

    for row in level["confusion_matrix"]:
        print("   ", row)


def set_warmup_mode(
    model: SpiderDiscMultitaskUNet,
) -> None:
    """
    Freeze the pretrained localizer and train only the new level head.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in (
        model.level_output_layer.parameters()
    ):
        parameter.requires_grad = True


def set_finetuning_mode(
    model: SpiderDiscMultitaskUNet,
) -> None:
    """
    Unfreeze the complete model for low-learning-rate fine-tuning.
    """
    for parameter in model.parameters():
        parameter.requires_grad = True


def create_optimizer(
    model: SpiderDiscMultitaskUNet,
    shared_learning_rate: float,
    level_learning_rate: float,
) -> torch.optim.Optimizer:
    shared_parameters = []
    binary_head_parameters = []

    for name, parameter in model.named_parameters():
        if name.startswith("level_output_layer."):
            continue

        if name.startswith("binary_output_layer."):
            binary_head_parameters.append(parameter)
        else:
            shared_parameters.append(parameter)

    return torch.optim.AdamW(
        [
            {
                "params": shared_parameters,
                "lr": shared_learning_rate,
            },
            {
                "params": binary_head_parameters,
                "lr": shared_learning_rate,
            },
            {
                "params": (
                    model.level_output_layer.parameters()
                ),
                "lr": level_learning_rate,
            },
        ],
        weight_decay=1e-4,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_result: dict,
    selection_score: float,
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_metrics": validation_result,
        "selection_score": selection_score,
        "level_names": LEVEL_NAMES,
        "config": {
            "manifest": str(args.manifest),
            "binary_checkpoint": str(
                args.binary_checkpoint
            ),
            "epochs": args.epochs,
            "warmup_epochs": args.warmup_epochs,
            "batch_size": args.batch_size,
            "shared_lr": args.shared_lr,
            "level_lr": args.level_lr,
            "patience": args.patience,
            "seed": args.seed,
        },
    }

    torch.save(checkpoint, path)

    verification = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if verification["epoch"] != epoch:
        raise RuntimeError(
            "Checkpoint epoch verification failed"
        )

    if "model_state_dict" not in verification:
        raise RuntimeError(
            "Checkpoint has no model_state_dict"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/processed/"
            "spider_disc_localization/"
            "manifest.csv"
        ),
    )

    parser.add_argument(
        "--binary-checkpoint",
        type=Path,
        default=Path(
            "outputs/"
            "spider_disc_localization_softdice_v1/"
            "best_model.pt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/spider_disc_multitask_v1"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--shared-lr",
        type=float,
        default=5e-5,
    )

    parser.add_argument(
        "--level-lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.warmup_epochs >= args.epochs:
        raise ValueError(
            "--warmup-epochs must be smaller "
            "than --epochs"
        )

    set_seed(args.seed)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_dataset = SpiderDiscMultitaskDataset(
        args.manifest,
        subset="training",
    )

    validation_dataset = (
        SpiderDiscMultitaskDataset(
            args.manifest,
            subset="validation",
        )
    )

    train_patients = {
        row["patient_id"]
        for row in train_dataset.rows
    }

    validation_patients = {
        row["patient_id"]
        for row in validation_dataset.rows
    }

    overlap = (
        train_patients & validation_patients
    )

    if overlap:
        raise ValueError(
            "Patient overlap detected: "
            f"{sorted(overlap)}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
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

    model = SpiderDiscMultitaskUNet().to(
        device
    )

    load_binary_pretraining(
        model,
        args.binary_checkpoint,
        device,
    )

    optimizer = create_optimizer(
        model=model,
        shared_learning_rate=args.shared_lr,
        level_learning_rate=args.level_lr,
    )

    binary_bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [20.0],
            device=device,
        )
    )

    set_warmup_mode(model)

    best_score = -1.0
    best_level_f1 = -1.0
    best_binary_dice = -1.0

    # epochs_without_improvement = 0
    history = []

    checkpoint_path = (
        args.output_dir / "best_model.pt"
    )

    print("Training images:", len(train_dataset))
    print(
        "Validation images:",
        len(validation_dataset),
    )
    print("Patient overlap:", len(overlap))
    print("Device:", device)
    print("Warm-up epochs:", args.warmup_epochs)
    print(
        "Shared learning rate:",
        args.shared_lr,
    )
    print(
        "Level-head learning rate:",
        args.level_lr,
    )

    for epoch in range(1, args.epochs + 1):
        if epoch == args.warmup_epochs + 1:
            set_finetuning_mode(model)
            epochs_without_improvement = 0

            print()
            print(
                "Warm-up complete: "
                "unfreezing the complete model"
            )

        stage = (
            "warmup"
            if epoch <= args.warmup_epochs
            else "finetuning"
        )

        train_result = run_epoch(
            model=model,
            loader=train_loader,
            binary_bce=binary_bce,
            device=device,
            optimizer=optimizer,
        )

        validation_result = run_epoch(
            model=model,
            loader=validation_loader,
            binary_bce=binary_bce,
            device=device,
            optimizer=None,
        )

        validation_level_f1 = (
            validation_result[
                "level"
            ]["macro_f1"]
        )

        validation_binary_dice = (
            validation_result[
                "binary"
            ]["dice"]
        )

        # Level recognition is the primary objective.
        # Binary Dice ensures localization is not destroyed.
        selection_score = (
            0.70 * validation_level_f1
            + 0.30 * validation_binary_dice
        )

        record = {
            "epoch": epoch,
            "stage": stage,
            "selection_score": selection_score,
            "train": train_result,
            "validation": validation_result,
        }

        history.append(record)

        print()
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"stage={stage}"
        )

        print_metrics(
            "Train",
            train_result,
        )

        print_metrics(
            "Validation",
            validation_result,
        )

        print(
            "Selection score:",
            round(selection_score, 4),
        )

        if selection_score > best_score + 1e-6:
            best_score = selection_score
            best_level_f1 = validation_level_f1
            best_binary_dice = (
                validation_binary_dice
            )

            epochs_without_improvement = 0

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_result=(
                    validation_result
                ),
                selection_score=selection_score,
                args=args,
            )

            with (
                args.output_dir
                / "best_validation_metrics.json"
            ).open("w") as file:
                json.dump(
                    {
                        "epoch": epoch,
                        "selection_score": (
                            selection_score
                        ),
                        **validation_result,
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

        # Do not stop during the level-head warm-up.
        if (
            epoch > args.warmup_epochs
            and epochs_without_improvement
            >= args.patience
        ):
            print("Early stopping")
            break

    print()
    print(
        "Best selection score:",
        round(best_score, 4),
    )

    print(
        "Best validation level macro F1:",
        round(best_level_f1, 4),
    )

    print(
        "Binary Dice at best checkpoint:",
        round(best_binary_dice, 4),
    )

    print("Best checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()