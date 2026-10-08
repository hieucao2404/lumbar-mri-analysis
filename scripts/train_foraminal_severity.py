import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import math
from collections import Counter

import torch

from src.data.foramina_dataset import (
    ForaminaDataset,
    SEVERITY_NAMES,
    load_foramina_manifest,
)
from src.models.foraminal_cnn import ForaminalSeverityCNN


NUMBER_OF_CLASSES = 4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_confusion_matrix(targets, predictions):
    matrix = torch.zeros(
        NUMBER_OF_CLASSES,
        NUMBER_OF_CLASSES,
        dtype=torch.long,
    )

    for target, prediction in zip(targets, predictions):
        matrix[target, prediction] += 1

    return matrix


def calculate_metrics(confusion_matrix):
    per_class = []
    f1_scores = []

    for class_index, class_name in enumerate(SEVERITY_NAMES):
        true_positive = confusion_matrix[
            class_index, class_index
        ].item()

        false_positive = (
            confusion_matrix[:, class_index].sum().item()
            - true_positive
        )

        false_negative = (
            confusion_matrix[class_index, :].sum().item()
            - true_positive
        )

        support = confusion_matrix[class_index, :].sum().item()

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative

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
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        f1_scores.append(f1)

        per_class.append({
            "class": class_name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        })

    correct = confusion_matrix.diag().sum().item()
    total = confusion_matrix.sum().item()

    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "per_class": per_class,
    }


def calculate_class_weights(train_rows):
    counts = Counter(row["severity"] for row in train_rows)

    missing_classes = [
        class_index for class_index in range(4)
        if counts[class_index] == 0
    ]

    if missing_classes:
        raise ValueError(
            f"Training data is missing severity classes: {missing_classes}"
        )

    largest_class_count = max(counts.values())
    total = len(train_rows)

    weights = [ math.sqrt(largest_class_count / counts[class_index])
        for class_index in range(4)]

    # for class_index in range(NUMBER_OF_CLASSES):
    #     count = counts[class_index]

    #     if count == 0:
    #         raise ValueError(
    #             f"Training split has no samples for class {class_index}"
    #         )

    #     weight = total / (NUMBER_OF_CLASSES * count)
    #     weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model,
    loader,
    loss_function,
    device,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_targets = []
    all_predictions = []

    for batch in loader:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )
        targets = batch["severity"].to(
            device,
            non_blocking=True,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_function(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)

        total_loss += loss.item() * images.size(0)

        all_targets.extend(
            targets.detach().cpu().tolist()
        )
        all_predictions.extend(
            predictions.detach().cpu().tolist()
        )

    confusion_matrix = make_confusion_matrix(
        all_targets,
        all_predictions,
    )
    metrics = calculate_metrics(confusion_matrix)

    metrics["loss"] = total_loss / len(loader.dataset)
    metrics["confusion_matrix"] = confusion_matrix.tolist()

    return metrics


def print_metrics(split_name, metrics):
    print(
        f"{split_name}: "
        f"loss={metrics['loss']:.4f}, "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )

    for result in metrics["per_class"]:
        print(
            f"  {result['class']:<8} "
            f"precision={result['precision']:.3f} "
            f"recall={result['recall']:.3f} "
            f"f1={result['f1']:.3f} "
            f"support={result['support']}"
        )

    print("  Confusion matrix [true row, predicted column]:")

    for row in metrics["confusion_matrix"]:
        print(f"    {row}")


def create_loader(
    rows,
    image_size,
    margin_ratio,
    batch_size,
    shuffle,
    workers,
    pin_memory,
):
    dataset = ForaminaDataset(
        rows,
        image_size=image_size,
        margin_ratio=margin_ratio,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/foramina.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/foraminal_severity"),
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--margin-ratio", type=float, default=0.20)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_foramina_manifest(args.manifest)

    train_rows = [
        row for row in rows if row["split"] == "train"
    ]
    validation_rows = [
        row for row in rows if row["split"] == "val"
    ]
    test_rows = [
        row for row in rows if row["split"] == "test"
    ]

    print(f"Training ROIs: {len(train_rows)}")
    print(f"Validation ROIs: {len(validation_rows)}")
    print(f"Test ROIs: {len(test_rows)}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    pin_memory = device.type == "cuda"

    train_loader = create_loader(
        train_rows,
        args.image_size,
        args.margin_ratio,
        args.batch_size,
        shuffle=True,
        workers=args.workers,
        pin_memory=pin_memory,
    )

    validation_loader = create_loader(
        validation_rows,
        args.image_size,
        args.margin_ratio,
        args.batch_size,
        shuffle=False,
        workers=args.workers,
        pin_memory=pin_memory,
    )

    test_loader = create_loader(
        test_rows,
        args.image_size,
        args.margin_ratio,
        args.batch_size,
        shuffle=False,
        workers=args.workers,
        pin_memory=pin_memory,
    )

    model = ForaminalSeverityCNN().to(device)

    class_weights = calculate_class_weights(train_rows).to(device)

    print(
        "Class weights:",
        [round(value, 4) for value in class_weights.tolist()],
    )

    training_loss_function = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # The validation and test loss remain unweighted so they represent
    # average prediction error on the real data distribution.
    evaluation_loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    checkpoint_path = args.output_dir / "best_model.pt"
    history_path = args.output_dir / "history.json"
    test_metrics_path = args.output_dir / "test_metrics.json"

    history = []
    best_macro_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            training_loss_function,
            device,
            optimizer=optimizer,
        )

        validation_metrics = run_epoch(
            model,
            validation_loader,
            evaluation_loss_function,
            device,
        )

        scheduler.step(validation_metrics["macro_f1"])

        current_learning_rate = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Learning rate: {current_learning_rate:.6g}")

        print_metrics("Train", train_metrics)
        print_metrics("Validation", validation_metrics)

        history.append({
            "epoch": epoch,
            "learning_rate": current_learning_rate,
            "train": train_metrics,
            "validation": validation_metrics,
        })

        history_path.write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

        validation_macro_f1 = validation_metrics["macro_f1"]

        if validation_macro_f1 > best_macro_f1:
            best_macro_f1 = validation_macro_f1
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": args.image_size,
                    "margin_ratio": args.margin_ratio,
                    "class_names": list(SEVERITY_NAMES),
                    "validation_macro_f1": best_macro_f1,
                    "epoch": epoch,
                },
                checkpoint_path,
            )

            print(f"Saved checkpoint: {checkpoint_path}")
        else:
            epochs_without_improvement += 1

            print(
                "No validation improvement: "
                f"{epochs_without_improvement}/"
                f"{args.patience}"
            )

        if epochs_without_improvement >= args.patience:
            print("Early stopping")
            break

    print("\nLoading the best validation checkpoint")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = run_epoch(
        model,
        test_loader,
        evaluation_loss_function,
        device,
    )

    print("\nFinal held-out test result")
    print_metrics("Test", test_metrics)

    test_metrics_path.write_text(
        json.dumps(test_metrics, indent=2),
        encoding="utf-8",
    )

    print(f"\nBest checkpoint: {checkpoint_path}")
    print(f"Training history: {history_path}")
    print(f"Test metrics: {test_metrics_path}")


if __name__ == "__main__":
    main()