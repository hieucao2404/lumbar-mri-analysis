import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.spider_posterior_dataset import (
    SpiderPosteriorDataset,
)
from src.models.spider_posterior_model import (
    SpiderPosteriorCNN,
)


LABEL_NAMES = ("Bulging", "Herniation")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def average_precision(
    truth: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """
    Average precision for one binary label.

    Calculated from the ranked predictions, independently of the
    fixed 0.5 decision threshold.
    """
    positives = int(truth.sum())

    if positives == 0:
        return float("nan")

    order = np.argsort(
        -probabilities,
        kind="stable",
    )

    ranked_truth = truth[order]
    cumulative_positives = np.cumsum(ranked_truth)
    ranks = np.arange(1, len(truth) + 1)

    precision_at_rank = (
        cumulative_positives / ranks
    )

    return float(
        precision_at_rank[ranked_truth == 1].sum()
        / positives
    )


def calculate_metrics(
    truth: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    if truth.shape != probabilities.shape:
        raise ValueError(
            f"Truth/probability shape mismatch: "
            f"{truth.shape} vs {probabilities.shape}"
        )

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    result = {}

    for label_index, label_name in enumerate(LABEL_NAMES):
        actual = truth[:, label_index].astype(np.int64)
        predicted = predictions[:, label_index]
        scores = probabilities[:, label_index]

        tp = int(((actual == 1) & (predicted == 1)).sum())
        fp = int(((actual == 0) & (predicted == 1)).sum())
        fn = int(((actual == 1) & (predicted == 0)).sum())
        tn = int(((actual == 0) & (predicted == 0)).sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        result[label_name.lower()] = {
            "support_positive": tp + fn,
            "support_negative": tn + fp,
            "predicted_positive": tp + fp,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "average_precision": average_precision(
                actual,
                scores,
            ),
        }

    return result


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict:
    training = optimizer is not None
    model.train(training)

    loss_sum = 0.0
    sample_count = 0
    all_truth = []
    all_probabilities = []

    context = (
        nullcontext()
        if training
        else torch.inference_mode()
    )

    with context:
        for inputs, targets in loader:
            inputs = inputs.to(
                device,
                non_blocking=True,
            )
            targets = targets.to(
                device,
                non_blocking=True,
            )

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(inputs)
            loss = criterion(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = inputs.shape[0]
            sample_count += batch_size
            loss_sum += float(loss.item()) * batch_size

            all_truth.append(
                targets.detach().cpu().numpy()
            )
            all_probabilities.append(
                torch.sigmoid(logits)
                .detach()
                .cpu()
                .numpy()
            )

    truth = np.concatenate(all_truth, axis=0)
    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )

    return {
        "loss": loss_sum / sample_count,
        "samples": sample_count,
        "threshold": 0.5,
        "labels": calculate_metrics(
            truth,
            probabilities,
            threshold=0.5,
        ),
    }


def print_result(
    name: str,
    result: dict,
) -> None:
    print(
        f"{name}: "
        f"loss={result['loss']:.4f} "
        f"samples={result['samples']}"
    )

    for label_name in LABEL_NAMES:
        metrics = result["labels"][label_name.lower()]

        print(
            f"  {label_name:<10} "
            f"AP={metrics['average_precision']:.4f} "
            f"P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} "
            f"TP={metrics['tp']} "
            f"FP={metrics['fp']} "
            f"FN={metrics['fn']} "
            f"TN={metrics['tn']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/processed/spider_posterior_crops/"
            "crops_with_targets.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/spider_posterior_baseline"
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_dataset = SpiderPosteriorDataset(
        args.manifest,
        "training",
    )
    validation_dataset = SpiderPosteriorDataset(
        args.manifest,
        "validation",
    )

    training_patients = {
        row["patient_id"]
        for row in training_dataset.rows
    }
    validation_patients = {
        row["patient_id"]
        for row in validation_dataset.rows
    }

    if training_patients & validation_patients:
        raise ValueError(
            "Training and validation share patients"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    training_loader = DataLoader(
        training_dataset,
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

    model = SpiderPosteriorCNN().to(device)

    # The rare herniation positives receive extra weight.
    # Keep this moderate for the first baseline.
    positive_weights = torch.tensor(
        [1.0, 4.0],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_validation_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    history = []

    checkpoint_path = (
        args.output_dir / "best_model.pt"
    )

    print("Training discs:", len(training_dataset))
    print("Validation discs:", len(validation_dataset))
    print("Patient overlap:", 0)
    print("Device:", device)
    print("Positive weights:", positive_weights.tolist())
    print("Decision threshold:", 0.5)

    for epoch in range(1, args.epochs + 1):
        training_result = run_epoch(
            model,
            training_loader,
            criterion,
            device,
            optimizer,
        )

        validation_result = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            optimizer=None,
        )

        record = {
            "epoch": epoch,
            "training": training_result,
            "validation": validation_result,
        }
        history.append(record)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print_result("Train", training_result)
        print_result("Validation", validation_result)

        current_loss = validation_result["loss"]

        if current_loss < best_validation_loss - 1e-6:
            best_validation_loss = current_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation": validation_result,
                    "positive_weights": [1.0, 4.0],
                    "threshold": 0.5,
                },
                checkpoint_path,
            )

            with (
                args.output_dir
                / "best_validation_metrics.json"
            ).open("w") as file:
                json.dump(
                    {
                        "epoch": epoch,
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
            json.dump(history, file, indent=2)

        if epochs_without_improvement >= args.patience:
            print("Early stopping")
            break

    print("\nBest epoch:", best_epoch)
    print(
        "Best validation loss:",
        round(best_validation_loss, 4),
    )
    print("Checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()