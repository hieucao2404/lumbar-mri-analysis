import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.spider_dataset import SpiderPfirrmannDataset
from src.models.spider_model import SpiderPfirrmannCNN


GRADE_NAMES = ["I", "II", "III", "IV", "V"]


def metrics_from_confusion(confusion: np.ndarray) -> dict:
    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / total) if total else 0.0

    per_grade = []
    f1_values = []

    for grade_id, name in enumerate(GRADE_NAMES):
        true_count = int(confusion[grade_id, :].sum())
        predicted_count = int(confusion[:, grade_id].sum())
        correct = int(confusion[grade_id, grade_id])

        precision = correct / predicted_count if predicted_count else 0.0
        recall = correct / true_count if true_count else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        f1_values.append(f1)
        per_grade.append({
            "grade": name,
            "support": true_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    # Grade IDs are 0–4, so the absolute difference is the grade error.
    grade_distance = np.abs(
        np.arange(5)[:, None] - np.arange(5)[None, :]
    )
    mean_absolute_grade_error = (
        float((confusion * grade_distance).sum() / total)
        if total
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "mean_absolute_grade_error": mean_absolute_grade_error,
        "per_grade": per_grade,
        "confusion_matrix": confusion.tolist(),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    training = optimizer is not None
    model.train(training)

    confusion = np.zeros((5, 5), dtype=np.int64)
    loss_sum = 0.0
    sample_count = 0

    context = nullcontext() if training else torch.inference_mode()

    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            scores = model(images)
            loss = criterion(scores, targets)

            if training:
                loss.backward()
                optimizer.step()

            predictions = scores.argmax(dim=1)

            batch_size = targets.size(0)
            loss_sum += loss.item() * batch_size
            sample_count += batch_size

            true_ids = targets.detach().cpu().numpy()
            predicted_ids = predictions.detach().cpu().numpy()

            for true_id, predicted_id in zip(true_ids, predicted_ids):
                confusion[int(true_id), int(predicted_id)] += 1

    result = metrics_from_confusion(confusion)
    result["loss"] = loss_sum / sample_count
    return result


def print_metrics(name: str, result: dict) -> None:
    print(
        f"{name}: "
        f"loss={result['loss']:.4f} "
        f"accuracy={result['accuracy']:.4f} "
        f"macro_f1={result['macro_f1']:.4f} "
        f"grade_mae={result['mean_absolute_grade_error']:.4f}"
    )

    for row in result["per_grade"]:
        print(
            f"  Grade {row['grade']}: "
            f"precision={row['precision']:.3f} "
            f"recall={row['recall']:.3f} "
            f"f1={row['f1']:.3f} "
            f"support={row['support']}"
        )

    print("  Confusion matrix [true grade, predicted grade]:")
    for row in result["confusion_matrix"]:
        print("   ", row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/spider_crops/crops.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/spider_pfirrmann_baseline"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # These two lines use the dataset class from the previous step.
    # If your constructor uses different argument names, adjust only these lines.
    train_dataset = SpiderPfirrmannDataset(
        args.manifest, "training"
    )
    val_dataset = SpiderPfirrmannDataset(
        args.manifest, "validation"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = SpiderPfirrmannCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    print("Device:", device)
    print("Training crops:", len(train_dataset))
    print("Validation crops:", len(val_dataset))
    print("Trainable parameters:", sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ))

    best_f1 = -1.0
    best_mae = float("inf")
    epochs_without_improvement = 0
    history = []
    checkpoint_path = args.output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_result = run_epoch(
            model, val_loader, criterion, device
        )

        print(f"\nEpoch {epoch}/{args.epochs}")
        print_metrics("Train", train_result)
        print_metrics("Validation", val_result)

        history.append({
            "epoch": epoch,
            "train": train_result,
            "validation": val_result,
        })

        current_f1 = val_result["macro_f1"]
        current_mae = val_result["mean_absolute_grade_error"]

        improved = (
            current_f1 > best_f1 + 1e-6
            or (
                abs(current_f1 - best_f1) <= 1e-6
                and current_mae < best_mae
            )
        )

        if improved:
            best_f1 = current_f1
            best_mae = current_mae
            epochs_without_improvement = 0

            saved_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": saved_state,
                "validation_metrics": val_result,
                "config": {
                    "manifest": str(args.manifest),
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "seed": args.seed,
                },
            }

            torch.save(checkpoint, checkpoint_path)

            # Reload immediately and confirm that the saved file produces
            # exactly the same validation predictions.
            verification_checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )

            verification_model = SpiderPfirrmannCNN().to(device)
            verification_model.load_state_dict(
                verification_checkpoint["model_state_dict"]
            )

            verification_result = run_epoch(
                verification_model,
                val_loader,
                criterion,
                device,
            )

            if (
                verification_result["confusion_matrix"]
                != val_result["confusion_matrix"]
            ):
                raise RuntimeError(
                    "Reloaded checkpoint does not reproduce the "
                    "validation confusion matrix.\n"
                    f"Before saving: {val_result['confusion_matrix']}\n"
                    f"After loading: "
                    f"{verification_result['confusion_matrix']}"
                )

            print(
                "Checkpoint reload verified:",
                f"macro_f1={verification_result['macro_f1']:.4f}",
                f"accuracy={verification_result['accuracy']:.4f}",
            )

            with (args.output_dir / "best_validation_metrics.json").open(
                "w"
            ) as file:
                json.dump(
                    {"epoch": epoch, **val_result},
                    file,
                    indent=2,
                )

            print("Saved best checkpoint:", checkpoint_path)
        else:
            epochs_without_improvement += 1
            print(
                "Epochs without improvement:",
                epochs_without_improvement,
            )

        with (args.output_dir / "history.json").open("w") as file:
            json.dump(history, file, indent=2)

        if epochs_without_improvement >= args.patience:
            print("Early stopping")
            break

    print("\nBest validation macro F1:", round(best_f1, 4))
    print("Best validation grade MAE:", round(best_mae, 4))
    print("Checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()