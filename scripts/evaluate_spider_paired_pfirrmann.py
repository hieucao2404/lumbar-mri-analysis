import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.models.spider_model import SpiderPfirrmannCNN


def confusion_matrix(
    truth: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    matrix = np.zeros((5, 5), dtype=np.int64)

    for true_grade, predicted_grade in zip(
        truth,
        predictions,
        strict=True,
    ):
        matrix[true_grade - 1, predicted_grade - 1] += 1

    return matrix


def print_metrics(
    name: str,
    truth: np.ndarray,
    predictions: np.ndarray,
    loss: float,
) -> None:
    matrix = confusion_matrix(truth, predictions)

    accuracy = float(np.mean(truth == predictions))
    grade_mae = float(
        np.mean(np.abs(truth - predictions))
    )

    f1_scores = []

    print(f"\n{name}")
    print(f"  loss: {loss:.4f}")
    print(f"  accuracy: {accuracy:.4f}")
    print(f"  grade MAE: {grade_mae:.4f}")

    for index in range(5):
        tp = int(matrix[index, index])
        support = int(matrix[index, :].sum())
        predicted_count = int(matrix[:, index].sum())

        precision = (
            tp / predicted_count
            if predicted_count
            else 0.0
        )
        recall = (
            tp / support
            if support
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        f1_scores.append(f1)

        print(
            f"  Grade {index + 1}: "
            f"precision={precision:.3f} "
            f"recall={recall:.3f} "
            f"f1={f1:.3f} "
            f"support={support}"
        )

    print(f"  macro F1: {np.mean(f1_scores):.4f}")
    print("  Confusion matrix [true row, predicted column]:")

    for row in matrix:
        print("   ", row.tolist())


@torch.no_grad()
def evaluate_column(
    model: torch.nn.Module,
    rows: list[dict],
    path_column: str,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    all_predictions = []
    all_confidences = []

    total_loss = 0.0
    total_examples = 0

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]

        images = np.stack(
            [
                np.load(row[path_column]).astype(np.float32)
                for row in batch_rows
            ]
        )

        if images.ndim != 3 or images.shape[1:] != (160, 160):
            raise ValueError(
                f"Unexpected crop shape: {images.shape}"
            )

        targets = torch.tensor(
            [
                int(row["pfirrmann_grade"]) - 1
                for row in batch_rows
            ],
            dtype=torch.long,
            device=device,
        )

        inputs = torch.from_numpy(
            images[:, None, :, :]
        ).to(device)

        scores = model(inputs)

        if scores.shape != (len(batch_rows), 5):
            raise ValueError(
                f"Unexpected model output shape: {scores.shape}"
            )

        loss = F.cross_entropy(scores, targets)

        probabilities = torch.softmax(scores, dim=1)
        confidence, predicted_ids = probabilities.max(dim=1)

        # Model IDs 0..4 correspond to clinical grades 1..5.
        all_predictions.extend(
            (predicted_ids.cpu().numpy() + 1).tolist()
        )
        all_confidences.extend(
            confidence.cpu().numpy().tolist()
        )

        count = len(batch_rows)
        total_loss += float(loss.item()) * count
        total_examples += count

    return (
        np.asarray(all_predictions, dtype=np.int64),
        np.asarray(all_confidences, dtype=np.float32),
        total_loss / total_examples,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/processed/spider_paired_disc_crops/"
            "paired_crops.csv"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/spider_pfirrmann_fov70/best_model.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/spider_pfirrmann_fov70/"
            "paired_validation_predictions.csv"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    with args.manifest.open(newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Paired-crop manifest is empty")

    if any(row["subset"] != "validation" for row in rows):
        raise ValueError(
            "This evaluation expects validation crops only"
        )

    truth = np.asarray(
        [int(row["pfirrmann_grade"]) for row in rows],
        dtype=np.int64,
    )

    if np.any((truth < 1) | (truth > 5)):
        raise ValueError("Pfirrmann grades must be 1 through 5")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load only a checkpoint you created and trust.
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    model = SpiderPfirrmannCNN().to(device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict)
    model.eval()

    expert_predictions, expert_confidence, expert_loss = (
        evaluate_column(
            model,
            rows,
            "expert_crop_path",
            device,
            args.batch_size,
        )
    )

    detected_predictions, detected_confidence, detected_loss = (
        evaluate_column(
            model,
            rows,
            "predicted_crop_path",
            device,
            args.batch_size,
        )
    )

    print("Checkpoint:", args.checkpoint)
    print("Device:", device)
    print("Paired discs:", len(rows))
    print("Patients:", len({row["patient_id"] for row in rows}))

    print_metrics(
        "Expert-mask crops",
        truth,
        expert_predictions,
        expert_loss,
    )
    print_metrics(
        "Detected-disc crops",
        truth,
        detected_predictions,
        detected_loss,
    )

    changed = expert_predictions != detected_predictions

    became_correct = (
        (expert_predictions != truth)
        & (detected_predictions == truth)
    )

    became_wrong = (
        (expert_predictions == truth)
        & (detected_predictions != truth)
    )

    print("\nPaired comparison")
    print("  Predictions changed:", int(changed.sum()))
    print("  Became correct:", int(became_correct.sum()))
    print("  Became wrong:", int(became_wrong.sum()))

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_rows = []

    for index, row in enumerate(rows):
        output_rows.append({
            **row,
            "expert_predicted_grade": int(
                expert_predictions[index]
            ),
            "expert_confidence": float(
                expert_confidence[index]
            ),
            "detected_predicted_grade": int(
                detected_predictions[index]
            ),
            "detected_confidence": float(
                detected_confidence[index]
            ),
        })

    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(output_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print("Saved per-disc predictions:", args.output)


if __name__ == "__main__":
    main()