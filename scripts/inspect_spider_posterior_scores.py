import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.spider_posterior_dataset import (
    SpiderPosteriorDataset,
)
from src.models.spider_posterior_model import (
    SpiderPosteriorCNN,
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
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/spider_posterior_baseline/best_model.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/spider_posterior_baseline/"
            "validation_probabilities.csv"
        ),
    )
    args = parser.parse_args()

    dataset = SpiderPosteriorDataset(
        args.manifest,
        "validation",
    )
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )

    model = SpiderPosteriorCNN().to(device)
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    probabilities = []

    with torch.inference_mode():
        for inputs, _ in loader:
            scores = model(inputs.to(device))
            probabilities.append(
                scores.sigmoid().cpu().numpy()
            )

    probabilities = np.concatenate(
        probabilities,
        axis=0,
    )

    if probabilities.shape != (len(dataset.rows), 2):
        raise ValueError(
            f"Unexpected probability shape: "
            f"{probabilities.shape}"
        )

    output_rows = []

    for row, scores in zip(
        dataset.rows,
        probabilities,
    ):
        output_rows.append({
            "patient_id": row["patient_id"],
            "series_name": row["series_name"],
            "ivd_label": row["ivd_label"],
            "selected_slice": row["selected_slice"],
            "same_slice_as_original_crop":
                row["same_slice_as_original_crop"],
            "bulging_true": row["disc_bulging"],
            "bulging_probability": float(scores[0]),
            "herniation_true": row["disc_herniation"],
            "herniation_probability": float(scores[1]),
        })

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(output_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    truth = np.array(
        [
            int(row["herniation_true"])
            for row in output_rows
        ],
        dtype=np.int64,
    )
    scores = probabilities[:, 1]

    positive_scores = scores[truth == 1]
    negative_scores = scores[truth == 0]

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Validation discs:", len(truth))
    print("Herniation positives:", int(truth.sum()))
    print(
        "Positive score range:",
        (
            round(float(positive_scores.min()), 4),
            round(float(positive_scores.max()), 4),
        ),
    )
    print(
        "Negative score range:",
        (
            round(float(negative_scores.min()), 4),
            round(float(negative_scores.max()), 4),
        ),
    )

    print("\nDescriptive threshold sweep")
    print("threshold  TP  FP  FN  TN  precision  recall  F1")

    for threshold in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        predicted = scores >= threshold

        tp = int(((truth == 1) & predicted).sum())
        fp = int(((truth == 0) & predicted).sum())
        fn = int(((truth == 1) & ~predicted).sum())
        tn = int(((truth == 0) & ~predicted).sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        print(
            f"{threshold:9.2f}"
            f" {tp:3d} {fp:3d} {fn:3d} {tn:3d}"
            f" {precision:10.3f}"
            f" {recall:7.3f}"
            f" {f1:5.3f}"
        )

    print("\nTop 20 herniation scores")

    top_indices = np.argsort(-scores)[:20]

    for rank, index in enumerate(
        top_indices,
        start=1,
    ):
        row = output_rows[int(index)]

        print(
            f"{rank:2d}. "
            f"{row['series_name']} "
            f"IVD={row['ivd_label']} "
            f"true={row['herniation_true']} "
            f"score={scores[index]:.4f} "
            f"same_slice="
            f"{row['same_slice_as_original_crop']}"
        )

    print("\nSaved:", args.output)


if __name__ == "__main__":
    main()