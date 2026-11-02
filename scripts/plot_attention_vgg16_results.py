"""Plot saved binary attention-VGG16 predictions without retraining."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def read_predictions(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty predictions: {path}")
    keys = [(row["patient_id"], row["series_name"], row["level"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate disc predictions: {path}")
    for row in rows:
        if int(row["abnormal"]) != int(row["status"] != "normal"):
            raise ValueError(f"Inconsistent label: {row}")
    return rows


def values(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray([int(row["abnormal"]) for row in rows], dtype=bool)
    scores = np.asarray([float(row["abnormal_score"]) for row in rows])
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be finite probabilities")
    return truth, scores


def counts(truth: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    flagged = scores >= threshold
    return {
        "tp": int(np.sum(flagged & truth)),
        "fp": int(np.sum(flagged & ~truth)),
        "fn": int(np.sum(~flagged & truth)),
        "tn": int(np.sum(~flagged & ~truth)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", type=Path,
        default=Path("outputs/spider_abnormal_vgg16_attention_guarded_v1"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    calibration = json.loads((args.experiment / "calibration.json").read_text())
    threshold = float(calibration["threshold"])
    oof = read_predictions(args.experiment / "oof_predictions.csv")
    validation = read_predictions(args.experiment / "validation_predictions.csv")
    if {row["patient_id"] for row in oof} & {
        row["patient_id"] for row in validation
    }:
        raise ValueError("Training and validation patients overlap")
    oof_y, oof_s = values(oof)
    val_y, val_s = values(validation)
    val_counts = counts(val_y, val_s, threshold)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # AP summarizes ranking over all thresholds. Cohort prevalences differ.
    ax = axes[0, 0]
    for rows, y, scores, color, label in (
        (oof, oof_y, oof_s, "#1764a5", "Training out-of-fold"),
        (validation, val_y, val_s, "#d95f02", "Reused validation"),
    ):
        precision, recall, _ = precision_recall_curve(y, scores)
        ap = average_precision_score(y, scores)
        ax.step(recall, precision, where="post", color=color, linewidth=2,
                label=f"{label}: AP {ap:.3f}, prevalence {y.mean():.3f}")
    ax.set(xlabel="Recall (sensitivity)", ylabel="Precision",
           title="Abnormal-disc precision–recall curves", xlim=(0, 1), ylim=(0, 1.03))
    ax.grid(alpha=0.2)
    ax.legend(loc="lower left", fontsize=9)

    # Rows are actual classes; columns are threshold decisions.
    ax = axes[0, 1]
    matrix = np.array([[val_counts["tn"], val_counts["fp"]],
                       [val_counts["fn"], val_counts["tp"]]])
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    for (row, column), number in np.ndenumerate(matrix):
        ax.text(column, row, str(number), ha="center", va="center",
                fontsize=18, fontweight="bold",
                color="white" if number > matrix.max() / 2 else "#10283f")
    ax.set_xticks([0, 1], ["No flag", "Flag"])
    ax.set_yticks([0, 1], ["Normal", "Abnormal"])
    ax.set(xlabel="Prediction", ylabel="Original SPIDER label",
           title=f"Validation at training threshold {threshold:.3f}")

    ax = axes[1, 0]
    bins = np.linspace(0, 1, 21)
    for status, color, label in (
        ("normal", "#1764a5", "Normal"),
        ("bulging_only", "#e69f00", "Bulging only"),
        ("herniation", "#b2182b", "Herniation"),
    ):
        group = [float(row["abnormal_score"]) for row in validation
                 if row["status"] == status]
        ax.hist(group, bins=bins, alpha=0.55, color=color,
                label=f"{label} (n={len(group)})")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold {threshold:.3f}")
    ax.set(xlabel="Predicted abnormal probability", ylabel="Disc count",
           title="Validation score distribution by original label")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    levels = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
    sensitivity, false_positive_rate, normal_n, abnormal_n = [], [], [], []
    for level in levels:
        level_rows = [row for row in validation if row["level"] == level]
        y, scores = values(level_rows)
        result = counts(y, scores, threshold)
        abnormal_n.append(int(y.sum()))
        normal_n.append(int((~y).sum()))
        sensitivity.append(result["tp"] / abnormal_n[-1])
        false_positive_rate.append(result["fp"] / normal_n[-1])
    x = np.arange(len(levels))
    width = 0.37
    ax.bar(x - width / 2, sensitivity, width, color="#009e73",
           label="Abnormal sensitivity")
    ax.bar(x + width / 2, false_positive_rate, width, color="#d55e00",
           label="Normal false-alert rate")
    ax.set_xticks(x, levels)
    ax.set(ylim=(0, 1.12), ylabel="Fraction", title="Validation decisions by level")
    for index, (n_abnormal, n_normal) in enumerate(zip(abnormal_n, normal_n)):
        ax.text(index, 1.03, f"A={n_abnormal} N={n_normal}",
                ha="center", fontsize=8)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle("SPIDER attention VGG16: normal vs bulging or herniation",
                 fontsize=16, fontweight="bold")
    args.output_dir.mkdir(parents=True)
    figure_path = args.output_dir / "performance.png"
    fig.savefig(figure_path, dpi=180, facecolor="white")
    plt.close(fig)
    summary = {
        "training_oof": {"discs": len(oof), "ap": float(average_precision_score(oof_y, oof_s)),
                         **counts(oof_y, oof_s, threshold)},
        "reused_validation": {"discs": len(validation),
                              "ap": float(average_precision_score(val_y, val_s)),
                              **val_counts},
        "threshold_from_training_oof": threshold,
        "figure": str(figure_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
