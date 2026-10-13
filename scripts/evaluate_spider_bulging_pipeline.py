import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


LEVEL_BY_IVD = {
    1: "L5-S1",
    2: "L4-L5",
    3: "L3-L4",
    4: "L2-L3",
    5: "L1-L2",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def run_module(arguments: list[str]) -> None:
    command = [sys.executable, "-m", *arguments]

    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if completed.returncode != 0:
        print(completed.stdout)
        raise RuntimeError(
            f"Command failed: {' '.join(command)}"
        )


def average_precision(
    truth: np.ndarray,
    scores: np.ndarray,
) -> float | None:
    positives = int(truth.sum())

    if positives == 0:
        return None

    order = np.argsort(-scores, kind="stable")
    ranked_truth = truth[order]

    cumulative_positives = np.cumsum(ranked_truth)
    ranks = np.arange(1, len(truth) + 1)

    precision_at_rank = cumulative_positives / ranks

    return float(
        precision_at_rank[ranked_truth == 1].sum()
        / positives
    )


def summarize(rows: list[dict], threshold: float) -> dict:
    total = len(rows)
    available = [
        row
        for row in rows
        if row["probability"] is not None
    ]

    if not available:
        return {
            "reference_discs": total,
            "available_predictions": 0,
        }

    truth = np.asarray(
        [row["ground_truth_bulging"] for row in available],
        dtype=np.int64,
    )
    scores = np.asarray(
        [row["probability"] for row in available],
        dtype=np.float64,
    )
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

    missing_positive = sum(
        row["ground_truth_bulging"] == 1
        for row in rows
        if row["probability"] is None
    )

    positive_total = sum(
        row["ground_truth_bulging"] == 1
        for row in rows
    )

    # A useful pipeline-level measure: a positive disc must first
    # receive an available level prediction, then test positive.
    recall_over_all_positive_references = (
        tp / positive_total
        if positive_total
        else 0.0
    )

    # Since most validation discs are positive, report the simple
    # always-positive comparison as well.
    always_positive_f1 = (
        2 * positive_total / (total + positive_total)
        if total + positive_total
        else 0.0
    )

    return {
        "reference_discs": total,
        "reference_positive": positive_total,
        "available_predictions": len(available),
        "missing_predictions": total - len(available),
        "missing_positive": missing_positive,
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn_among_available": fn,
        "tn": tn,
        "precision_among_available": precision,
        "recall_among_available": recall,
        "f1_among_available": f1,
        "average_precision_among_available": (
            average_precision(truth, scores)
        ),
        "positive_recall_over_all_references": (
            recall_over_all_positive_references
        ),
        "always_positive_f1_on_all_references": (
            always_positive_f1
        ),
    }


def print_summary(name: str, summary: dict) -> None:
    print(f"\n{name}")
    print("  Reference discs:", summary["reference_discs"])
    print("  Reference positives:", summary["reference_positive"])
    print("  Available predictions:", summary["available_predictions"])
    print("  Missing predictions:", summary["missing_predictions"])
    print("  Missing positives:", summary["missing_positive"])

    print(
        "  TP/FP/FN/TN among available:",
        summary["tp"],
        summary["fp"],
        summary["fn_among_available"],
        summary["tn"],
    )

    for label, key in [
        ("Precision", "precision_among_available"),
        ("Recall", "recall_among_available"),
        ("F1", "f1_among_available"),
        ("AP", "average_precision_among_available"),
        (
            "Positive recall over all references",
            "positive_recall_over_all_references",
        ),
        (
            "Always-positive F1",
            "always_positive_f1_on_all_references",
        ),
    ]:
        value = summary[key]
        print(
            f"  {label}: "
            f"{value:.4f}" if value is not None
            else f"  {label}: unavailable"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--crop-manifest",
        type=Path,
        default=Path(
            "data/processed/spider_crops/crops.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/spider_bulging_pipeline_validation"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use 2 for a smoke test; omit for full validation.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse JSON files already generated in output-dir.",
    )

    args = parser.parse_args()

    if not 0 < args.threshold < 1:
        raise ValueError("Threshold must be between 0 and 1")

    manifest_rows = read_csv(args.crop_manifest)

    validation_rows = [
        row
        for row in manifest_rows
        if row["subset"] == "validation"
        and int(row["ivd_label"]) in LEVEL_BY_IVD
    ]

    if not validation_rows:
        raise ValueError(
            "No validation lumbar discs in the crop manifest"
        )

    series_names = sorted({
        row["series_name"]
        for row in validation_rows
    })

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        series_names = series_names[:args.limit]

    selected_series = set(series_names)
    validation_rows = [
        row
        for row in validation_rows
        if row["series_name"] in selected_series
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_dir = args.output_dir / "studies"
    json_dir.mkdir(exist_ok=True)

    predictions_by_series = {}

    for index, series_name in enumerate(series_names, start=1):
        base_json = json_dir / f"{series_name}.json"
        bulging_json = (
            json_dir / f"{series_name}_with_bulging.json"
        )

        if not args.reuse_existing or not base_json.is_file():
            run_module([
                "scripts.run_spider_sagittal_pipeline",
                "--series-name",
                series_name,
                "--output",
                str(base_json),
            ])

        if not args.reuse_existing or not bulging_json.is_file():
            run_module([
                "scripts.add_spider_bulging",
                "--series-name",
                series_name,
                "--pipeline-json",
                str(base_json),
                "--output",
                str(bulging_json),
                "--bulging-threshold",
                str(args.threshold),
            ])

        with bulging_json.open() as file:
            result = json.load(file)

        predictions_by_series[series_name] = {
            level["position"]: level
            for level in result["levels"]
        }

        print(
            f"[{index}/{len(series_names)}] "
            f"{series_name}"
        )

    comparison = []
    seen = set()

    for row in validation_rows:
        series_name = row["series_name"]
        level_name = LEVEL_BY_IVD[int(row["ivd_label"])]
        key = (series_name, level_name)

        if key in seen:
            raise ValueError(
                f"Duplicate reference level: {key}"
            )
        seen.add(key)

        level_result = predictions_by_series[
            series_name
        ].get(level_name)

        probability = None

        if (
            level_result is not None
            and level_result.get("status") == "available"
        ):
            probability = level_result.get(
                "disc_bulging_probability"
            )

        comparison.append({
            "patient_id": row["patient_id"],
            "series_name": series_name,
            "level": level_name,
            "ground_truth_bulging": int(
                row["disc_bulging"]
            ),
            "probability": (
                float(probability)
                if probability is not None
                else None
            ),
            "prediction": (
                int(float(probability) >= args.threshold)
                if probability is not None
                else None
            ),
        })

    all_summary = summarize(
        comparison,
        args.threshold,
    )

    # These two patients previously had uncertain SPIDER disc
    # numbering. Report the remaining subset separately.
    reliable_rows = [
        row
        for row in comparison
        if row["patient_id"] not in {"107", "152"}
    ]
    reliable_summary = summarize(
        reliable_rows,
        args.threshold,
    )

    csv_path = args.output_dir / "per_disc_predictions.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(comparison[0].keys()),
        )
        writer.writeheader()
        writer.writerows(comparison)

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w") as file:
        json.dump(
            {
                "all_validation": all_summary,
                "excluding_uncertain_numbering": (
                    reliable_summary
                ),
                "evaluated_series": len(series_names),
                "is_partial_run": args.limit is not None,
            },
            file,
            indent=2,
        )

    print_summary("All selected validation studies", all_summary)
    print_summary(
        "Excluding patients 107 and 152",
        reliable_summary,
    )
    print("\nPredictions:", csv_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()