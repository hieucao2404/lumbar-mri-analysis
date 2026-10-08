import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


LUMBAR_LEVELS = {
    1: "L5-S1",
    2: "L4-L5",
    3: "L3-L4",
    4: "L2-L3",
    5: "L1-L2",
}


def box_center_y(row: dict) -> float:
    y1 = float(row["y1"])
    y2 = float(row["y2"])

    if y2 <= y1:
        raise ValueError(
            f"Invalid box for {row['series_name']}: "
            f"y1={y1}, y2={y2}"
        )

    return (y1 + y2) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/spider_discs.csv"),
    )

    args = parser.parse_args()

    with args.manifest.open(newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Manifest is empty")

    series_discs = defaultdict(dict)
    label_counts = Counter()

    skipped_rows = 0

    for row in rows:
        label = int(row["ivd_label"])

        # Ignore unsupported or abnormal labels such as label 0.
        if label < 1:
            skipped_rows += 1
            continue

        series_name = row["series_name"]
        center_y = box_center_y(row)

        if label in series_discs[series_name]:
            raise ValueError(
                f"Duplicate IVD label {label} in {series_name}"
            )

        series_discs[series_name][label] = center_y
        label_counts[label] += 1

    correct_pairs = 0
    incorrect_pairs = 0
    comparison_examples = []

    complete_lumbar_series = 0
    incomplete_lumbar_series = 0

    for series_name, discs in series_discs.items():
        available_labels = sorted(discs)

        if all(label in discs for label in range(1, 6)):
            complete_lumbar_series += 1
        else:
            incomplete_lumbar_series += 1

        for lower_label, upper_label in zip(
            available_labels,
            available_labels[1:],
        ):
            # Only compare consecutive anatomical labels.
            if upper_label != lower_label + 1:
                continue

            lower_center_y = discs[lower_label]
            upper_center_y = discs[upper_label]

            # Image y coordinates increase toward the bottom.
            # Label 1 should therefore have a larger y value than label 2.
            correct = lower_center_y > upper_center_y

            if correct:
                correct_pairs += 1
            else:
                incorrect_pairs += 1

                if len(comparison_examples) < 10:
                    comparison_examples.append({
                        "series": series_name,
                        "lower_label": lower_label,
                        "lower_center_y": round(lower_center_y, 2),
                        "upper_label": upper_label,
                        "upper_center_y": round(upper_center_y, 2),
                    })

    total_pairs = correct_pairs + incorrect_pairs

    if total_pairs == 0:
        raise RuntimeError(
            "No consecutive disc-label pairs were available"
        )

    ordering_accuracy = correct_pairs / total_pairs

    print("Manifest:", args.manifest)
    print("Rows:", len(rows))
    print("Series:", len(series_discs))
    print("Skipped rows:", skipped_rows)
    print("Label counts:", dict(sorted(label_counts.items())))

    print()
    print("Disc-order verification")
    print("Consecutive pairs:", total_pairs)
    print("Correctly ordered pairs:", correct_pairs)
    print("Incorrectly ordered pairs:", incorrect_pairs)
    print(f"Ordering accuracy: {ordering_accuracy:.4f}")

    print()
    print("Lumbar coverage")
    print(
        "Series containing labels 1 through 5:",
        complete_lumbar_series,
    )
    print(
        "Series missing one or more labels 1 through 5:",
        incomplete_lumbar_series,
    )

    print()
    print("Proposed SPIDER-to-anatomy mapping")

    for label, level in LUMBAR_LEVELS.items():
        print(f"  IVD label {label} -> {level}")

    if comparison_examples:
        print()
        print("First ordering violations:")

        for example in comparison_examples:
            print(example)

    print()

    if ordering_accuracy >= 0.99:
        print(
            "RESULT: The proposed inferior-to-superior ordering is "
            "strongly supported."
        )
    elif ordering_accuracy >= 0.95:
        print(
            "RESULT: The ordering is mostly consistent, but the "
            "violations should be reviewed."
        )
    else:
        print(
            "RESULT: Do not hard-code the proposed mapping yet. "
            "The orientation or label interpretation needs review."
        )


if __name__ == "__main__":
    main()