import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def read_binary_label(row: dict, column: str, row_number: int) -> int:
    value = row.get(column, "").strip()

    if value not in {"0", "1"}:
        raise ValueError(
            f"Row {row_number}: {column} must be 0 or 1; "
            f"received {value!r}"
        )

    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit posterior-disc labels on SPIDER lumbar T2 crops"
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/spider_crops/crops.csv"),
    )
    args = parser.parse_args()

    with args.manifest.open(newline="") as file:
        all_rows = list(csv.DictReader(file))

    required = {
        "patient_id",
        "series_name",
        "subset",
        "ivd_label",
        "disc_bulging",
        "disc_herniation",
        "crop_path",
    }

    if not all_rows:
        raise ValueError("Crop manifest is empty")

    missing_columns = required - set(all_rows[0])

    if missing_columns:
        raise ValueError(
            f"Missing manifest columns: {sorted(missing_columns)}"
        )

    split_rows = defaultdict(list)
    split_patients = defaultdict(set)
    seen_discs = set()

    for row_number, row in enumerate(all_rows, start=2):
        series_name = row["series_name"]
        subset = row["subset"]

        # This model will receive sagittal T2 lumbar disc crops.
        if not series_name.endswith("_t2"):
            continue

        if subset not in {"training", "validation"}:
            raise ValueError(
                f"Row {row_number}: unexpected subset {subset!r}"
            )

        ivd_label = int(row["ivd_label"])

        # SPIDER: 1=L5-S1, 2=L4-L5, ... 5=L1-L2.
        # Labels 6+ describe discs above the five lumbar levels.
        if ivd_label not in range(1, 6):
            continue

        patient_id = row["patient_id"]
        disc_key = (series_name, ivd_label)

        if disc_key in seen_discs:
            raise ValueError(
                f"Duplicate crop for {disc_key}"
            )

        seen_discs.add(disc_key)

        crop_path = Path(row["crop_path"])

        if not crop_path.is_file():
            raise FileNotFoundError(
                f"Row {row_number}: missing {crop_path}"
            )

        audited_row = {
            "patient_id": patient_id,
            "series_name": series_name,
            "subset": subset,
            "ivd_label": ivd_label,
            "bulge": read_binary_label(
                row,
                "disc_bulging",
                row_number,
            ),
            "herniation": read_binary_label(
                row,
                "disc_herniation",
                row_number,
            ),
        }

        split_rows[subset].append(audited_row)
        split_patients[subset].add(patient_id)

    overlap = (
        split_patients["training"]
        & split_patients["validation"]
    )

    if overlap:
        raise ValueError(
            f"Patient overlap: {sorted(overlap)[:10]}"
        )

    level_names = {
        1: "L5-S1",
        2: "L4-L5",
        3: "L3-L4",
        4: "L2-L3",
        5: "L1-L2",
    }

    print("Manifest:", args.manifest)
    print("Patient overlap:", len(overlap))

    for subset in ("training", "validation"):
        rows = split_rows[subset]

        if not rows:
            raise ValueError(
                f"No lumbar T2 crops in {subset}"
            )

        combinations = Counter(
            (row["bulge"], row["herniation"])
            for row in rows
        )

        bulge_positive = sum(
            row["bulge"] for row in rows
        )

        herniation_positive = sum(
            row["herniation"] for row in rows
        )

        patients_with_bulge = {
            row["patient_id"]
            for row in rows
            if row["bulge"] == 1
        }

        patients_with_herniation = {
            row["patient_id"]
            for row in rows
            if row["herniation"] == 1
        }

        print(f"\n{subset.upper()}")
        print("  Patients:", len(split_patients[subset]))
        print("  Lumbar crops:", len(rows))
        print(
            "  Bulging positive:",
            bulge_positive,
            f"({bulge_positive / len(rows):.1%})",
        )
        print(
            "  Herniation positive:",
            herniation_positive,
            f"({herniation_positive / len(rows):.1%})",
        )
        print(
            "  Patients with bulging:",
            len(patients_with_bulge),
        )
        print(
            "  Patients with herniation:",
            len(patients_with_herniation),
        )

        print("  Label combinations (bulging, herniation):")

        for combination in (
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
        ):
            print(
                f"    {combination}: "
                f"{combinations[combination]}"
            )

        print("  Herniation positives by level:")

        for ivd_label in range(1, 6):
            level_rows = [
                row
                for row in rows
                if row["ivd_label"] == ivd_label
            ]

            positives = sum(
                row["herniation"]
                for row in level_rows
            )

            print(
                f"    {level_names[ivd_label]}: "
                f"{positives}/{len(level_rows)}"
            )

    print("\nAudit passed")


if __name__ == "__main__":
    main()