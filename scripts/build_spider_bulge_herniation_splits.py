import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


IVD = {
    "L5-S1": "1",
    "L4-L5": "2",
    "L3-L4": "3",
    "L2-L3": "4",
    "L1-L2": "5",
}
FIELDS = (
    "patient_id", "series_name", "level", "split",
    "bulging", "herniation", "status",
    "stack_path", "sample_weight",
)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def status(bulging, herniation):
    if herniation:
        return "herniation"
    if bulging:
        return "bulging_only"
    return "normal"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/processed/herniation_level_stacks/manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/spider_bulge_herniation_v1"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)

    source = {
        (row["series_name"], row["ivd_label"]): row
        for row in read_csv(Path("data/processed/spider_discs.csv"))
    }
    crops = read_csv(args.source_manifest)

    rows = []
    patient_split = {}

    for crop in crops:
        key = (crop["series_name"], IVD[crop["level"]])
        annotation = source[key]

        bulging = int(annotation["disc_bulging"])
        herniation = int(annotation["disc_herniation"])

        if str(herniation) != crop["herniation"]:
            raise ValueError(f"Herniation label mismatch: {key}")

        patient = crop["patient_id"]
        split = crop["split"]
        previous = patient_split.setdefault(patient, split)
        if previous != split:
            raise ValueError(f"Patient in both splits: {patient}")

        rows.append({
            "patient_id": patient,
            "series_name": crop["series_name"],
            "level": crop["level"],
            "split": split,
            "bulging": bulging,
            "herniation": herniation,
            "status": status(bulging, herniation),
            "stack_path": crop["stack_path"],
            "sample_weight": "",
        })

    training = [r for r in rows if r["split"] == "training"]
    validation = [r for r in rows if r["split"] == "validation"]

    # Equal total sampling weight for each of the three displayed groups.
    # Within each group, first balance patients, then their discs.
    discs_per_patient_status = Counter(
        (r["status"], r["patient_id"]) for r in training
    )
    patients_per_status = Counter(
        group for group, _patient in discs_per_patient_status
    )

    for row in training:
        group = row["status"]
        patient = row["patient_id"]
        row["sample_weight"] = (
            1.0
            / patients_per_status[group]
            / discs_per_patient_status[(group, patient)]
        )

    # Diagnostic subset only. Do not select checkpoints on this set.
    smallest_class = min(
        Counter(r["status"] for r in validation).values()
    )
    rng = random.Random(args.seed)
    balanced = []

    for group in ("normal", "bulging_only", "herniation"):
        candidates = [
            r for r in validation if r["status"] == group
        ]
        rng.shuffle(candidates)

        selected = []
        seen_patients = set()
        for row in candidates:
            if row["patient_id"] not in seen_patients:
                selected.append(row)
                seen_patients.add(row["patient_id"])
            if len(selected) == smallest_class:
                break

        if len(selected) != smallest_class:
            raise ValueError(f"Not enough distinct patients: {group}")
        balanced.extend(selected)

    args.output.mkdir(parents=True)
    write_csv(args.output / "training.csv", training)
    write_csv(args.output / "validation_natural.csv", validation)
    write_csv(args.output / "validation_balanced.csv", balanced)

    report = {
        "source_manifest": str(args.source_manifest),
        "training": dict(Counter(r["status"] for r in training)),
        "validation_natural": dict(
            Counter(r["status"] for r in validation)
        ),
        "validation_balanced": dict(
            Counter(r["status"] for r in balanced)
        ),
        "note": (
            "Balanced validation is a subset of natural validation, "
            "not an independent test set."
        ),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
