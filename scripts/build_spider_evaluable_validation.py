"""Separate visibible levels from levels absent in the MRI."""

import csv
from pathlib import Path

DATA = Path("data/processed/spider_bulge_herniation_mri_only_geometry_v1")
SOURCE = Path("data/processed/spider_discs.csv")

LEVEL_TO_IVD = {
    "L1-L2": "5",
    "L2-L3": "4",
    "L3-L4": "3",
    "L4-L5": "2",
    "L5-S1": "1",
}

def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

source = read_csv(SOURCE)
validation = read_csv(DATA / "validation_natural.csv")

documented = {
    (row["series_name"], row["ivd_label"])
    for row in source
    if row["subset"] == "validation"
    and row["series_name"].endswith("_t2")
    and row["ivd_label"] in {"1", "2", "3", "4", "5"}
}

evaluable = []
unobservable = []

for row in validation:
    key = (row["series_name"], LEVEL_TO_IVD[row["level"]])
    if key in documented:
        evaluable.append(row)
    else:
        unobservable.append({
            **row,
            "visibility_reason": "Level absent from source MRI disc annotations",
        })

expected = {("35_t2", "L1-L2"), ("35_t2", "L2-L3")}
found = {(row["series_name"], row["level"]) for row in unobservable}
if found != expected:
    raise ValueError(f"Unexpected unobservable levels: {found}")

fields = list(validation[0])
write_csv(DATA / "validation_evaluable.csv", evaluable, fields)
write_csv(
    DATA / "validation_unobservable.csv",
    unobservable,
    fields + ["visibility_reason"],
)

print("Evaluable:", len(evaluable))
print("Unobservable:", len(unobservable))
print("Unobservable levels:", sorted(found))