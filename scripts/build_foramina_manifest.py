import argparse
import csv
import os
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image

LEVELS = {"L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"}
LABEL_PATTERN = re.compile(r"^(LFS|RFS)([0-3])$")

COLUMNS = [
    "patient_id",
    "image_path",
    "x1", "y1", "x2", "y2",
    "level",
    "side",
    "severity",
    "split",
]

class MissingImageError(FileNotFoundError):
    pass

def parse_xml(xml_path):
    root = ET.parse(xml_path).getroot()

    filename = root.findtext("filename")
    if not filename:
        raise ValueError(f"{xml_path}: missing <filename>")

    # Try the filename recorded in XML, then the XML stem with common
    # image extensions. Accept exactly one existing image.
    stem = xml_path.stem
    candidates = [
        xml_path.parent / Path(filename).name,
        xml_path.parent / f"{stem}.png",
        xml_path.parent / f"{stem}.jpg",
        xml_path.parent / f"{stem}.jpeg",
    ]

    matches = list(dict.fromkeys(
        path for path in candidates if path.is_file()
    ))

    if not matches:
        raise MissingImageError(
            f"{xml_path}: no matching image for {filename!r}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"{xml_path}: ambiguous matching images: {matches}"
        )

    image_path = matches[0]

    with Image.open(image_path) as image:
        width, height = image.size

    patient_id = xml_path.parent.name
    rows = []

    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        match = LABEL_PATTERN.fullmatch(name)

        if match is None:
            raise ValueError(f"{xml_path}: unexpected object name {name!r}")

        side_code, grade_text = match.groups()
        side = "L" if side_code == "LFS" else "R"
        severity = int(grade_text)

        level = (obj.findtext("level") or "").strip()
        if level not in LEVELS:
            raise ValueError(f"{xml_path}: invalid level {level!r}")

        box_element = obj.find("bndbox")
        if box_element is None:
            raise ValueError(f"{xml_path}: missing <bnbbox>")

        x1 = int(box_element.findtext("xmin"))
        y1 = int(box_element.findtext("ymin"))
        x2 = int(box_element.findtext("xmax"))
        y2 = int(box_element.findtext("ymax"))

        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(
                f"{xml_path}: box {(x1, y1, x2, y2)} "
                f"outside image {(image, height)}"
            )

        rows.append({
            "patient_id": patient_id,
            "image_path": image_path.resolve(),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "level": level,
            "side": side,
            "severity": severity,
        })

    return rows

def assign_patient_splits(rows, seed=42):
    patients = sorted({row["patient_id"] for row in rows})
    random.Random(seed).shuffle(patients)

    train_end = int(0.70 * len(patients))
    val_end = int(0.85 * len(patients))

    split_by_patient = {}


    for patient in patients[:train_end]:
        split_by_patient[patient] = "train"
    for patient in patients[train_end:val_end]:
        split_by_patient[patient] = "val"
    for patient in patients[val_end:]:
        split_by_patient[patient] = "test"

    for row in rows:
        row["split"] = split_by_patient[row["patient_id"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    annotation_dir = args.annotations.resolve()
    output_path = args.output.resolve()

    if not annotation_dir.is_dir():
        raise SystemExit(f"Directory not found: {annotation_dir}")

    xml_files = sorted(annotation_dir.rglob("*.xml"))
    if not xml_files:
        raise SystemExit(f"No XML files found under {annotation_dir}")

    rows = []
    missing_images = []

    for xml_path in xml_files:
        try:
            rows.extend(parse_xml(xml_path))
        except MissingImageError as exc:
            missing_images.append(str(exc))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = output_path.parent / "foramina_missing_images.txt"
    report_path.write_text(
        "\n".join(missing_images) + ("\n" if missing_images else ""),
        encoding="utf-8",
    )

    print(
        f"Missing image pairs: {len(missing_images)}/{len(xml_files)} XML files"
    )
    print(f"Missing-image report: {report_path}")

    # Stop if the download appears substantially incomplete.
    if len(missing_images) / len(xml_files) > 0.02:
        raise SystemExit(
            "More than 2% of XML files lack images. "
            "Check the dataset extraction before training."
        )

    if not rows:
        raise SystemExit("XML files were found, but no objects were parsed")

    assign_patient_splits(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()

        for row in rows:
            saved_row = row.copy()

            # Paths in the CSV are relative to the CSV's location.
            saved_row["image_path"] = os.path.relpath(
                row["image_path"],
                start=output_path.parent,
            )

            writer.writerow(saved_row)

    print(f"XML files: {len(xml_files)}")
    print(f"Foramen annotations: {len(rows)}")
    print(f"Patients with annotations: {len(set(r['patient_id'] for r in rows))}")
    print(f"Saved: {output_path}")

    for split in ("train", "val", "test"):
        subset = [row for row in rows if row["split"] == split]
        counts = Counter(row["severity"] for row in subset)
        patient_count = len({row["patient_id"] for row in subset})

        print(
            f"{split}: {patient_count} patients, "
            f"{len(subset)} ROIs, grades 0/1/2/3 = "
            f"{[counts[grade] for grade in range(4)]}"
        )


if __name__ == "__main__":
    main()

