"""Compare the fixed SPIDER localizer with LSS mid-sagittal disc polygons.

This checks anatomical level boxes only. LSS polygons do not label disc
bulging or herniation, so this audit cannot score either classifier.
"""

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pydicom
import torch

from scripts.export_spider_disc_localization import letterbox_pair, normalize_slice
from src.inference.herniation_levels import (
    LEVELS,
    LOCALIZATION_CHECKPOINT,
    load_model,
    localize_levels,
)


DEFAULT_ROOT = Path("data/raw/lss_mri_dataset")


def reference_boxes(xml_path: Path) -> tuple[str, dict[str, list[int]]]:
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename")
    boxes = {}
    for item in root.findall("object"):
        level = item.findtext("name")
        if level not in LEVELS:
            continue
        points = [(int(p.findtext("x")), int(p.findtext("y")))
                  for p in item.findall("polygon/pt")]
        if not points:
            continue
        if level in boxes:
            raise ValueError(f"Duplicate level polygon: {xml_path} {level}")
        boxes[level] = [min(x for x, _ in points), min(y for _, y in points),
                        max(x for x, _ in points) + 1,
                        max(y for _, y in points) + 1]
    return filename, boxes


def iou(a: list[int], b: list[int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    model = load_model(LOCALIZATION_CHECKPOINT, device)
    rows, failures = [], []
    for xml_path in sorted((args.root / "Segmentation/Masks").glob("*.xml")):
        patient = xml_path.name.split("_")[0]
        try:
            filename, references = reference_boxes(xml_path)
            source_name = Path(filename).stem.split("_", 1)[-1] + ".dcm"
            dicom_path = args.root / "DICOM" / patient / source_name
            dcm = pydicom.dcmread(dicom_path)
            pixels = dcm.pixel_array
            if pixels.ndim != 2:
                raise ValueError("Expected 2D mid-sagittal DICOM")
            image, _ = letterbox_pair(
                normalize_slice(pixels), np.zeros_like(pixels, dtype=np.uint8), 256)
            volume_shape = np.empty((*pixels.shape, 1), dtype=np.uint8)
            predicted, _ = localize_levels(
                {"series_name": f"lss_{patient}"}, volume_shape,
                model, device, localization_image=image)
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            failures.append({"patient_id": patient, "error": str(error)})
            continue
        for level in LEVELS:
            reference = references.get(level)
            box = predicted.get(level, {}).get("box")
            if reference is None:
                continue
            rows.append({
                "patient_id": patient, "level": level,
                "reference_box": json.dumps(reference),
                "predicted_box": json.dumps(box) if box else "",
                "iou": iou(reference, box) if box else 0.0,
                "center_inside_reference": int(
                    reference[0] <= (box[0] + box[2]) / 2 < reference[2]
                    and reference[1] <= (box[1] + box[3]) / 2 < reference[3]
                ) if box else 0,
            })
    args.output.mkdir(parents=True)
    with (args.output / "boxes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "patient_id", "level", "reference_box", "predicted_box", "iou",
            "center_inside_reference"))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "failures.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("patient_id", "error"))
        writer.writeheader()
        writer.writerows(failures)
    valid = [row for row in rows if row["predicted_box"]]
    report = {
        "patients_with_xml": len({r["patient_id"] for r in rows}) + len(failures),
        "patients_failed": len(failures),
        "reference_discs": len(rows),
        "predicted_discs_at_same_level": len(valid),
        "mean_box_iou": float(np.mean([r["iou"] for r in rows])) if rows else None,
        "iou_at_least_0_5": sum(r["iou"] >= 0.5 for r in rows),
        "center_inside_reference": sum(r["center_inside_reference"] for r in rows),
        "by_level": {
            level: {
                "reference_discs": len(group),
                "iou_at_least_0_5": sum(r["iou"] >= 0.5 for r in group),
                "center_inside_reference": sum(r["center_inside_reference"]
                                               for r in group),
            }
            for level in LEVELS
            for group in [[r for r in rows if r["level"] == level]]
        },
        "interpretation": "Anatomical boxes on annotated middle slices; no pathology labels.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
