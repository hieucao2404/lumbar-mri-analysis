"""Classifier-only transfer checks on external single-slice lumbar images.

LSMA supplies report-derived None/Diffuse Bulge labels for lower levels.
Roboflow supplies disc/hdisc boxes without lumbar levels. Neither source
matches SPIDER's eight-slice expert-graded task; scores stay separate.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.train_spider_vgg16_disc_status import CLASSES, make_head
from src.inference.herniation_levels import make_stack
from src.models.abnormal_disc_cnn import AbnormalDiscCNN
from src.models.abnormal_disc_vgg16 import VGG16AbnormalDisc


LSMA_MANIFEST = Path("data/processed/lsma_binary_candidate_v1/manifest.csv")
LSMA_NOTES = Path(
    "data/raw/LSMA-PQR Lumbar Spine Multi-view Annotations with/"
    "LSMA-PQR Lumbar Spine Multi-view Annotations with/structured_notes.csv"
)
LSMA_BAGS = Path("data/processed/lsma_pqr_weak_bags_v1")
ROBOFLOW = Path("data/raw/disc herniation in lumbar spine.v4i.yolov4pytorch")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_models(device):
    binary_checkpoint = torch.load(
        "outputs/spider_vgg16_disc_status_guarded_v1/binary_herniation/fixed.pt",
        map_location="cpu", weights_only=True,
    )
    three_checkpoint = torch.load(
        "outputs/spider_vgg16_disc_status_guarded_v1/three_class/fixed.pt",
        map_location="cpu", weights_only=True,
    )
    abnormal_checkpoint = torch.load(
        "outputs/spider_abnormal_disc_vgg16_guarded_full_v1/fixed.pt",
        map_location="cpu", weights_only=True,
    )
    encoder_state = binary_checkpoint["model_state_dict"]
    for other in (three_checkpoint, abnormal_checkpoint):
        other_state = other["model_state_dict"]
        for key, value in encoder_state.items():
            if key.startswith("slice_encoder.") and not torch.equal(value, other_state[key]):
                raise ValueError("External models do not share the same frozen encoder")

    encoder = VGG16AbnormalDisc(pretrained=False).to(device)
    encoder.load_state_dict(encoder_state)
    encoder.eval()

    def head_from(checkpoint, outputs):
        head = make_head(outputs).to(device)
        state = {
            key.removeprefix("slice_head."): value
            for key, value in checkpoint["model_state_dict"].items()
            if key.startswith("slice_head.")
        }
        head.load_state_dict(state)
        return head.eval()

    heads = {
        "binary": head_from(binary_checkpoint, 1),
        "three": head_from(three_checkpoint, 3),
        "abnormal": head_from(abnormal_checkpoint, 1),
    }
    small_checkpoint = torch.load(
        "outputs/spider_abnormal_disc_cnn_fixed_v1/fixed.pt",
        map_location="cpu", weights_only=True,
    )
    small = AbnormalDiscCNN().to(device)
    small.load_state_dict(small_checkpoint["model_state_dict"])
    small.eval()
    return encoder, heads, small, {
        "binary": float(binary_checkpoint["herniation_threshold"]),
        "three": float(three_checkpoint["herniation_threshold"]),
    }


@torch.no_grad()
def score_crops(crops, encoder, heads, small, device, batch_size=16):
    result = {key: [] for key in (*heads, "small")}
    result.update({f"three_{name}": [] for name in CLASSES})
    for start in range(0, len(crops), batch_size):
        single = torch.from_numpy(np.stack(crops[start:start + batch_size]))
        single = single[:, None].float().to(device) / 255.0
        rgb = single.expand(-1, 3, -1, -1)
        rgb = (rgb - encoder.mean) / encoder.std
        features = encoder.pool(encoder.slice_encoder(rgb)).flatten(1)
        binary = heads["binary"](features).squeeze(-1).sigmoid()
        three = heads["three"](features).softmax(dim=1)
        abnormal = heads["abnormal"](features).squeeze(-1).sigmoid()
        # All eight slices are identical in these external sources.
        stacks = single.unsqueeze(1).expand(-1, 8, -1, -1, -1)
        small_score = small(stacks).sigmoid()
        result["binary"].extend(binary.cpu().tolist())
        result["three"].extend(three[:, 2].cpu().tolist())
        result["abnormal"].extend(abnormal.cpu().tolist())
        result["small"].extend(small_score.cpu().tolist())
        for index, name in enumerate(CLASSES):
            result[f"three_{name}"].extend(three[:, index].cpu().tolist())
    return result


def lsma_entries():
    entries = []
    for row in read_csv(LSMA_MANIFEST):
        entries.append({
            "patient_id": row["patient_id"], "level": row["level"],
            "external_label": row["disc_type"],
            "bag_path": row["bag_path"], "bag_index": int(row["bag_index"]),
            "label_scope": "report-derived None versus Diffuse Bulge",
        })
    # Central Protrusion is reported separately without mapping it to
    # SPIDER's expert binary herniation grade.
    for row in read_csv(LSMA_NOTES):
        for level, column, bag_index in (
            ("L4-L5", "L4_L5_Disc_Type", 1),
            ("L5-S1", "L5_S1_Disc_Type", 0),
        ):
            if row[column] == "Central Protrusion":
                entries.append({
                    "patient_id": f"lsma_{row['Patient_ID']}", "level": level,
                    "external_label": "Central Protrusion",
                    "bag_path": str(LSMA_BAGS / f"patient_{int(row['Patient_ID']):04d}.npy"),
                    "bag_index": bag_index,
                    "label_scope": "unmapped protrusion; score distribution only",
                })
    keys = [(item["patient_id"], item["level"]) for item in entries]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate LSMA patient-level entry")
    crops = []
    bag_cache = {}
    for item in entries:
        path = item["bag_path"]
        if path not in bag_cache:
            bag_cache[path] = np.load(path, allow_pickle=False)
        crop = bag_cache[path][item["bag_index"], 0]
        if crop.shape != (160, 160) or crop.dtype != np.uint8:
            raise ValueError(f"Invalid LSMA crop: {path}")
        crops.append(crop)
    return entries, crops


def roboflow_entries():
    entries, crops, seen_sources = [], [], set()
    for split in ("valid", "test"):
        root = ROBOFLOW / split
        for line in (root / "_annotations.txt").read_text().splitlines():
            parts = line.split()
            filename = parts[0]
            source_id = filename.split("_jpeg_jpg.rf.")[0]
            if source_id in seen_sources:
                raise ValueError(f"Repeated external source scan: {source_id}")
            seen_sources.add(source_id)
            image = np.asarray(Image.open(root / filename).convert("L"))
            boxes = []
            for token in parts[1:]:
                x1, y1, x2, y2, class_id = map(int, token.split(","))
                if class_id not in (0, 1) or not (0 <= x1 < x2 <= image.shape[1]
                                                    and 0 <= y1 < y2 <= image.shape[0]):
                    raise ValueError(f"Invalid Roboflow annotation: {filename} {token}")
                boxes.append(([x1, y1, x2, y2], class_id))
            boxes.sort(key=lambda item: (item[0][1] + item[0][3]) / 2)
            centers = [(box[1] + box[3]) / 2 for box, _ in boxes]
            for index, (box, class_id) in enumerate(boxes):
                top = 0 if index == 0 else round((centers[index - 1] + centers[index]) / 2)
                bottom = (image.shape[0] if index == len(boxes) - 1
                          else round((centers[index] + centers[index + 1]) / 2))
                if not top < bottom:
                    raise ValueError(f"Overlapping Roboflow level bands: {filename}")
                crop = make_stack(
                    image[:, :, None], 0, box, crop_scale=1.5,
                    posterior_shift=0.25, band=(top, bottom),
                )[0, 0]
                entries.append({
                    "split": split, "source_id": source_id, "filename": filename,
                    "box": json.dumps(box), "external_label": "hdisc" if class_id else "disc",
                    "label_scope": "Roboflow disc/hdisc object box; no lumbar level",
                })
                crops.append(crop)
    return entries, crops, len(seen_sources)


def binary_metrics(rows, scores, threshold=None):
    truth = np.asarray([row["external_label"] in ("Diffuse Bulge", "hdisc")
                        for row in rows], dtype=bool)
    scores = np.asarray(scores)
    result = {
        "discs": len(rows), "positives": int(truth.sum()),
        "prevalence": float(truth.mean()),
        "ap": float(average_precision_score(truth, scores)),
        "roc_auc": float(roc_auc_score(truth, scores)),
    }
    if threshold is not None:
        flags = scores >= threshold
        tp = int((flags & truth).sum())
        fp = int((flags & ~truth).sum())
        fn = int((~flags & truth).sum())
        tn = int((~flags & ~truth).sum())
        result.update(threshold=float(threshold), tp=tp, fp=fp, fn=fn, tn=tn,
                      sensitivity=tp / (tp + fn),
                      precision=tp / (tp + fp) if tp + fp else 0.0)
    return result


def main():
    parser = argparse.ArgumentParser()
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
    encoder, heads, small, thresholds = load_models(device)
    lsma_rows, lsma_crops = lsma_entries()
    lsma_scores = score_crops(lsma_crops, encoder, heads, small, device)
    for index, row in enumerate(lsma_rows):
        row.update({f"{key}_score": value[index] for key, value in lsma_scores.items()})
    comparable = [row for row in lsma_rows if row["external_label"] in ("None", "Diffuse Bulge")]
    vgg_calibration = json.loads(Path(
        "outputs/spider_abnormal_disc_vgg16_guarded_full_v1/calibration.json"
    ).read_text())
    small_calibration = json.loads(Path(
        "outputs/spider_abnormal_disc_cnn_fixed_v1/calibration.json"
    ).read_text())
    lsma_report = {
        "label_counts": dict(Counter(row["external_label"] for row in lsma_rows)),
        "none_vs_diffuse_bulge": {
            "vgg16_abnormal": binary_metrics(
                comparable, [row["abnormal_score"] for row in comparable],
                vgg_calibration["threshold"]),
            "lightweight_cnn_abnormal": binary_metrics(
                comparable, [row["small_score"] for row in comparable],
                small_calibration["threshold"]),
            "three_class_bulge_probability": binary_metrics(
                comparable, [row["three_bulging_only_score"] for row in comparable]),
        },
        "central_protrusion_unmapped": {
            "count": sum(row["external_label"] == "Central Protrusion" for row in lsma_rows),
            "binary_herniation_score_mean": float(np.mean([
                row["binary_score"] for row in lsma_rows
                if row["external_label"] == "Central Protrusion"])),
            "three_class_herniation_score_mean": float(np.mean([
                row["three_score"] for row in lsma_rows
                if row["external_label"] == "Central Protrusion"])),
            "note": "Central Protrusion is not mapped to SPIDER expert herniation labels.",
        },
        "input": "supplied mask crop from one sagittal T2 image, repeated eight times",
    }

    robo_rows, robo_crops, sources = roboflow_entries()
    robo_scores = score_crops(robo_crops, encoder, heads, small, device)
    for index, row in enumerate(robo_rows):
        row.update({f"{key}_score": value[index] for key, value in robo_scores.items()})
    robo_report = {
        "source_images": sources,
        "label_counts": dict(Counter(row["external_label"] for row in robo_rows)),
        "binary_herniation": binary_metrics(
            robo_rows, robo_scores["binary"], thresholds["binary"]),
        "three_class_herniation_probability": binary_metrics(
            robo_rows, robo_scores["three"], thresholds["three"]),
        "input": "Roboflow annotated disc box on one equalized sagittal JPEG, repeated eight times",
        "caveat": "disc/hdisc object labels are not SPIDER expert per-level herniation grades.",
    }
    args.output.mkdir(parents=True)
    write_csv(args.output / "lsma_predictions.csv", lsma_rows)
    write_csv(args.output / "roboflow_predictions.csv", robo_rows)
    report = {
        "task": "classifier-only external transfer, no external retraining or calibration",
        "lsma": lsma_report, "roboflow": robo_report,
        "limitations": [
            "External datasets differ in labels, acquisition and slice count.",
            "Supplied annotations define crops; end-to-end level localization is not tested.",
            "Roboflow valid/test images have no lumbar level labels or verified SPIDER patient independence.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
