"""Evaluate the saved SPIDER abnormal VGG16 on external LSMA discs.

LSMA's single supplied sagittal crop is repeated across eight slots. This is
a classifier-only transfer check on report-derived None versus Diffuse Bulge.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_vgg16_external_transfer import (
    binary_metrics,
    lsma_entries,
    write_csv,
)
from src.models.attention_vgg16_disc import AttentionVGG16Disc
from src.models.lstm_attention_vgg16_disc import LSTMAttentionVGG16Disc
from src.models.trainable_vgg16_disc import TrainableVGG16Disc


@torch.no_grad()
def score_crops(model, crops, device, batch_size=16):
    model.eval()
    scores = []
    for start in range(0, len(crops), batch_size):
        single = torch.from_numpy(np.stack(crops[start:start + batch_size]))
        single = single[:, None, None].float().to(device) / 255.0
        stacks = single.expand(-1, 8, -1, -1, -1)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            probabilities = model(stacks).sigmoid()
        scores.extend(probabilities.float().cpu().tolist())
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("outputs/spider_abnormal_vgg16_finetuned_guarded_v1/fixed.pt"),
    )
    parser.add_argument(
        "--calibration", type=Path,
        default=Path("outputs/spider_abnormal_vgg16_finetuned_guarded_v1/calibration.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_kind = checkpoint.get("model_kind")
    if model_kind not in (
        "abnormal_disc_vgg16_finetuned",
        "abnormal_disc_vgg16_slice_attention",
        "abnormal_disc_vgg16_bilstm_attention",
    ):
        raise ValueError(f"Unsupported binary abnormal checkpoint: {model_kind}")
    calibration = json.loads(args.calibration.read_text())
    if calibration["calibration_source"] != str(args.checkpoint.parent / "oof_predictions.csv"):
        raise ValueError("Calibration does not belong to this checkpoint directory")
    threshold = float(calibration["threshold"])

    model_types = {
        "abnormal_disc_vgg16_finetuned": lambda: TrainableVGG16Disc(
            num_classes=1, pretrained=False
        ),
        "abnormal_disc_vgg16_slice_attention": lambda: AttentionVGG16Disc(
            pretrained=False
        ),
        "abnormal_disc_vgg16_bilstm_attention": lambda: LSTMAttentionVGG16Disc(
            pretrained=False
        ),
    }
    model = model_types[model_kind]().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    all_rows, all_crops = lsma_entries()
    comparable = [
        (row, crop) for row, crop in zip(all_rows, all_crops)
        if row["external_label"] in ("None", "Diffuse Bulge")
    ]
    rows = [row for row, _ in comparable]
    selected_crops = [crop for _, crop in comparable]
    scores = score_crops(model, selected_crops, device)
    if len(rows) != len(scores):
        raise RuntimeError("Prediction count mismatch")
    for row, score in zip(rows, scores):
        row["abnormal_score"] = score
        row["abnormal_flag"] = int(score >= threshold)

    report = {
        "task": "external classifier-only normal versus diffuse bulge transfer",
        "model": str(args.checkpoint),
        "model_kind": model_kind,
        "threshold_source": calibration["calibration_source"],
        "input": "one supplied sagittal T2 mask crop repeated across eight model slots",
        "overall": binary_metrics(rows, scores, threshold),
        "by_level": {
            level: binary_metrics(
                [row for row in rows if row["level"] == level],
                [row["abnormal_score"] for row in rows if row["level"] == level],
                threshold,
            ) for level in ("L4-L5", "L5-S1")
        },
        "limitations": [
            "LSMA None/Diffuse Bulge are report-derived labels, not SPIDER expert grades.",
            "Only one supplied slice and annotation-based crop are available per disc.",
            "This does not evaluate MRI-only localization or lumbar level assignment.",
        ],
    }
    args.output.mkdir(parents=True)
    write_csv(args.output / "predictions.csv", rows)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
