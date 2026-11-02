"""Disc-level axial-review flags from sagittal SPIDER T2 scans.

The box marks the localized disc, not a herniation lesion. A positive flag
means bulging or herniation is possible and axial MRI review is recommended.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from src.inference.mri_slice_selector import select_sagittal_slice

from src.inference.herniation_levels import (
    LEVELS, LOCALIZATION_CHECKPOINT, level_bands, load_model,
    load_series_volume, localize_levels, make_stack, read_localization_records,
)
from src.models.abnormal_disc_cnn import AbnormalDiscCNN

DEFAULT_CHECKPOINT = Path("outputs/spider_abnormal_disc_cnn_fixed_v1/fixed.pt")
DEFAULT_CALIBRATION = Path("outputs/spider_abnormal_disc_cnn_fixed_v1/calibration.json")


def load_review_classifier(checkpoint_path, calibration_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_kind") != "abnormal_disc_cnn":
        raise ValueError("Expected an abnormal-disc review checkpoint")
    calibration = json.loads(calibration_path.read_text())
    if calibration.get("task") != "flag possible bulging or herniation for axial review":
        raise ValueError("Calibration target does not match the review classifier")
    threshold = float(calibration["threshold"])
    if not 0 < threshold < 1:
        raise ValueError("Invalid review threshold")
    classifier = AbnormalDiscCNN().to(device)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()
    return classifier, threshold


@torch.inference_mode()
def predict_review_flags(series_name, records, localization_model,
                         classifier, threshold, device):
    if series_name not in records:
        raise KeyError(f"Unknown SPIDER series: {series_name}")
    record = records[series_name]
    volume = load_series_volume(series_name)
    selected_slice, localization_image, slice_selection = (
        select_sagittal_slice(
            record,
            volume,
            localization_model,
            device,
            middle_window=2,
        )
    )

    predicted, decoder = localize_levels(
        record,
        volume,
        localization_model,
        device,
        localization_image=localization_image,
    )
    bands = level_bands(predicted, volume.shape[0])
    available = [level for level in LEVELS if level in predicted]
    scores = {}
    if available:
        stacks = np.stack([
            make_stack(volume, selected_slice, predicted[level]["box"],
                       crop_scale=1.5, posterior_shift=0.25,
                       band=bands[level])
            for level in available
        ])
        inputs = torch.from_numpy(stacks.astype(np.float32) / 255.0).to(device)
        scores = dict(zip(available, classifier(inputs).sigmoid().cpu().tolist()))

    levels = []
    review_levels = []
    for level in LEVELS:
        if level not in predicted:
            levels.append({"level": level, "status": "unavailable",
                           "reason": "No disc component assigned to this level",
                           "review_recommended": None})
            continue
        score = float(scores[level])
        flag = score >= threshold
        levels.append({
            "level": level, "status": "available",
            "disc_bounding_box_original": predicted[level]["box"],
            "box_type": "localized disc, not lesion extent",
            "level_model_probability": predicted[level]["level_model_probability"],
            "abnormal_disc_score": score,
            "review_recommended": flag,
            "review_reason": "Possible bulging or herniation; verify on axial MRI" if flag else None,
        })
        if flag:
            review_levels.append(level)
    return {
        "study_id": record["patient_id"], "series_name": series_name,
        "sequence": "sagittal T2", "selected_slice": selected_slice,
        "status": "partial" if len(available) < len(LEVELS) else "ok",
        "level_decoder": decoder, "review_threshold": threshold,
        "review_levels": review_levels, "axial_review_performed": False,
        "levels": levels,
        "slice_selection": slice_selection,
        "interpretation": "Review flag for possible disc bulging or herniation; axial MRI verification required.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series-name", required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    records = read_localization_records()
    localizer = load_model(LOCALIZATION_CHECKPOINT, device)
    classifier, threshold = load_review_classifier(args.checkpoint, args.calibration, device)
    result = predict_review_flags(args.series_name, records, localizer,
                                  classifier, threshold, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"series_name": args.series_name,
                      "review_levels": result["review_levels"],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
