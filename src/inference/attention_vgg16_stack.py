"""Run binary attention VGG16 on preprocessed, level-specific disc stacks.

This entry point accepts prepared [8, 1, 160, 160] uint8 crops. It does not
localize discs or convert raw MRI from another dataset into SPIDER-style crops.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.models.attention_vgg16_disc import AttentionVGG16Disc


LEVELS = ("L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1")
MODEL_KIND = "abnormal_disc_vgg16_slice_attention"
CALIBRATION_TASK = "flag possible bulging or herniation for axial review"


def load_classifier(checkpoint_path: Path, calibration_path: Path,
                    device: torch.device) -> tuple[AttentionVGG16Disc, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_kind") != MODEL_KIND:
        raise ValueError(f"Expected {MODEL_KIND} checkpoint")
    calibration = json.loads(calibration_path.read_text())
    if calibration.get("task") != CALIBRATION_TASK:
        raise ValueError("Calibration task does not match this classifier")
    threshold = float(calibration["threshold"])
    if not 0 < threshold < 1:
        raise ValueError("Invalid review threshold")
    model = AttentionVGG16Disc(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, threshold


def read_input(path: Path) -> tuple[dict, dict[str, np.ndarray], dict[str, list[int]]]:
    request = json.loads(path.read_text())
    for field in ("study_id", "sagittal_series_name"):
        if not isinstance(request.get(field), str) or not request[field]:
            raise ValueError(f"Missing nonempty {field}")
    selected = request.get("selected_sagittal_slice")
    if selected is not None and (type(selected) is not int or selected < 0):
        raise ValueError("selected_sagittal_slice must be nonnegative or null")
    discs = request.get("discs")
    if not isinstance(discs, list) or not discs:
        raise ValueError("Input must contain at least one disc")

    stacks: dict[str, np.ndarray] = {}
    boxes: dict[str, list[int]] = {}
    for disc in discs:
        level = disc.get("level")
        if level not in LEVELS or level in stacks:
            raise ValueError(f"Unknown or duplicate level: {level}")
        stack_path = Path(disc["stack_path"])
        stack = np.load(stack_path, allow_pickle=False)
        if stack.shape != (8, 1, 160, 160) or stack.dtype != np.uint8:
            raise ValueError(f"Expected uint8 [8,1,160,160]: {stack_path}")
        stacks[level] = stack
        box = disc.get("disc_box_xyxy_original")
        if box is not None:
            if (not isinstance(box, list) or len(box) != 4
                    or any(type(value) is not int for value in box)
                    or box[0] >= box[2] or box[1] >= box[3]):
                raise ValueError(f"Invalid sagittal disc box for {level}")
            boxes[level] = box
    return request, stacks, boxes


@torch.inference_mode()
def predict_stacks(model: AttentionVGG16Disc, stacks: dict[str, np.ndarray],
                   device: torch.device) -> dict[str, float]:
    available = [level for level in LEVELS if level in stacks]
    batch = np.stack([stacks[level] for level in available])
    inputs = torch.from_numpy(batch.astype(np.float32) / 255.0).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=device.type == "cuda"):
        scores = model(inputs).sigmoid().float().cpu().tolist()
    return dict(zip(available, map(float, scores)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
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

    request, stacks, boxes = read_input(args.input)
    model, threshold = load_classifier(args.checkpoint, args.calibration, device)
    scores = predict_stacks(model, stacks, device)
    levels = []
    review_levels = []
    for level in LEVELS:
        if level not in scores:
            levels.append({
                "level": level, "status": "unavailable",
                "abnormal_disc_score": None, "review_recommended": None,
                "reason": "No preprocessed stack supplied",
            })
            continue
        score = scores[level]
        flag = score >= threshold
        item = {
            "level": level, "status": "available",
            "abnormal_disc_score": score,
            "review_recommended": flag,
        }
        if level in boxes:
            item["disc_box_xyxy_original"] = boxes[level]
            item["box_coordinate_system"] = "original sagittal image pixels"
        levels.append(item)
        if flag:
            review_levels.append(level)

    result = {
        "schema_version": "sagittal-review/v1",
        "study_id": request["study_id"],
        "sagittal_series_name": request["sagittal_series_name"],
        "selected_sagittal_slice": request.get("selected_sagittal_slice"),
        "status": "ok" if len(scores) == len(LEVELS) else "partial",
        "model_kind": MODEL_KIND,
        "review_threshold": threshold,
        "review_levels": review_levels,
        "levels": levels,
        "interpretation": "Possible bulging or herniation; axial review needed to classify subtype.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "study_id": result["study_id"], "review_levels": review_levels,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
