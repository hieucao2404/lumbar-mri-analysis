import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts.build_spider_paired_crops import (
    predicted_box_to_original,
)
from scripts.evaluate_spider_level_decoders import (
    fused_sequence_assignments,
    gated_hybrid_assignments,
    position_assignments,
)
from scripts.evaluate_spider_multitask_components import (
    add_predicted_levels,
    extract_components,
    load_model,
    predict,
)
from scripts.export_spider_herniation_stacks_v3 import (
    load_volume,
    make_fixed_crop,
)
from src.models.herniation_classifier import HerniationStackClassifier

LEVELS = (
    "L1-L2",
    "L2-L3",
    "L3-L4",
    "L4-L5",
    "L5-S1",
)

LOCALIZATION_MANIFEST = Path(
    "data/processed/spider_disc_localization/manifest.csv"
)
LOCALIZATION_CHECKPOINT = Path(
    "outputs/spider_disc_multitask_coord_v2/best_model.pt"
)
RAW_IMAGE_DIR = Path("data/raw/spider/images")


def read_localization_records() -> dict[str, dict]:
    with LOCALIZATION_MANIFEST.open(newline="") as file:
        rows = list(csv.DictReader(file))

    return {row["series_name"]: row for row in rows}


def load_series_volume(series_name: str) -> np.ndarray:
    path = RAW_IMAGE_DIR / f"{series_name}.mha"
    if not path.is_file():
        raise FileNotFoundError(path)

    volume = load_volume(path)
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {volume.shape}")

    return volume


def localize_levels(
    record: dict,
    volume: np.ndarray,
    localization_model,
    device: torch.device,
    *,
    localization_image: np.ndarray | None = None,
) -> tuple[dict[str, dict], dict]:
    """
    Return one predicted box per available lumbar level.

    Boxes use original-image [x1, y1, x2, y2] coordinates.
    """
    if localization_image is None:
        # Existing behavior, retained for comparison.
        example_path = Path(record["example_path"])
        with np.load(example_path) as data:
            image = data["image"].astype(np.float32)
    else:
        image = localization_image.astype(np.float32, copy=False)

    if image.shape != (256, 256):
        raise ValueError(
            f"Expected a 256x256 localization image, got {image.shape}"
        )

    binary_probability, level_probabilities = predict(
        localization_model, image, device
    )

    components = extract_components(
        binary_probability >= 0.5,
        minimum_area=25,
    )
    components = add_predicted_levels(
        components, level_probabilities
    )

    position = position_assignments(components)

    fused, decoder_result = fused_sequence_assignments(
        components,
        learned_weight=1.0,
        position_weight=0.75,
        upper_weight=0.20,
        maximum_spacing_cv=0.55,
    )

    assignments, decoder_source = gated_hybrid_assignments(
        position=position,
        fused=fused,
        decoder_result=decoder_result,
        spacing_cv_threshold=0.20,
    )

    height, width, _ = volume.shape
    by_level = {}

    for component_index, level_id in enumerate(assignments):
        if level_id not in range(5):
            continue

        component = components[component_index]
        component_with_series = {
            **component,
            "series_name": record["series_name"],
        }

        box = predicted_box_to_original(
            component_with_series,
            original_height=height,
            original_width=width,
            canvas_size=256,
        )

        level = LEVELS[level_id]
        candidate = {
            "box": [int(value) for value in box],
            "component_id": int(component["component_id"]),
            "level_model_probability": float(
                component["level_probabilities"][level_id]
            ),
        }

        # Defensive handling if a decoder assigns a level twice.
        previous = by_level.get(level)
        if (
            previous is None
            or candidate["level_model_probability"]
            > previous["level_model_probability"]
        ):
            by_level[level] = candidate

    decoder_info = {
        "source": decoder_source,
        "status": decoder_result.get("status", "uncertain"),
        "reasons": decoder_result.get("reasons", []),
    }
    return by_level, decoder_info


def make_stack(
    volume: np.ndarray,
    selected_slice: int,
    box: list[int],
    *,
    crop_scale: float = 1.5,
    posterior_shift: float = 0.25,
    band: tuple[int, int] | None = None,
) -> np.ndarray:
    slice_count = volume.shape[2]
    if not 0 <= selected_slice < slice_count:
        raise ValueError(f"Invalid selected slice {selected_slice}")

    indices = np.clip(
        np.arange(selected_slice - 3, selected_slice + 5),
        0,
        slice_count - 1,
    )

    stack = np.stack(
        [
            make_fixed_crop(
                volume[:, :, int(index)],
                box,
                crop_scale=crop_scale,
                posterior_shift=posterior_shift,
            )[0:1]
            for index in indices
        ],
        axis=0,
    )

    if stack.shape != (8, 1, 160, 160):
        raise ValueError(f"Unexpected stack shape {stack.shape}")

    if band is not None:
        top, bottom = band
        if not 0 <= top < bottom <= volume.shape[0]:
            raise ValueError(f"Invalid level band {band}")

        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid disc box {box}")
        center_y = (y1 + y2) / 2
        if not top <= center_y < bottom:
            raise ValueError(f"Target disc center outside band: {box}, {band}")

        # Match make_fixed_crop's source rectangle. Mask after cropping so
        # slice normalization still uses the original, unmasked MRI image.
        half_side = max(48.0, crop_scale * width, 2.5 * height)
        source_top = round(center_y - half_side)
        source_bottom = round(center_y + half_side)
        source_y = source_top + (
            np.arange(160) + 0.5
        ) * (source_bottom - source_top) / 160
        outside = (source_y < top) | (source_y >= bottom)
        stack[:, :, outside, :] = 0

    return stack


@torch.inference_mode()
def predict_series(
    series_name: str,
    records: dict[str, dict],
    localization_model,
    classifier: HerniationStackClassifier,
    threshold: float,
    device: torch.device,
) -> dict:
    if series_name not in records:
        raise KeyError(f"Unknown series: {series_name}")

    record = records[series_name]
    volume = load_series_volume(series_name)
    selected_slice = int(record["sagittal_slice"])

    predicted_levels, decoder_info = localize_levels(
        record, volume, localization_model, device
    )

    available = [
        level for level in LEVELS
        if level in predicted_levels
    ]

    probabilities = {}

    if available:
        stacks = np.stack(
            [
                make_stack(
                    volume,
                    selected_slice,
                    predicted_levels[level]["box"],
                )
                for level in available
            ],
            axis=0,
        )

        inputs = torch.from_numpy(
            stacks.astype(np.float32) / 255.0
        ).to(device)

        scores = torch.sigmoid(classifier(inputs))
        probabilities = dict(
            zip(available, scores.cpu().tolist())
        )

    level_results = []
    positive_levels = []

    for level in LEVELS:
        if level not in predicted_levels:
            level_results.append({
                "level": level,
                "status": "unavailable",
                "reason": "No disc component assigned to this level",
                "herniation_detected": None,
            })
            continue

        probability = float(probabilities[level])
        positive = probability >= threshold

        level_results.append({
            "level": level,
            "status": "available",
            "bounding_box_original": (
                predicted_levels[level]["box"]
            ),
            "level_model_probability": (
                predicted_levels[level][
                    "level_model_probability"
                ]
            ),
            "herniation_probability": probability,
            "herniation_detected": positive,
        })

        if positive:
            positive_levels.append(level)

    missing = len(available) < len(LEVELS)

    return {
        "study_id": record["patient_id"],
        "series_name": series_name,
        "sequence": "sagittal T2",
        "selected_slice": selected_slice,
        "status": "partial" if missing else "ok",
        "level_decoder": decoder_info,
        "herniation_threshold": threshold,
        "herniation_present": bool(positive_levels),
        "herniated_levels": positive_levels,
        "levels": level_results,
        "disclaimer": (
            "Research prototype; not a confirmed clinical diagnosis."
        ),
    }


def load_classifier(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[HerniationStackClassifier, float]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model = HerniationStackClassifier().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, float(checkpoint["threshold"])



def level_bands(predicted_levels, image_height):
    available = [
        level for level in LEVELS if level in predicted_levels
    ]
    centers = {
        level: (
            predicted_levels[level]["box"][1]
            + predicted_levels[level]["box"][3]
        ) / 2
        for level in available
    }

    ys = [centers[level] for level in available]
    if any(a >= b for a, b in zip(ys, ys[1:])):
        raise ValueError("Predicted levels are out of vertical order")

    bands = {}
    for i, level in enumerate(available):
        top = 0 if i == 0 else round((ys[i - 1] + ys[i]) / 2)
        bottom = (
            image_height if i == len(available) - 1
            else round((ys[i] + ys[i + 1]) / 2)
        )
        bands[level] = (top, bottom)
    return bands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series-name", required=True)
    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        default=Path(
            "outputs/herniation_levels/best.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    records = read_localization_records()

    localization_model = load_model(
        LOCALIZATION_CHECKPOINT,
        device,
    )
    classifier, threshold = load_classifier(
        args.classifier_checkpoint,
        device,
    )

    result = predict_series(
        args.series_name,
        records,
        localization_model,
        classifier,
        threshold,
        device,
    )

    output_path = args.output or (
        Path("outputs/herniation_levels/studies")
        / f"{args.series_name}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2) + "\n"
    )

    print(
        json.dumps({
            "series_name": args.series_name,
            "status": result["status"],
            "herniated_levels": result["herniated_levels"],
            "output": str(output_path),
        }, indent=2)
    )


if __name__ == "__main__":
    main()
