import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from scripts.evaluate_spider_multitask_components import (
    extract_components,
    load_model,
    predict,
)
from scripts.run_spider_sagittal_pipeline import (
    load_localization_image,
    load_manifest_record,
    load_original_sagittal_slice,
)
from scripts.build_spider_paired_crops import normalize_slice
from src.models.spider_posterior_model import SpiderPosteriorCNN


def make_context_crop(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    output_size: int = 160,
) -> np.ndarray:
    """Use the geometry from the posterior-crop exporter."""
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1

    if box_width <= 0 or box_height <= 0:
        raise ValueError(f"Invalid disc box: {box}")

    center_x = (x1 + x2) / 2 + 0.35 * box_width
    center_y = (y1 + y2) / 2

    half_side = int(
        np.ceil(max(
            2.0 * box_width,
            2.5 * box_height,
            48,
        ))
    )

    left = round(center_x) - half_side
    top = round(center_y) - half_side
    side = 2 * half_side

    canvas = np.zeros((side, side), dtype=np.float32)
    height, width = image.shape

    source_x1 = max(0, left)
    source_y1 = max(0, top)
    source_x2 = min(width, left + side)
    source_y2 = min(height, top + side)

    if source_x1 < source_x2 and source_y1 < source_y2:
        dest_x1 = source_x1 - left
        dest_y1 = source_y1 - top

        canvas[
            dest_y1:dest_y1 + (source_y2 - source_y1),
            dest_x1:dest_x1 + (source_x2 - source_x1),
        ] = image[
            source_y1:source_y2,
            source_x1:source_x2,
        ]

    return np.asarray(
        Image.fromarray(canvas).resize(
            (output_size, output_size),
            Image.Resampling.BILINEAR,
        ),
        dtype=np.float32,
    )


def undo_letterbox_mask(
    component_pixels: np.ndarray,
    original_shape: tuple[int, int],
) -> np.ndarray:
    """Map a component from the 256×256 localization image to the MRI slice."""
    original_height, original_width = original_shape
    canvas_height, canvas_width = component_pixels.shape

    if canvas_height != canvas_width:
        raise ValueError("Expected a square localization canvas")

    size = canvas_height
    scale = min(
        size / original_width,
        size / original_height,
    )

    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))
    offset_x = (size - resized_width) // 2
    offset_y = (size - resized_height) // 2

    unpadded = component_pixels[
        offset_y:offset_y + resized_height,
        offset_x:offset_x + resized_width,
    ].astype(np.uint8)

    restored = Image.fromarray(unpadded).resize(
        (original_width, original_height),
        Image.Resampling.NEAREST,
    )

    return np.asarray(restored, dtype=np.float32)


def load_posterior_model(
    checkpoint_path: Path,
    device: torch.device,
) -> SpiderPosteriorCNN:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    # The training script saves a checkpoint dictionary.
    state = checkpoint.get("model_state_dict", checkpoint)

    model = SpiderPosteriorCNN().to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series-name", required=True)
    parser.add_argument(
        "--pipeline-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/processed/spider_disc_localization/manifest.csv"
        ),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/spider"),
    )
    parser.add_argument(
        "--localization-checkpoint",
        type=Path,
        default=Path(
            "outputs/spider_disc_multitask_coord_v2/best_model.pt"
        ),
    )
    parser.add_argument(
        "--posterior-checkpoint",
        type=Path,
        default=Path(
            "outputs/spider_posterior_baseline/best_model.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--bulging-threshold",
        type=float,
        default=0.5,
    )
    args = parser.parse_args()

    with args.pipeline_json.open() as file:
        result = json.load(file)

    if result["series_name"] != args.series_name:
        raise ValueError(
            "Pipeline JSON belongs to a different series"
        )

    record = load_manifest_record(
        args.manifest,
        args.series_name,
    )

    localization_image = load_localization_image(record)

    slice_index = int(record["sagittal_slice"])
    if result["sagittal_slice"] != slice_index:
        raise ValueError("Pipeline JSON uses a different sagittal slice")

    original_slice = load_original_sagittal_slice(
        args.raw_root,
        args.series_name,
        slice_index,
    )
    normalized_slice = normalize_slice(original_slice)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    localization_model = load_model(
        args.localization_checkpoint,
        device,
    )

    with torch.inference_mode():
        binary_probability, _ = predict(
            localization_model,
            localization_image,
            device,
        )

    probability_threshold = result["localization"][
        "probability_threshold"
    ]
    minimum_area = result["localization"][
        "minimum_component_area"
    ]

    components = extract_components(
        binary_probability >= probability_threshold,
        minimum_area,
    )
    components_by_id = {
        int(component["component_id"]): component
        for component in components
    }

    model = load_posterior_model(
        args.posterior_checkpoint,
        device,
    )

    available_levels = [
        level
        for level in result["levels"]
        if level["status"] == "available"
    ]

    inputs = []
    for level in available_levels:
        component_id = int(level["component_id"])
        if component_id not in components_by_id:
            raise ValueError(
                f"Component {component_id} could not be reproduced"
            )

        box = tuple(level["bounding_box_original"])
        component = components_by_id[component_id]

        original_mask = undo_letterbox_mask(
            component["pixels"],
            original_slice.shape,
        )

        image_crop = make_context_crop(
            normalized_slice,
            box,
        )
        mask_crop = make_context_crop(
            original_mask,
            box,
        )

        # Training used a binary target-mask channel.
        mask_crop = (mask_crop >= 0.5).astype(np.float32)

        if mask_crop.sum() == 0:
            raise ValueError(
                f"Mask vanished for {level['position']}"
            )

        inputs.append(
            np.stack([image_crop, mask_crop], axis=0)
        )

    if inputs:
        batch = torch.from_numpy(
            np.stack(inputs).astype(np.float32)
        ).to(device)

        with torch.inference_mode():
            logits = model(batch)
            if logits.shape != (len(inputs), 2):
                raise ValueError(
                    f"Unexpected posterior output: {logits.shape}"
                )

            probabilities = torch.sigmoid(logits).cpu().numpy()

        for level, scores in zip(available_levels, probabilities):
            bulging_probability = float(scores[0])

            level["disc_bulging"] = (
                "detected"
                if bulging_probability >= args.bulging_threshold
                else "not_detected"
            )
            level["disc_bulging_probability"] = (
                bulging_probability
            )

            # Keep the unvalidated herniation head out of the
            # study-level finding.
            level["disc_herniation"] = "unavailable"
            level["posterior_disc_finding"] = "unavailable"

    for level in result["levels"]:
        if level["status"] != "available":
            level["disc_bulging"] = "unavailable"
            level["disc_herniation"] = "unavailable"

    output_path = args.output
    if output_path is None:
        output_path = (
            Path("outputs/sagittal_inference")
            / f"{args.series_name}_with_bulging.json"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(result, file, indent=2)

    for level in result["levels"]:
        print(
            level["position"],
            "bulging=",
            level["disc_bulging"],
            "probability=",
            level.get("disc_bulging_probability"),
        )

    print("Saved:", output_path)


if __name__ == "__main__":
    main()