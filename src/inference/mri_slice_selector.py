"""Choose a sagittal slice using MRI pixels and the disc localizer only"""

import numpy as np
import torch

from scripts.export_spider_disc_localization import (
    letterbox_pair,
    normalize_slice,
)
from scripts.evaluate_spider_multitask_components import (
    extract_components,
    predict,
)
from src.inference.herniation_levels import (
    level_bands,
    localize_levels,
)

@torch.inference_mode()
def select_sagittal_slice(
    record: dict,
    volume: np.ndarray,
    localizer: torch.nn.Module,
    device: torch.device,
    middle_window: int = 2,
    selection_mode: str = "component",
) -> tuple[int, np.ndarray, dict]:
    """
    volume shape: [height, width, sagittal_slices]

    Returns:
        selected_index
        selected_image: normalized, letterboxed [256, 256] image
        diagnostics: scores for all examined slices
    """

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got {volume.shape}")
    if middle_window < 0:
        raise ValueError("middle_window must be nonnegative")

    if selection_mode not in {"component", "guarded_confidence"}:
        raise ValueError(f"Unknown selection mode: {selection_mode}")

    slice_count = volume.shape[2]
    if slice_count == 0:
        raise ValueError("Volume has no sagittal slices")

    middle = (slice_count - 1) // 2
    first = max(0, middle - middle_window)
    last = min(slice_count, middle + middle_window + 1)

    candidates = []

    for index in range(first, last):
        raw_slice = volume[:, :, index]

        try:
            normalized = normalize_slice(raw_slice)
        except ValueError:
            continue

        # letterbox_pair expect an image and a mask. This all-zero
        # mask is only placeholder; no reference annotation is read
        empty_mask = np.zeros(raw_slice.shape, dtype=np.uint8)
        image, _ = letterbox_pair(normalized, empty_mask, 256)

        binary_probability, _ = predict(localizer, image, device)
        components = extract_components(
            binary_probability >= 0.5,
            minimum_area=25,
        )

        candidates.append({
            "index": index,
            "image": image,
            "component_count": len(components),
            "component_area": sum(item["area"] for item in components),
        })

    if not candidates:
        raise ValueError("No usable MRI slice in the middle window")

    # Keep this first versioon identical to the earllier auditted rule:
    # prefer more detected discs, then greater total disc area.
    ranked = sorted(
        candidates,
        key=lambda item: (
            item["component_count"],
            item["component_area"],
        ),
        reverse=True,
    )

    attempts = []
    evaluated = []

    for candidate in ranked:
        try:
            boxes, decoder = localize_levels(
                record,
                volume,
                localizer,
                device,
                localization_image=candidate["image"],
            )
            level_bands(boxes, volume.shape[0])
        except ValueError as error:
            attempts.append({
                "slice": candidate["index"],
                "accepted": False,
                "reason": str(error),
            })
            continue

        minimum_probability = min(
            (box["level_model_probability"] for box in boxes.values()),
            default=0.0,
        )
        evaluated.append({
            **candidate,
            "boxes": boxes,
            "decoder_source": decoder["source"],
            "minimum_level_probability": minimum_probability,
        })
        attempts.append({
            "slice": candidate["index"],
            "accepted": True,
            "assigned_levels": len(boxes),
            "minimum_level_probability": minimum_probability,
            "decoder_source": decoder["source"],
        })

    if not evaluated:
        raise ValueError("No candidate slice has valid level geometry")

    # Reproduce the existing choice: first complete candidate in the
    # component-ranked list, or first candidate with the most levels.
    complete = [item for item in evaluated if len(item["boxes"]) == 5]
    baseline = (
        complete[0]
        if complete
        else max(evaluated, key=lambda item: len(item["boxes"]))
    )

    confidence = max(
        evaluated,
        key=lambda item: (
            len(item["boxes"]) == 5,
            len(item["boxes"]),
            item["minimum_level_probability"],
            item["component_count"],
            item["component_area"],
        ),
    )

    confidence_gain = (
        confidence["minimum_level_probability"]
        - baseline["minimum_level_probability"]
    )
    decoder_ok = (
        confidence["decoder_source"] == "fused_sequence"
        or baseline["decoder_source"] != "fused_sequence"
    )
    guarded_switch = (
        selection_mode == "guarded_confidence"
        and len(confidence["boxes"]) == 5
        and confidence_gain >= 0.10
        and decoder_ok
    )
    selected = confidence if guarded_switch else baseline

    diagnostics = {
        "method": selection_mode,
        "middle_window": middle_window,
        "selected_slice": selected["index"],
        "assigned_levels": len(selected["boxes"]),
        "baseline_slice": baseline["index"],
        "confidence_slice": confidence["index"],
        "guarded_switch": guarded_switch,
        "confidence_gain": confidence_gain,
        "candidates": [
            {
                "slice": item["index"],
                "components": item["component_count"],
                "area": item["component_area"],
            }
            for item in candidates
        ],
        "attempts": attempts,
    }

    return selected["index"], selected["image"], diagnostics
