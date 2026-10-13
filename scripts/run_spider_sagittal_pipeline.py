import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from scripts.build_spider_paired_crops import (
    make_crop,
    normalize_slice,
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

from src.models.spider_model import SpiderPfirrmannCNN


LEVEL_NAMES = (
    "L1-L2",
    "L2-L3",
    "L3-L4",
    "L4-L5",
    "L5-S1",
)

PFIRRMANN_NAMES = (
    "Pfirrmann I",
    "Pfirrmann II",
    "Pfirrmann III",
    "Pfirrmann IV",
    "Pfirrmann V",
)


def load_manifest_record(
    manifest_path: Path,
    series_name: str,
) -> dict:
    """
    Find one SPIDER localization record by series name.
    """

    with manifest_path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    matches = [
        row
        for row in rows
        if row["series_name"] == series_name
    ]

    if not matches:
        raise ValueError(
            f"Series {series_name!r} was not found in "
            f"{manifest_path}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Series {series_name!r} appears more than once in "
            f"{manifest_path}"
        )

    return matches[0]


def load_localization_image(
    record: dict,
) -> np.ndarray:
    """
    Load the already-preprocessed 256x256 image used by the
    multitask localization model.
    """

    example_path = Path(record["example_path"])

    if not example_path.is_file():
        raise FileNotFoundError(
            f"Localization example does not exist: {example_path}"
        )

    with np.load(example_path) as data:
        if "image" not in data.files:
            raise ValueError(
                f"{example_path} does not contain an image array"
            )

        image = data["image"].astype(np.float32)

    if image.shape != (256, 256):
        raise ValueError(
            f"Expected localization image shape (256, 256), "
            f"got {image.shape}"
        )

    if not np.isfinite(image).all():
        raise ValueError(
            f"Localization image contains non-finite values: "
            f"{example_path}"
        )

    return image


def load_original_sagittal_slice(
    raw_root: Path,
    series_name: str,
    slice_index: int,
) -> np.ndarray:
    """
    Load the original-resolution sagittal slice using the same LPS
    orientation and array flipping used during dataset export.
    """

    image_path = (
        raw_root
        / "images"
        / f"{series_name}.mha"
    )

    if not image_path.is_file():
        raise FileNotFoundError(
            f"SPIDER image does not exist: {image_path}"
        )

    image_sitk = sitk.DICOMOrient(
        sitk.ReadImage(str(image_path)),
        "LPS",
    )

    volume = np.flip(
        sitk.GetArrayFromImage(image_sitk),
        axis=0,
    )

    if volume.ndim != 3:
        raise ValueError(
            f"Expected a three-dimensional MRI volume, "
            f"got shape {volume.shape}"
        )

    if slice_index < 0 or slice_index >= volume.shape[2]:
        raise IndexError(
            f"Sagittal slice {slice_index} is outside axis 2 "
            f"with size {volume.shape[2]}"
        )

    image_slice = np.take(
        volume,
        slice_index,
        axis=2,
    )

    if image_slice.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional sagittal slice, "
            f"got shape {image_slice.shape}"
        )

    return image_slice.astype(np.float32)


def load_pfirrmann_model(
    checkpoint_path: Path,
    device: torch.device,
) -> SpiderPfirrmannCNN:
    """
    Load the trained Pfirrmann CNN.

    This supports both:
    - a plain PyTorch state dictionary;
    - a checkpoint containing model_state_dict.
    """

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Pfirrmann checkpoint does not exist: "
            f"{checkpoint_path}"
        )

    # Only load checkpoints that you created and trust.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = SpiderPfirrmannCNN().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model


@torch.inference_mode()
def classify_pfirrmann_crops(
    model: SpiderPfirrmannCNN,
    indexed_crops: list[tuple[int, np.ndarray]],
    device: torch.device,
) -> dict[int, dict]:
    """
    Classify a collection of 160x160 disc crops.

    The returned dictionary is indexed by the corresponding component
    index from the localization result.
    """

    if not indexed_crops:
        return {}

    component_indices = [
        component_index
        for component_index, _ in indexed_crops
    ]

    crops = np.stack([
        crop
        for _, crop in indexed_crops
    ]).astype(np.float32)

    if crops.ndim != 3 or crops.shape[1:] != (160, 160):
        raise ValueError(
            f"Expected crop batch [N, 160, 160], "
            f"got {crops.shape}"
        )

    inputs = torch.from_numpy(
        crops[:, None, :, :]
    ).to(device)

    scores = model(inputs)

    if scores.shape != (len(indexed_crops), 5):
        raise ValueError(
            f"Expected Pfirrmann scores [N, 5], "
            f"got {tuple(scores.shape)}"
        )

    probabilities = torch.softmax(
        scores,
        dim=1,
    )

    confidences, predicted_ids = probabilities.max(dim=1)

    probabilities_numpy = (
        probabilities.cpu().numpy()
    )

    predicted_ids_numpy = (
        predicted_ids.cpu().numpy()
    )

    confidences_numpy = (
        confidences.cpu().numpy()
    )

    results = {}

    for batch_index, component_index in enumerate(
        component_indices
    ):
        predicted_id = int(
            predicted_ids_numpy[batch_index]
        )

        results[component_index] = {
            "grade_id": predicted_id + 1,
            "grade": PFIRRMANN_NAMES[predicted_id],
            "confidence": float(
                confidences_numpy[batch_index]
            ),
            "probabilities": [
                float(value)
                for value in probabilities_numpy[batch_index]
            ],
        }

    return results


def build_pipeline_result(
    record: dict,
    components: list[dict],
    hybrid_assignments: list[int],
    hybrid_source: str,
    decoder_result: dict,
    original_boxes: dict[int, tuple[int, int, int, int]],
    classification_results: dict[int, dict],
    probability_threshold: float,
    minimum_area: int,
) -> dict:
    """
    Create the final machine-readable sagittal response.

    Every lumbar level is included. Missing levels are explicitly
    marked unavailable.
    """

    results_by_level = {}

    for component_index, level_id in enumerate(
        hybrid_assignments
    ):
        if level_id not in range(5):
            continue

        component = components[component_index]

        if component_index not in classification_results:
            continue

        classification = classification_results[
            component_index
        ]

        original_box = original_boxes[
            component_index
        ]

        level_probability = float(
            component["level_probabilities"][level_id]
        )

        results_by_level[level_id] = {
            "position": LEVEL_NAMES[level_id],
            "status": "available",
            "level_assignment_method": hybrid_source,
            "level_model_probability": level_probability,
            "bounding_box_original": [
                int(value)
                for value in original_box
            ],
            "component_id": int(
                component["component_id"]
            ),
            "component_area_256": int(
                component["area"]
            ),
            "disc_degeneration": (
                classification["grade"]
            ),
            "disc_degeneration_confidence": (
                classification["confidence"]
            ),
            "disc_degeneration_probabilities": {
                PFIRRMANN_NAMES[index]: float(probability)
                for index, probability in enumerate(
                    classification["probabilities"]
                )
            },

            # These outputs do not yet have trained models.
            "spinal_canal_stenosis": "unavailable",
            "posterior_disc_finding": "unavailable",
        }

    level_outputs = []

    for level_id, level_name in enumerate(LEVEL_NAMES):
        if level_id in results_by_level:
            level_outputs.append(
                results_by_level[level_id]
            )
        else:
            level_outputs.append({
                "position": level_name,
                "status": "unavailable",
                "reason": (
                    "No valid detected component was assigned "
                    "to this lumbar level"
                ),
                "disc_degeneration": "unavailable",
                "spinal_canal_stenosis": "unavailable",
                "posterior_disc_finding": "unavailable",
            })

    available_count = sum(
        level["status"] == "available"
        for level in level_outputs
    )

    reasons = list(
        decoder_result.get("reasons", [])
    )

    if available_count < 5:
        reasons.append(
            f"Only {available_count} of five lumbar levels "
            "were available"
        )

    overall_status = (
        "ok"
        if available_count == 5
        and decoder_result.get("status") == "ok"
        else "uncertain"
    )

    return {
        "study_id": record["patient_id"],
        "series_name": record["series_name"],
        "view": "sagittal",
        "sequence": "T2",
        "status": overall_status,
        "reasons": reasons,
        "sagittal_slice": int(
            record["sagittal_slice"]
        ),
        "classification_slice_policy": (
            "localization_selected_slice"
        ),
        "level_decoder": {
            "method": "gated_hybrid",
            "selected_source": hybrid_source,
            "decoder_status": decoder_result.get(
                "status",
                "uncertain",
            ),
            "spacing_cv": (
                float(decoder_result["spacing_cv"])
                if "spacing_cv" in decoder_result
                else None
            ),
        },
        "localization": {
            "probability_threshold": (
                probability_threshold
            ),
            "minimum_component_area": minimum_area,
            "detected_components": len(components),
        },
        "levels": level_outputs,
        "disclaimer": (
            "Research prototype output. This result is not a "
            "confirmed clinical diagnosis."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the integrated SPIDER sagittal lumbar pipeline"
        )
    )

    parser.add_argument(
        "--series-name",
        required=True,
        help="SPIDER series name, for example 101_t2",
    )

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/spider"),
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/processed/spider_disc_localization/"
            "manifest.csv"
        ),
    )

    parser.add_argument(
        "--localization-checkpoint",
        type=Path,
        default=Path(
            "outputs/spider_disc_multitask_coord_v2/"
            "best_model.pt"
        ),
    )

    parser.add_argument(
        "--pfirrmann-checkpoint",
        type=Path,
        default=Path(
            "outputs/spider_pfirrmann_repro_v1/"
            "best_model.pt"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--probability-threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--minimum-area",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--learned-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--position-weight",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--upper-weight",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--maximum-spacing-cv",
        type=float,
        default=0.55,
    )

    parser.add_argument(
        "--hybrid-spacing-cv",
        type=float,
        default=0.20,
    )

    args = parser.parse_args()

    if not 0.0 < args.probability_threshold < 1.0:
        raise ValueError(
            "Probability threshold must be between zero and one"
        )

    if args.minimum_area <= 0:
        raise ValueError(
            "Minimum component area must be positive"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if args.output is None:
        args.output = (
            Path("outputs/sagittal_inference")
            / f"{args.series_name}.json"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Device:", device)
    print("Series:", args.series_name)

    # ------------------------------------------------------------
    # 1. Load the study information and preprocessed 256x256 image.
    # ------------------------------------------------------------

    record = load_manifest_record(
        args.manifest,
        args.series_name,
    )

    localization_image = load_localization_image(
        record
    )

    slice_index = int(
        record["sagittal_slice"]
    )

    original_slice = load_original_sagittal_slice(
        args.raw_root,
        args.series_name,
        slice_index,
    )

    print(
        "Localization image:",
        localization_image.shape,
    )

    print(
        "Original slice:",
        original_slice.shape,
    )

    print(
        "Sagittal slice index:",
        slice_index,
    )

    # ------------------------------------------------------------
    # 2. Run the multitask localization and level model.
    # ------------------------------------------------------------

    localization_model = load_model(
        args.localization_checkpoint,
        device,
    )

    binary_probability, level_probabilities = predict(
        localization_model,
        localization_image,
        device,
    )

    binary_mask = (
        binary_probability
        >= args.probability_threshold
    )

    components = extract_components(
        binary_mask,
        args.minimum_area,
    )

    components = add_predicted_levels(
        components,
        level_probabilities,
    )

    print(
        "Detected disc components:",
        len(components),
    )

    # ------------------------------------------------------------
    # 3. Apply the final gated-hybrid level decoder.
    # ------------------------------------------------------------

    positional = position_assignments(
        components
    )

    fused, decoder_result = (
        fused_sequence_assignments(
            components,
            learned_weight=args.learned_weight,
            position_weight=args.position_weight,
            upper_weight=args.upper_weight,
            maximum_spacing_cv=(
                args.maximum_spacing_cv
            ),
        )
    )

    hybrid, hybrid_source = (
        gated_hybrid_assignments(
            position=positional,
            fused=fused,
            decoder_result=decoder_result,
            spacing_cv_threshold=(
                args.hybrid_spacing_cv
            ),
        )
    )

    print(
        "Decoder status:",
        decoder_result["status"],
    )

    print(
        "Hybrid source:",
        hybrid_source,
    )

    print(
        "Spacing CV:",
        decoder_result.get("spacing_cv"),
    )

    # ------------------------------------------------------------
    # 4. Map selected component boxes back to original coordinates
    #    and reproduce the Pfirrmann training crop format.
    # ------------------------------------------------------------

    normalized_original = normalize_slice(
        original_slice
    )

    original_height, original_width = (
        original_slice.shape
    )

    canvas_size = localization_image.shape[0]

    indexed_crops = []
    original_boxes = {}

    for component_index, level_id in enumerate(hybrid):
        # -1 or class 5 means this component was not assigned to one
        # of the five lumbar levels.
        if level_id not in range(5):
            continue

        component = components[
            component_index
        ]

        original_box = predicted_box_to_original(
            component,
            original_height,
            original_width,
            canvas_size,
        )

        crop = make_crop(
            normalized_original,
            original_box,
            output_size=160,
        )

        if crop.shape != (160, 160):
            raise ValueError(
                f"Unexpected Pfirrmann crop shape: "
                f"{crop.shape}"
            )

        original_boxes[
            component_index
        ] = original_box

        indexed_crops.append(
            (component_index, crop)
        )

    print(
        "Lumbar crops produced:",
        len(indexed_crops),
    )

    # ------------------------------------------------------------
    # 5. Classify Pfirrmann grades.
    # ------------------------------------------------------------

    pfirrmann_model = load_pfirrmann_model(
        args.pfirrmann_checkpoint,
        device,
    )

    classification_results = (
        classify_pfirrmann_crops(
            pfirrmann_model,
            indexed_crops,
            device,
        )
    )

    # ------------------------------------------------------------
    # 6. Create the final structured response.
    # ------------------------------------------------------------

    result = build_pipeline_result(
        record=record,
        components=components,
        hybrid_assignments=hybrid,
        hybrid_source=hybrid_source,
        decoder_result=decoder_result,
        original_boxes=original_boxes,
        classification_results=classification_results,
        probability_threshold=(
            args.probability_threshold
        ),
        minimum_area=args.minimum_area,
    )

    with args.output.open("w") as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    print()
    print("Final sagittal result")

    for level in result["levels"]:
        print(
            f"  {level['position']}: "
            f"{level['disc_degeneration']}"
        )

    print()
    print("Status:", result["status"])
    print("Saved:", args.output)


if __name__ == "__main__":
    main()