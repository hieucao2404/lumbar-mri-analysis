import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.models.spider_disc_multitask_unet import (
    SpiderDiscMultitaskUNet,
)


LEVEL_NAMES = [
    "L1-L2",
    "L2-L3",
    "L3-L4",
    "L4-L5",
    "L5-S1",
    "Upper",
]

NUMBER_OF_LEVEL_CLASSES = len(LEVEL_NAMES)


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> SpiderDiscMultitaskUNet:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected a structured checkpoint dictionary"
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain model_state_dict"
        )

    model = SpiderDiscMultitaskUNet().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()
    return model


def stored_level_to_model_class(
    stored_value: int,
) -> int:
    """
    Stored level-mask values:
        1   = L1-L2
        2   = L2-L3
        3   = L3-L4
        4   = L4-L5
        5   = L5-S1
        255 = Upper

    Model classes:
        0-4 = L1-L2 through L5-S1
        5   = Upper
    """
    if 1 <= stored_value <= 5:
        return stored_value - 1

    if stored_value == 255:
        return 5

    raise ValueError(
        f"Unsupported stored level: {stored_value}"
    )


def extract_components(
    binary_mask: np.ndarray,
    minimum_area: int,
) -> list[dict]:
    if binary_mask.ndim != 2:
        raise ValueError(
            f"Expected 2D mask, received {binary_mask.shape}"
        )

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary_mask.astype(np.uint8),
            connectivity=8,
        )
    )

    components = []

    for component_id in range(1, count):
        area = int(
            stats[
                component_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < minimum_area:
            continue

        x = int(
            stats[
                component_id,
                cv2.CC_STAT_LEFT,
            ]
        )

        y = int(
            stats[
                component_id,
                cv2.CC_STAT_TOP,
            ]
        )

        width = int(
            stats[
                component_id,
                cv2.CC_STAT_WIDTH,
            ]
        )

        height = int(
            stats[
                component_id,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        pixels = labels == component_id

        components.append({
            "component_id": component_id,
            "area": area,
            "pixels": pixels,
            "x1": x,
            "y1": y,
            "x2": x + width,
            "y2": y + height,
            "center_x": float(
                centroids[component_id, 0]
            ),
            "center_y": float(
                centroids[component_id, 1]
            ),
        })

    return components


def add_reference_levels(
    components: list[dict],
    level_mask: np.ndarray,
) -> list[dict]:
    result = []

    for component in components:
        values = level_mask[
            component["pixels"]
        ]

        valid = values[
            (
                ((values >= 1) & (values <= 5))
                | (values == 255)
            )
        ]

        if valid.size == 0:
            raise ValueError(
                "Reference disc component has no level label"
            )

        unique_values, counts = np.unique(
            valid,
            return_counts=True,
        )

        stored_level = int(
            unique_values[np.argmax(counts)]
        )

        result.append({
            **component,
            "true_level_id": (
                stored_level_to_model_class(
                    stored_level
                )
            ),
            "true_level": LEVEL_NAMES[
                stored_level_to_model_class(
                    stored_level
                )
            ],
        })

    return result


def add_predicted_levels(
    components: list[dict],
    level_probabilities: np.ndarray,
) -> list[dict]:
    """
    Average level probabilities over all pixels in each predicted
    disc component.
    """
    if level_probabilities.shape[0] != 6:
        raise ValueError(
            "Expected six level-probability channels, "
            f"received {level_probabilities.shape}"
        )

    result = []

    for component in components:
        pixels = component["pixels"]

        component_probabilities = (
            level_probabilities[:, pixels].mean(
                axis=1
            )
        )

        predicted_level_id = int(
            np.argmax(component_probabilities)
        )

        confidence = float(
            component_probabilities[
                predicted_level_id
            ]
        )

        result.append({
            **component,
            "predicted_level_id": (
                predicted_level_id
            ),
            "predicted_level": LEVEL_NAMES[
                predicted_level_id
            ],
            "level_confidence": confidence,
            "level_probabilities": (
                component_probabilities.tolist()
            ),
        })

    return result


def mask_iou(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    intersection = np.logical_and(
        first,
        second,
    ).sum()

    union = np.logical_or(
        first,
        second,
    ).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def match_components(
    references: list[dict],
    predictions: list[dict],
    minimum_iou: float,
) -> list[dict]:
    """
    Greedy highest-IoU matching.

    Each reference and prediction can be used at most once.
    """
    candidates = []

    for reference_index, reference in enumerate(
        references
    ):
        for prediction_index, prediction in enumerate(
            predictions
        ):
            iou = mask_iou(
                reference["pixels"],
                prediction["pixels"],
            )

            if iou >= minimum_iou:
                candidates.append({
                    "reference_index": reference_index,
                    "prediction_index": (
                        prediction_index
                    ),
                    "iou": iou,
                })

    candidates.sort(
        key=lambda candidate: candidate["iou"],
        reverse=True,
    )

    used_references = set()
    used_predictions = set()
    matches = []

    for candidate in candidates:
        reference_index = candidate[
            "reference_index"
        ]

        prediction_index = candidate[
            "prediction_index"
        ]

        if reference_index in used_references:
            continue

        if prediction_index in used_predictions:
            continue

        used_references.add(reference_index)
        used_predictions.add(prediction_index)
        matches.append(candidate)

    return matches


def calculate_classification_metrics(
    confusion: np.ndarray,
) -> dict:
    per_class = []
    f1_values = []

    for class_id, class_name in enumerate(
        LEVEL_NAMES
    ):
        true_positive = int(
            confusion[class_id, class_id]
        )

        false_positive = int(
            confusion[:, class_id].sum()
            - true_positive
        )

        false_negative = int(
            confusion[class_id, :].sum()
            - true_positive
        )

        support = int(
            confusion[class_id, :].sum()
        )

        precision_denominator = (
            true_positive + false_positive
        )

        recall_denominator = (
            true_positive + false_negative
        )

        precision = (
            true_positive / precision_denominator
            if precision_denominator > 0
            else 0.0
        )

        recall = (
            true_positive / recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        f1 = (
            2.0 * precision * recall
            / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        per_class.append({
            "class_id": class_id,
            "class_name": class_name,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

        f1_values.append(f1)

    total = int(confusion.sum())
    correct = int(np.trace(confusion))

    return {
        "accuracy": (
            correct / total
            if total > 0
            else 0.0
        ),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def safe_divide(
    numerator: int | float,
    denominator: int | float,
) -> float:
    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


@torch.inference_mode()
def predict(
    model: SpiderDiscMultitaskUNet,
    image: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(
        image.astype(np.float32)
    ).unsqueeze(0).unsqueeze(0).to(device)

    binary_logits, level_logits = model(tensor)

    binary_probability = torch.sigmoid(
        binary_logits
    )[0, 0]

    level_probabilities = torch.softmax(
        level_logits,
        dim=1,
    )[0]

    return (
        binary_probability.cpu().numpy(),
        level_probabilities.cpu().numpy(),
    )

def component_from_pixels(
    pixels: np.ndarray,
    component_id: int,
) -> dict:
    y_coordinates, x_coordinates = np.where(
        pixels
    )

    if x_coordinates.size == 0:
        raise ValueError(
            "Cannot create a component from an empty mask"
        )

    x1 = int(x_coordinates.min())
    y1 = int(y_coordinates.min())
    x2 = int(x_coordinates.max()) + 1
    y2 = int(y_coordinates.max()) + 1

    return {
        "component_id": component_id,
        "area": int(pixels.sum()),
        "pixels": pixels,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "center_x": float(
            x_coordinates.mean()
        ),
        "center_y": float(
            y_coordinates.mean()
        ),
    }


def extract_reference_components(
    level_mask: np.ndarray,
    minimum_area: int,
) -> list[dict]:
    """
    Build reference objects directly from semantic level labels.

    Each lumbar level 1-5 represents exactly one anatomical disc,
    even if its mask contains a small disconnected fragment.

    Upper discs all use value 255, so connected components are still
    required to separate individual upper discs.
    """
    references = []

    # Stored values 1 through 5 correspond to the five lumbar levels.
    for stored_level in range(1, 6):
        pixels = level_mask == stored_level
        area = int(pixels.sum())

        if area == 0:
            continue

        if area < minimum_area:
            continue

        component = component_from_pixels(
            pixels=pixels,
            component_id=stored_level,
        )

        true_level_id = stored_level - 1

        references.append({
            **component,
            "true_level_id": true_level_id,
            "true_level": LEVEL_NAMES[
                true_level_id
            ],
        })

    # Value 255 may contain several separate upper discs.
    upper_mask = (
        level_mask == 255
    ).astype(np.uint8)

    upper_components = extract_components(
        upper_mask,
        minimum_area,
    )

    for upper_index, component in enumerate(
        upper_components,
        start=1,
    ):
        references.append({
            **component,
            "component_id": (
                f"upper_{upper_index}"
            ),
            "true_level_id": 5,
            "true_level": "Upper",
        })

    return references


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/processed/"
            "spider_disc_localization/"
            "manifest.csv"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/"
            "spider_disc_multitask_coord_v2/"
            "best_model.pt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/"
            "spider_disc_multitask_coord_v2/"
            "component_evaluation"
        ),
    )

    parser.add_argument(
        "--subset",
        choices=["training", "validation"],
        default="validation",
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
        "--matching-iou",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--exclude-patient",
        action="append",
        default=[],
        help=(
            "Patient ID to exclude. May be supplied multiple times."
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.manifest.open(newline="") as file:
        all_rows = list(csv.DictReader(file))

    excluded_patients = set(
        args.exclude_patient
    )

    rows = [
        row
        for row in all_rows
        if row["subset"] == args.subset
        and row["patient_id"]
        not in excluded_patients
    ]

    if not rows:
        raise ValueError(
            f"No rows found for subset {args.subset}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(
        args.checkpoint,
        device,
    )

    total_reference_components = 0
    total_predicted_components = 0
    total_matches = 0
    iou_sum = 0.0

    correct_matched_levels = 0

    total_lumbar_references = 0
    correct_end_to_end_lumbar = 0

    level_confusion = np.zeros(
        (
            NUMBER_OF_LEVEL_CLASSES,
            NUMBER_OF_LEVEL_CLASSES,
        ),
        dtype=np.int64,
    )

    level_reference_counts = np.zeros(
        NUMBER_OF_LEVEL_CLASSES,
        dtype=np.int64,
    )

    level_end_to_end_correct = np.zeros(
        NUMBER_OF_LEVEL_CLASSES,
        dtype=np.int64,
    )

    studies_all_annotated_lumbar_correct = 0
    eligible_five_level_studies = 0
    studies_all_five_levels_correct = 0

    output_records = []
    study_reviews = []

    for row in rows:
        with np.load(
            row["example_path"],
            allow_pickle=False,
        ) as data:
            image = data["image"].astype(
                np.float32
            )

            reference_binary_mask = (
                data["mask"].astype(np.uint8)
            )

            level_mask = data[
                "level_mask"
            ].astype(np.int64)

        binary_probability, level_probabilities = (
            predict(
                model,
                image,
                device,
            )
        )

        predicted_binary_mask = (
            binary_probability
            >= args.probability_threshold
        ).astype(np.uint8)

        reference_components = (
            extract_reference_components(
                level_mask=level_mask,
                minimum_area=args.minimum_area,
            )
        )
        predicted_components = extract_components(
            predicted_binary_mask,
            args.minimum_area,
        )

        predicted_components = add_predicted_levels(
            predicted_components,
            level_probabilities,
        )

        matches = match_components(
            reference_components,
            predicted_components,
            args.matching_iou,
        )

        matched_reference_indices = {
            match["reference_index"]
            for match in matches
        }

        matched_prediction_indices = {
            match["prediction_index"]
            for match in matches
        }

        total_reference_components += len(
            reference_components
        )

        total_predicted_components += len(
            predicted_components
        )

        total_matches += len(matches)

        lumbar_reference_indices = [
            index
            for index, reference in enumerate(
                reference_components
            )
            if reference["true_level_id"] < 5
        ]

        total_lumbar_references += len(
            lumbar_reference_indices
        )

        for reference in reference_components:
            level_reference_counts[
                reference["true_level_id"]
            ] += 1

        correct_lumbar_reference_indices = set()

        for match in matches:
            reference_index = match[
                "reference_index"
            ]

            prediction_index = match[
                "prediction_index"
            ]

            reference = reference_components[
                reference_index
            ]

            prediction = predicted_components[
                prediction_index
            ]

            true_level_id = reference[
                "true_level_id"
            ]

            predicted_level_id = prediction[
                "predicted_level_id"
            ]

            correct = (
                true_level_id
                == predicted_level_id
            )

            iou_sum += match["iou"]

            level_confusion[
                true_level_id,
                predicted_level_id,
            ] += 1

            if correct:
                correct_matched_levels += 1

                level_end_to_end_correct[
                    true_level_id
                ] += 1

                if true_level_id < 5:
                    correct_end_to_end_lumbar += 1

                    correct_lumbar_reference_indices.add(
                        reference_index
                    )

            output_records.append({
                "patient_id": row["patient_id"],
                "series_name": row["series_name"],
                "subset": row["subset"],
                "record_type": "matched",
                "reference_component_id": (
                    reference["component_id"]
                ),
                "predicted_component_id": (
                    prediction["component_id"]
                ),
                "true_level": reference[
                    "true_level"
                ],
                "predicted_level": prediction[
                    "predicted_level"
                ],
                "level_confidence": prediction[
                    "level_confidence"
                ],
                "mask_iou": match["iou"],
                "correct_level": int(correct),
                "x1": prediction["x1"],
                "y1": prediction["y1"],
                "x2": prediction["x2"],
                "y2": prediction["y2"],
            })

        for reference_index, reference in enumerate(
            reference_components
        ):
            if (
                reference_index
                in matched_reference_indices
            ):
                continue

            output_records.append({
                "patient_id": row["patient_id"],
                "series_name": row["series_name"],
                "subset": row["subset"],
                "record_type": "missed_reference",
                "reference_component_id": (
                    reference["component_id"]
                ),
                "predicted_component_id": "",
                "true_level": reference[
                    "true_level"
                ],
                "predicted_level": "",
                "level_confidence": "",
                "mask_iou": "",
                "correct_level": 0,
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
            })

        unmatched_lumbar_predictions = 0

        for prediction_index, prediction in enumerate(
            predicted_components
        ):
            if (
                prediction_index
                in matched_prediction_indices
            ):
                continue

            if prediction["predicted_level_id"] < 5:
                unmatched_lumbar_predictions += 1

            output_records.append({
                "patient_id": row["patient_id"],
                "series_name": row["series_name"],
                "subset": row["subset"],
                "record_type": "extra_prediction",
                "reference_component_id": "",
                "predicted_component_id": (
                    prediction["component_id"]
                ),
                "true_level": "",
                "predicted_level": prediction[
                    "predicted_level"
                ],
                "level_confidence": prediction[
                    "level_confidence"
                ],
                "mask_iou": "",
                "correct_level": 0,
                "x1": prediction["x1"],
                "y1": prediction["y1"],
                "x2": prediction["x2"],
                "y2": prediction["y2"],
            })

        all_annotated_lumbar_correct = (
            len(correct_lumbar_reference_indices)
            == len(lumbar_reference_indices)
            and unmatched_lumbar_predictions == 0
        )

        if all_annotated_lumbar_correct:
            studies_all_annotated_lumbar_correct += 1

        lumbar_reference_levels = [
            reference_components[index][
                "true_level_id"
            ]
            for index in lumbar_reference_indices
        ]

        has_exactly_five_levels = (
            sorted(lumbar_reference_levels)
            == [0, 1, 2, 3, 4]
        )

        if has_exactly_five_levels:
            eligible_five_level_studies += 1

            if all_annotated_lumbar_correct:
                studies_all_five_levels_correct += 1

        if not all_annotated_lumbar_correct:
            study_reviews.append({
                "patient_id": row["patient_id"],
                "series_name": row["series_name"],
                "reference_components": len(
                    reference_components
                ),
                "predicted_components": len(
                    predicted_components
                ),
                "matched_components": len(matches),
                "lumbar_references": len(
                    lumbar_reference_indices
                ),
                "correct_lumbar_levels": len(
                    correct_lumbar_reference_indices
                ),
                "extra_lumbar_predictions": (
                    unmatched_lumbar_predictions
                ),
            })

    detection_precision = safe_divide(
        total_matches,
        total_predicted_components,
    )

    detection_recall = safe_divide(
        total_matches,
        total_reference_components,
    )

    detection_f1 = safe_divide(
        2.0
        * detection_precision
        * detection_recall,
        detection_precision + detection_recall,
    )

    matched_level_metrics = (
        calculate_classification_metrics(
            level_confusion
        )
    )

    end_to_end_lumbar_accuracy = safe_divide(
        correct_end_to_end_lumbar,
        total_lumbar_references,
    )

    mean_matched_iou = safe_divide(
        iou_sum,
        total_matches,
    )

    per_level_end_to_end = []

    for class_id, class_name in enumerate(
        LEVEL_NAMES
    ):
        reference_count = int(
            level_reference_counts[class_id]
        )

        correct_count = int(
            level_end_to_end_correct[class_id]
        )

        per_level_end_to_end.append({
            "class_id": class_id,
            "class_name": class_name,
            "reference_count": reference_count,
            "correct_detection_and_level": (
                correct_count
            ),
            "end_to_end_accuracy": safe_divide(
                correct_count,
                reference_count,
            ),
        })

    summary = {
        "checkpoint": str(args.checkpoint),
        "subset": args.subset,
        "images": len(rows),
        "excluded_patients": sorted(
            excluded_patients
        ),
        "probability_threshold": (
            args.probability_threshold
        ),
        "minimum_area": args.minimum_area,
        "matching_iou_threshold": (
            args.matching_iou
        ),
        "reference_components": (
            total_reference_components
        ),
        "predicted_components": (
            total_predicted_components
        ),
        "matched_components": total_matches,
        "detection_precision": (
            detection_precision
        ),
        "detection_recall": detection_recall,
        "detection_f1": detection_f1,
        "mean_matched_mask_iou": (
            mean_matched_iou
        ),
        "matched_component_level_metrics": (
            matched_level_metrics
        ),
        "total_lumbar_references": (
            total_lumbar_references
        ),
        "correct_end_to_end_lumbar": (
            correct_end_to_end_lumbar
        ),
        "end_to_end_lumbar_accuracy": (
            end_to_end_lumbar_accuracy
        ),
        "per_level_end_to_end": (
            per_level_end_to_end
        ),
        "studies_all_annotated_lumbar_correct": (
            studies_all_annotated_lumbar_correct
        ),
        "eligible_five_level_studies": (
            eligible_five_level_studies
        ),
        "studies_all_five_levels_correct": (
            studies_all_five_levels_correct
        ),
    }

    summary_path = (
        args.output_dir / "summary.json"
    )

    with summary_path.open("w") as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    records_path = (
        args.output_dir / "component_predictions.csv"
    )

    with records_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "patient_id",
                "series_name",
                "subset",
                "record_type",
                "reference_component_id",
                "predicted_component_id",
                "true_level",
                "predicted_level",
                "level_confidence",
                "mask_iou",
                "correct_level",
                "x1",
                "y1",
                "x2",
                "y2",
            ],
        )

        writer.writeheader()
        writer.writerows(output_records)

    print("Checkpoint:", args.checkpoint)
    print("Device:", device)
    print("Subset:", args.subset)
    print("Images:", len(rows))
    print(
        "Excluded patients:",
        sorted(excluded_patients),
    )

    print()
    print("Disc-component localization")
    print(
        "  Reference components:",
        total_reference_components,
    )
    print(
        "  Predicted components:",
        total_predicted_components,
    )
    print(
        "  Matched components:",
        total_matches,
    )
    print(
        "  Precision:",
        round(detection_precision, 4),
    )
    print(
        "  Recall:",
        round(detection_recall, 4),
    )
    print(
        "  F1:",
        round(detection_f1, 4),
    )
    print(
        "  Mean matched IoU:",
        round(mean_matched_iou, 4),
    )

    print()
    print("Level classification among matched discs")
    print(
        "  Accuracy:",
        round(
            matched_level_metrics["accuracy"],
            4,
        ),
    )
    print(
        "  Macro F1:",
        round(
            matched_level_metrics["macro_f1"],
            4,
        ),
    )

    for result in matched_level_metrics[
        "per_class"
    ]:
        print(
            f"  {result['class_name']:<6} "
            f"precision={result['precision']:.3f} "
            f"recall={result['recall']:.3f} "
            f"f1={result['f1']:.3f} "
            f"support={result['support']}"
        )

    print()
    print(
        "Level confusion matrix "
        "[true row, predicted column]"
    )
    print("Order:", LEVEL_NAMES)

    for row in level_confusion:
        print(" ", row.tolist())

    print()
    print("End-to-end lumbar-level result")
    print(
        "  Lumbar references:",
        total_lumbar_references,
    )
    print(
        "  Correct detection and level:",
        correct_end_to_end_lumbar,
    )
    print(
        "  End-to-end lumbar accuracy:",
        round(end_to_end_lumbar_accuracy, 4),
    )

    print()
    print("Per-level end-to-end accuracy")

    for result in per_level_end_to_end:
        print(
            f"  {result['class_name']:<6} "
            f"{result['correct_detection_and_level']}/"
            f"{result['reference_count']} "
            f"accuracy="
            f"{result['end_to_end_accuracy']:.4f}"
        )

    print()
    print("Complete-study results")
    print(
        "  All annotated lumbar levels correct:",
        f"{studies_all_annotated_lumbar_correct}/"
        f"{len(rows)}",
    )
    print(
        "  Studies with all five reference levels:",
        eligible_five_level_studies,
    )
    print(
        "  All five levels correct:",
        f"{studies_all_five_levels_correct}/"
        f"{eligible_five_level_studies}",
    )

    print()
    print("First studies requiring review:")

    for review in study_reviews[:12]:
        print(" ", review)

    print()
    print("Summary:", summary_path)
    print("Predictions:", records_path)


if __name__ == "__main__":
    main()