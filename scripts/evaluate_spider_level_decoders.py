import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_spider_multitask_components import (
    LEVEL_NAMES,
    NUMBER_OF_LEVEL_CLASSES,
    add_predicted_levels,
    calculate_classification_metrics,
    extract_components,
    extract_reference_components,
    load_model,
    match_components,
    predict,
    safe_divide,
)
from src.postprocess.disc_level_sequence import (
    decode_lumbar_sequence,
)


METHOD_NAMES = (
    "independent",
    "position",
    "fused_sequence",
    "gated_hybrid"
)


def independent_assignments(
    components: list[dict],
) -> list[int]:
    """Use the coordinate-aware CNN argmax directly."""

    return [
        int(component["predicted_level_id"])
        for component in components
    ]


def position_assignments(
    components: list[dict],
) -> list[int]:
    """
    Assign the five lowest components to L1-L2 through L5-S1.

    Components above the selected five are classified as Upper.
    If fewer than five components exist, no complete lumbar sequence
    can be assigned, so all components are marked Upper.
    """

    assignments = [5] * len(components)

    if len(components) < 5:
        return assignments

    ordered_indices = sorted(
        range(len(components)),
        key=lambda index: components[index]["center_y"],
    )

    lumbar_indices = ordered_indices[-5:]

    for level_id, component_index in enumerate(
        lumbar_indices
    ):
        assignments[component_index] = level_id

    return assignments


def fused_sequence_assignments(
    components: list[dict],
    learned_weight: float,
    position_weight: float,
    upper_weight: float,
    maximum_spacing_cv: float,
) -> tuple[list[int], dict]:
    """
    Combine CNN probabilities, vertical position, and anatomical order.
    """

    assignments = [5] * len(components)

    decoder_result = decode_lumbar_sequence(
        components,
        learned_weight=learned_weight,
        position_weight=position_weight,
        upper_weight=upper_weight,
        maximum_spacing_cv=maximum_spacing_cv,
    )

    for assignment in decoder_result["assignments"]:
        component_index = int(
            assignment["original_index"]
        )

        assignments[component_index] = int(
            assignment["level_id"]
        )

    return assignments, decoder_result

def gated_hybrid_assignments(
    position: list[int],
    fused: list[int],
    decoder_result: dict,
    spacing_cv_threshold: float,
) -> tuple[list[int], str]:
    """
    Use fusion only when it produces a regularly spaced lumbar
    sequence. Otherwise, retain the stable position baseline.

    The returned source is included for evaluation and debugging.
    """

    spacing_cv = decoder_result.get(
        "spacing_cv",
        float("inf"),
    )

    use_fusion = (
        decoder_result["status"] == "ok"
        and spacing_cv <= spacing_cv_threshold
    )

    if use_fusion:
        return fused.copy(), "fused_sequence"

    return position.copy(), "position"


def create_method_metrics() -> dict:
    return {
        "confusion": np.zeros(
            (
                NUMBER_OF_LEVEL_CLASSES,
                NUMBER_OF_LEVEL_CLASSES,
            ),
            dtype=np.int64,
        ),
        "lumbar_references": 0,
        "correct_end_to_end_lumbar": 0,
        "level_reference_counts": np.zeros(
            NUMBER_OF_LEVEL_CLASSES,
            dtype=np.int64,
        ),
        "level_end_to_end_correct": np.zeros(
            NUMBER_OF_LEVEL_CLASSES,
            dtype=np.int64,
        ),
        "all_annotated_correct": 0,
        "eligible_five_level_studies": 0,
        "all_five_correct": 0,
    }


def evaluate_method_on_study(
    metrics: dict,
    reference_components: list[dict],
    predicted_components: list[dict],
    matches: list[dict],
    assignments: list[int],
) -> dict:
    if len(assignments) != len(predicted_components):
        raise ValueError(
            "Assignment count does not match component count"
        )

    matched_prediction_indices = {
        match["prediction_index"]
        for match in matches
    }

    lumbar_reference_indices = [
        index
        for index, reference in enumerate(
            reference_components
        )
        if reference["true_level_id"] < 5
    ]

    metrics["lumbar_references"] += len(
        lumbar_reference_indices
    )

    for reference in reference_components:
        true_level_id = int(
            reference["true_level_id"]
        )

        metrics["level_reference_counts"][
            true_level_id
        ] += 1

    correct_lumbar_reference_indices = set()

    matched_records = []

    for match in matches:
        reference_index = int(
            match["reference_index"]
        )

        prediction_index = int(
            match["prediction_index"]
        )

        reference = reference_components[
            reference_index
        ]

        prediction = predicted_components[
            prediction_index
        ]

        true_level_id = int(
            reference["true_level_id"]
        )

        predicted_level_id = int(
            assignments[prediction_index]
        )

        correct = true_level_id == predicted_level_id

        metrics["confusion"][
            true_level_id,
            predicted_level_id,
        ] += 1

        if correct:
            metrics["level_end_to_end_correct"][
                true_level_id
            ] += 1

            if true_level_id < 5:
                metrics[
                    "correct_end_to_end_lumbar"
                ] += 1

                correct_lumbar_reference_indices.add(
                    reference_index
                )

        probabilities = prediction[
            "level_probabilities"
        ]

        matched_records.append({
            "reference_index": reference_index,
            "prediction_index": prediction_index,
            "reference_component_id": reference[
                "component_id"
            ],
            "predicted_component_id": prediction[
                "component_id"
            ],
            "true_level_id": true_level_id,
            "predicted_level_id": predicted_level_id,
            "true_level": LEVEL_NAMES[true_level_id],
            "predicted_level": LEVEL_NAMES[
                predicted_level_id
            ],
            "confidence": float(
                probabilities[predicted_level_id]
            ),
            "mask_iou": float(match["iou"]),
            "correct": int(correct),
        })

    unmatched_lumbar_predictions = 0

    for prediction_index, predicted_level_id in enumerate(
        assignments
    ):
        if prediction_index in matched_prediction_indices:
            continue

        if predicted_level_id < 5:
            unmatched_lumbar_predictions += 1

    all_annotated_correct = (
        len(correct_lumbar_reference_indices)
        == len(lumbar_reference_indices)
        and unmatched_lumbar_predictions == 0
    )

    if all_annotated_correct:
        metrics["all_annotated_correct"] += 1

    lumbar_reference_levels = sorted(
        reference_components[index][
            "true_level_id"
        ]
        for index in lumbar_reference_indices
    )

    has_all_five_levels = (
        lumbar_reference_levels == [0, 1, 2, 3, 4]
    )

    if has_all_five_levels:
        metrics["eligible_five_level_studies"] += 1

        if all_annotated_correct:
            metrics["all_five_correct"] += 1

    return {
        "correct_lumbar_levels": len(
            correct_lumbar_reference_indices
        ),
        "lumbar_references": len(
            lumbar_reference_indices
        ),
        "unmatched_lumbar_predictions": (
            unmatched_lumbar_predictions
        ),
        "all_annotated_correct": (
            all_annotated_correct
        ),
        "has_all_five_levels": has_all_five_levels,
        "matched_records": matched_records,
    }


def finalize_method_metrics(
    metrics: dict,
    image_count: int,
) -> dict:
    classification = calculate_classification_metrics(
        metrics["confusion"]
    )

    per_level_end_to_end = []

    for class_id, class_name in enumerate(
        LEVEL_NAMES
    ):
        reference_count = int(
            metrics["level_reference_counts"][
                class_id
            ]
        )

        correct_count = int(
            metrics["level_end_to_end_correct"][
                class_id
            ]
        )

        per_level_end_to_end.append({
            "class_id": class_id,
            "class_name": class_name,
            "reference_count": reference_count,
            "correct_detection_and_level": correct_count,
            "end_to_end_accuracy": safe_divide(
                correct_count,
                reference_count,
            ),
        })

    return {
        "matched_component_level_metrics": (
            classification
        ),
        "lumbar_references": int(
            metrics["lumbar_references"]
        ),
        "correct_end_to_end_lumbar": int(
            metrics["correct_end_to_end_lumbar"]
        ),
        "end_to_end_lumbar_accuracy": safe_divide(
            metrics["correct_end_to_end_lumbar"],
            metrics["lumbar_references"],
        ),
        "per_level_end_to_end": (
            per_level_end_to_end
        ),
        "all_annotated_lumbar_correct": int(
            metrics["all_annotated_correct"]
        ),
        "images": image_count,
        "complete_study_accuracy": safe_divide(
            metrics["all_annotated_correct"],
            image_count,
        ),
        "eligible_five_level_studies": int(
            metrics["eligible_five_level_studies"]
        ),
        "all_five_levels_correct": int(
            metrics["all_five_correct"]
        ),
        "all_five_level_accuracy": safe_divide(
            metrics["all_five_correct"],
            metrics["eligible_five_level_studies"],
        ),
    }


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
            "level_decoder_comparison"
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
        help=(
            "Use fused assignments only when their spacing CV "
            "is at or below this threshold"
        ),
    )

    parser.add_argument(
        "--exclude-patient",
        action="append",
        default=[],
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
        and row["patient_id"] not in excluded_patients
    ]

    if not rows:
        raise ValueError(
            f"No samples found for subset {args.subset}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(
        args.checkpoint,
        device,
    )

    method_metrics = {
        method: create_method_metrics()
        for method in METHOD_NAMES
    }

    total_reference_components = 0
    total_predicted_components = 0
    total_matches = 0
    iou_sum = 0.0

    decoder_status_counts = Counter()
    hybrid_source_counts = Counter()

    output_records = []
    study_records = []

    for row in rows:
        with np.load(
            row["example_path"],
            allow_pickle=False,
        ) as data:
            image = data["image"].astype(
                np.float32
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

        total_reference_components += len(
            reference_components
        )

        total_predicted_components += len(
            predicted_components
        )

        total_matches += len(matches)

        iou_sum += sum(
            float(match["iou"])
            for match in matches
        )

        independent = independent_assignments(
            predicted_components
        )

        position = position_assignments(
            predicted_components
        )

        fused, decoder_result = (
            fused_sequence_assignments(
                predicted_components,
                learned_weight=args.learned_weight,
                position_weight=args.position_weight,
                upper_weight=args.upper_weight,
                maximum_spacing_cv=(
                    args.maximum_spacing_cv
                ),
            )
        )

        hybrid, hybrid_source = gated_hybrid_assignments(
            position=position,
            fused=fused,
            decoder_result=decoder_result,
            spacing_cv_threshold=args.hybrid_spacing_cv,
        )

        hybrid_source_counts[hybrid_source] += 1

        assignments_by_method = {
            "independent": independent,
            "position": position,
            "fused_sequence": fused,
            "gated_hybrid": hybrid,
        }

        decoder_status_counts[
            decoder_result["status"]
        ] += 1

        study_record = {
            "patient_id": row["patient_id"],
            "series_name": row["series_name"],
            "predicted_components": len(
                predicted_components
            ),
            "reference_components": len(
                reference_components
            ),
            "matched_components": len(matches),
            "decoder_status": decoder_result[
                "status"
            ],
            "decoder_reasons": " | ".join(
                decoder_result["reasons"]
            ),
            "decoder_spacing_cv": (
                decoder_result.get(
                    "spacing_cv",
                    "",
                )
            ),
            "hybrid_source": hybrid_source,
        }

        for method in METHOD_NAMES:
            study_result = evaluate_method_on_study(
                method_metrics[method],
                reference_components,
                predicted_components,
                matches,
                assignments_by_method[method],
            )

            study_record[
                f"{method}_correct_lumbar"
            ] = study_result[
                "correct_lumbar_levels"
            ]

            study_record[
                f"{method}_lumbar_references"
            ] = study_result[
                "lumbar_references"
            ]

            study_record[
                f"{method}_all_correct"
            ] = int(
                study_result[
                    "all_annotated_correct"
                ]
            )

            for match_record in study_result[
                "matched_records"
            ]:
                output_records.append({
                    "patient_id": row["patient_id"],
                    "series_name": row["series_name"],
                    "subset": row["subset"],
                    "method": method,
                    **match_record,
                })

        study_records.append(study_record)

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

    summary = {
        "checkpoint": str(args.checkpoint),
        "subset": args.subset,
        "images": len(rows),
        "excluded_patients": sorted(
            excluded_patients
        ),
        "configuration": {
            "probability_threshold": (
                args.probability_threshold
            ),
            "minimum_area": args.minimum_area,
            "matching_iou": args.matching_iou,
            "learned_weight": args.learned_weight,
            "position_weight": args.position_weight,
            "upper_weight": args.upper_weight,
            "maximum_spacing_cv": (
                args.maximum_spacing_cv
            ),
            "hybrid_spacing_cv": args.hybrid_spacing_cv,
        },
        "localization": {
            "reference_components": (
                total_reference_components
            ),
            "predicted_components": (
                total_predicted_components
            ),
            "matched_components": total_matches,
            "precision": detection_precision,
            "recall": detection_recall,
            "f1": detection_f1,
            "mean_matched_iou": safe_divide(
                iou_sum,
                total_matches,
            ),
        },
        "decoder_status_counts": dict(
            decoder_status_counts
        ),
        "hybrid_source_counts": dict(
            hybrid_source_counts
        ),
        "methods": {
            method: finalize_method_metrics(
                method_metrics[method],
                len(rows),
            )
            for method in METHOD_NAMES
        },
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

    predictions_path = (
        args.output_dir / "matched_predictions.csv"
    )

    with predictions_path.open(
        "w",
        newline="",
    ) as file:
        fieldnames = [
            "patient_id",
            "series_name",
            "subset",
            "method",
            "reference_index",
            "prediction_index",
            "reference_component_id",
            "predicted_component_id",
            "true_level_id",
            "predicted_level_id",
            "true_level",
            "predicted_level",
            "confidence",
            "mask_iou",
            "correct",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(output_records)

    studies_path = (
        args.output_dir / "study_comparison.csv"
    )

    with studies_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                study_records[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(study_records)

    print("Checkpoint:", args.checkpoint)
    print("Device:", device)
    print("Subset:", args.subset)
    print("Images:", len(rows))
    print(
        "Excluded patients:",
        sorted(excluded_patients),
    )

    print()
    print("Disc localization")
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
        round(
            safe_divide(iou_sum, total_matches),
            4,
        ),
    )

    print()
    print(
        "Sequence-decoder status:",
        dict(decoder_status_counts),
    )
    print(
        "Hybrid source counts:",
        dict(hybrid_source_counts),
    )

    print()
    print("Level-decoder comparison")

    for method in METHOD_NAMES:
        result = summary["methods"][method]
        matched = result[
            "matched_component_level_metrics"
        ]

        print()
        print(method)
        print(
            "  Matched level accuracy:",
            round(matched["accuracy"], 4),
        )
        print(
            "  Matched level macro F1:",
            round(matched["macro_f1"], 4),
        )
        print(
            "  End-to-end lumbar accuracy:",
            round(
                result[
                    "end_to_end_lumbar_accuracy"
                ],
                4,
            ),
        )
        print(
            "  All annotated levels correct:",
            f"{result['all_annotated_lumbar_correct']}/"
            f"{result['images']}",
        )
        print(
            "  All five levels correct:",
            f"{result['all_five_levels_correct']}/"
            f"{result['eligible_five_level_studies']}",
        )

    print()
    print("Summary:", summary_path)
    print("Predictions:", predictions_path)
    print("Studies:", studies_path)


if __name__ == "__main__":
    main()