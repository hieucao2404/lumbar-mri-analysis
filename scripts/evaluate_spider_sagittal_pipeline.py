import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
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
    extract_reference_components,
    load_model,
    match_components,
    predict,
)
from scripts.run_spider_sagittal_pipeline import (
    classify_pfirrmann_crops,
    load_localization_image,
    load_original_sagittal_slice,
    load_pfirrmann_model,
)


LEVEL_NAMES = (
    "L1-L2",
    "L2-L3",
    "L3-L4",
    "L4-L5",
    "L5-S1",
)


# SPIDER numbers discs from inferior to superior.
# IVD 1 = L5-S1; IVD 5 = L1-L2.
LEVEL_TO_IVD = {
    0: 5,
    1: 4,
    2: 3,
    3: 2,
    4: 1,
}


def divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def read_validation_records(
    manifest_path: Path,
    excluded_patients: set[str],
) -> list[dict]:
    with manifest_path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    records = [
        row
        for row in rows
        if row["subset"] == "validation"
        and row["patient_id"] not in excluded_patients
    ]

    if not records:
        raise ValueError("No validation records were selected")

    names = [row["series_name"] for row in records]

    if len(names) != len(set(names)):
        raise ValueError("Duplicate series names in manifest")

    return sorted(records, key=lambda row: row["series_name"])


def read_grades(path: Path) -> dict[tuple[str, int], int]:
    grades = {}

    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            patient_id = str(int(row["Patient"]))
            ivd_label = int(row["IVD label"])
            grade = int(row["Pfirrman grade"])

            if ivd_label not in range(1, 6):
                continue

            if grade not in range(1, 6):
                raise ValueError(
                    f"Invalid grade for patient {patient_id}, "
                    f"IVD {ivd_label}: {grade}"
                )

            key = (patient_id, ivd_label)

            if key in grades:
                raise ValueError(f"Duplicate grading row: {key}")

            grades[key] = grade

    return grades


def grade_metrics(confusion: np.ndarray) -> dict:
    per_grade = []
    f1_values = []

    for grade_index in range(5):
        true_positive = int(
            confusion[grade_index, grade_index]
        )

        support = int(confusion[grade_index, :].sum())
        predicted = int(confusion[:, grade_index].sum())

        precision = divide(true_positive, predicted)
        recall = divide(true_positive, support)

        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        f1_values.append(f1)

        per_grade.append({
            "grade": grade_index + 1,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    return {
        "accuracy": divide(
            int(np.trace(confusion)),
            int(confusion.sum()),
        ),
        "macro_f1": float(np.mean(f1_values)),
        "per_grade": per_grade,
        "confusion_matrix": confusion.tolist(),
    }


def save_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the integrated SPIDER sagittal pipeline"
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
        "--gradings",
        type=Path,
        default=Path(
            "data/raw/spider/radiological_gradings.csv"
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
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/spider_sagittal_pipeline_v1"
        ),
    )
    parser.add_argument(
        "--exclude-patient",
        action="append",
        default=[],
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
    )

    args = parser.parse_args()

    if not 0.0 < args.probability_threshold < 1.0:
        raise ValueError("Invalid probability threshold")

    if not 0.0 < args.matching_iou <= 1.0:
        raise ValueError("Invalid matching IoU threshold")

    if args.minimum_area < 1:
        raise ValueError("Minimum area must be positive")

    excluded_patients = set(args.exclude_patient)

    records = read_validation_records(
        args.manifest,
        excluded_patients,
    )
    grades = read_grades(args.gradings)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load both models only once.
    localization_model = load_model(
        args.localization_checkpoint,
        device,
    )
    pfirrmann_model = load_pfirrmann_model(
        args.pfirrmann_checkpoint,
        device,
    )

    per_disc = []
    per_study = []

    grade_confusion = np.zeros((5, 5), dtype=np.int64)
    grade_absolute_errors = []

    reference_component_count = 0
    predicted_component_count = 0
    matched_component_count = 0

    lumbar_reference_count = 0
    correct_detection_and_level = 0
    correct_detection_level_and_grade = 0
    level_correct_with_known_grade = 0

    complete_level_studies = 0
    complete_grade_studies = 0
    eligible_five_level_studies = 0

    decoder_sources = Counter()

    for study_number, record in enumerate(records, start=1):
        patient_id = record["patient_id"]
        series_name = record["series_name"]

        image = load_localization_image(record)

        with np.load(record["example_path"]) as data:
            binary_reference = data["mask"].astype(bool)
            level_reference = data["level_mask"]

        if (
            binary_reference.shape != image.shape
            or level_reference.shape != image.shape
        ):
            raise ValueError(
                f"{series_name}: reference/image shape mismatch"
            )

        # 1. Localize discs.
        binary_probability, level_probabilities = predict(
            localization_model,
            image,
            device,
        )

        predicted_components = extract_components(
            binary_probability >= args.probability_threshold,
            args.minimum_area,
        )

        predicted_components = add_predicted_levels(
            predicted_components,
            level_probabilities,
        )

        reference_components = extract_reference_components(
            level_reference,
            args.minimum_area,
        )

        matches = match_components(
            reference_components,
            predicted_components,
            args.matching_iou,
        )

        reference_component_count += len(reference_components)
        predicted_component_count += len(predicted_components)
        matched_component_count += len(matches)

        # 2. Assign lumbar levels using the production decoder.
        positional = position_assignments(
            predicted_components
        )

        fused, decoder_result = (
            fused_sequence_assignments(
                predicted_components,
                learned_weight=args.learned_weight,
                position_weight=args.position_weight,
                upper_weight=args.upper_weight,
                maximum_spacing_cv=args.maximum_spacing_cv,
            )
        )

        assignments, decoder_source = (
            gated_hybrid_assignments(
                position=positional,
                fused=fused,
                decoder_result=decoder_result,
                spacing_cv_threshold=args.hybrid_spacing_cv,
            )
        )

        if len(assignments) != len(predicted_components):
            raise ValueError(
                f"{series_name}: assignment count mismatch"
            )

        decoder_sources[decoder_source] += 1

        # 3. Recreate the exact crop path used by integrated inference.
        original_slice = load_original_sagittal_slice(
            args.raw_root,
            series_name,
            int(record["sagittal_slice"]),
        )

        normalized_original = normalize_slice(
            original_slice
        )

        original_height, original_width = (
            original_slice.shape
        )

        indexed_crops = []

        for prediction_index, level_id in enumerate(assignments):
            if level_id not in range(5):
                continue

            component = predicted_components[
                prediction_index
            ]

            original_box = predicted_box_to_original(
                component,
                original_height,
                original_width,
                image.shape[0],
            )

            crop = make_crop(
                normalized_original,
                original_box,
                output_size=160,
            )

            indexed_crops.append(
                (prediction_index, crop)
            )

        classifications = classify_pfirrmann_crops(
            pfirrmann_model,
            indexed_crops,
            device,
        )

        # Spatial matching is independent of predicted level.
        match_by_reference = {
            int(match["reference_index"]): match
            for match in matches
        }

        matched_prediction_indices = {
            int(match["prediction_index"])
            for match in matches
        }

        study_lumbar_references = [
            (index, reference)
            for index, reference in enumerate(
                reference_components
            )
            if int(reference["true_level_id"]) < 5
        ]

        study_correct_levels = 0
        study_correct_grades = 0
        study_known_grades = 0

        for reference_index, reference in (
            study_lumbar_references
        ):
            true_level_id = int(
                reference["true_level_id"]
            )

            ivd_label = LEVEL_TO_IVD[
                true_level_id
            ]

            grade_key = (
                str(int(patient_id)),
                ivd_label,
            )

            true_grade = grades.get(grade_key)

            # An annotated lumbar level must have a grading row.
            if true_grade is None:
                raise ValueError(
                    f"{series_name}: missing Pfirrmann grade "
                    f"for IVD {ivd_label}"
                )

            lumbar_reference_count += 1
            study_known_grades += 1

            match = match_by_reference.get(
                reference_index
            )

            output = {
                "patient_id": patient_id,
                "series_name": series_name,
                "true_level": LEVEL_NAMES[
                    true_level_id
                ],
                "true_ivd_label": ivd_label,
                "true_grade": true_grade,
                "matched": 0,
                "mask_iou": "",
                "predicted_component_id": "",
                "predicted_level": "unavailable",
                "level_correct": 0,
                "predicted_grade": "",
                "grade_correct": 0,
                "grade_absolute_error": "",
                "end_to_end_correct": 0,
            }

            if match is not None:
                prediction_index = int(
                    match["prediction_index"]
                )

                predicted_component = (
                    predicted_components[
                        prediction_index
                    ]
                )

                predicted_level_id = int(
                    assignments[
                        prediction_index
                    ]
                )

                output["matched"] = 1
                output["mask_iou"] = float(
                    match["iou"]
                )
                output["predicted_component_id"] = (
                    predicted_component[
                        "component_id"
                    ]
                )

                if predicted_level_id in range(5):
                    output["predicted_level"] = (
                        LEVEL_NAMES[
                            predicted_level_id
                        ]
                    )

                    classification = (
                        classifications[
                            prediction_index
                        ]
                    )

                    output["predicted_grade"] = (
                        classification[
                            "grade_id"
                        ]
                    )

                level_correct = (
                    predicted_level_id
                    == true_level_id
                )

                if level_correct:
                    study_correct_levels += 1
                    correct_detection_and_level += 1
                    level_correct_with_known_grade += 1

                    predicted_grade = int(
                        classification[
                            "grade_id"
                        ]
                    )

                    error = abs(
                        predicted_grade - true_grade
                    )

                    grade_absolute_errors.append(
                        error
                    )

                    grade_confusion[
                        true_grade - 1,
                        predicted_grade - 1,
                    ] += 1

                    grade_correct = (
                        predicted_grade == true_grade
                    )

                    output["level_correct"] = 1
                    output["grade_correct"] = int(
                        grade_correct
                    )
                    output[
                        "grade_absolute_error"
                    ] = error
                    output[
                        "end_to_end_correct"
                    ] = int(grade_correct)

                    if grade_correct:
                        study_correct_grades += 1
                        correct_detection_level_and_grade += 1

            per_disc.append(output)

        # A prediction assigned to a lumbar level but matching an
        # upper disc (or no disc) also invalidates a complete study.
        extra_lumbar_predictions = 0

        for prediction_index, predicted_level in (
            enumerate(assignments)
        ):
            if predicted_level not in range(5):
                continue

            if prediction_index not in (
                matched_prediction_indices
            ):
                extra_lumbar_predictions += 1
                continue

            corresponding_match = next(
                match
                for match in matches
                if int(match["prediction_index"])
                == prediction_index
            )

            reference_index = int(
                corresponding_match[
                    "reference_index"
                ]
            )

            if (
                reference_components[
                    reference_index
                ]["true_level_id"] == 5
            ):
                extra_lumbar_predictions += 1

        has_all_five = sorted(
            int(reference["true_level_id"])
            for _, reference in study_lumbar_references
        ) == [0, 1, 2, 3, 4]

        if has_all_five:
            eligible_five_level_studies += 1

        all_annotated_levels_correct = (
            study_correct_levels
            == len(study_lumbar_references)
            and extra_lumbar_predictions == 0
        )

        all_annotated_grades_correct = (
            all_annotated_levels_correct
            and study_correct_grades
            == study_known_grades
        )

        if has_all_five and all_annotated_levels_correct:
            complete_level_studies += 1

        if has_all_five and all_annotated_grades_correct:
            complete_grade_studies += 1

        per_study.append({
            "patient_id": patient_id,
            "series_name": series_name,
            "reference_lumbar_discs": len(
                study_lumbar_references
            ),
            "predicted_components": len(
                predicted_components
            ),
            "matched_components": len(matches),
            "decoder_source": decoder_source,
            "decoder_status": decoder_result[
                "status"
            ],
            "correct_levels": study_correct_levels,
            "correct_grades_end_to_end": (
                study_correct_grades
            ),
            "extra_lumbar_predictions": (
                extra_lumbar_predictions
            ),
            "all_five_reference_levels": int(
                has_all_five
            ),
            "all_annotated_levels_correct": int(
                all_annotated_levels_correct
            ),
            "all_annotated_grades_correct": int(
                all_annotated_grades_correct
            ),
        })

        print(
            f"[{study_number}/{len(records)}] "
            f"{series_name}: "
            f"levels={study_correct_levels}/"
            f"{len(study_lumbar_references)} "
            f"grades={study_correct_grades}/"
            f"{study_known_grades} "
            f"decoder={decoder_source}"
        )

    localization_precision = divide(
        matched_component_count,
        predicted_component_count,
    )

    localization_recall = divide(
        matched_component_count,
        reference_component_count,
    )

    localization_f1 = (
        2.0
        * localization_precision
        * localization_recall
        / (
            localization_precision
            + localization_recall
        )
        if localization_precision
        + localization_recall
        else 0.0
    )

    conditional_grade_metrics = grade_metrics(
        grade_confusion
    )

    conditional_grade_metrics["mae"] = (
        float(np.mean(grade_absolute_errors))
        if grade_absolute_errors
        else None
    )

    summary = {
        "validation_studies": len(records),
        "excluded_patients": sorted(
            excluded_patients
        ),
        "matching_mask_iou_threshold": (
            args.matching_iou
        ),
        "localization_all_disc_components": {
            "references": (
                reference_component_count
            ),
            "predictions": (
                predicted_component_count
            ),
            "matched": (
                matched_component_count
            ),
            "precision": (
                localization_precision
            ),
            "recall": (
                localization_recall
            ),
            "f1": localization_f1,
        },
        "lumbar_level_end_to_end": {
            "references": (
                lumbar_reference_count
            ),
            "correct_detection_and_level": (
                correct_detection_and_level
            ),
            "accuracy": divide(
                correct_detection_and_level,
                lumbar_reference_count,
            ),
        },
        "pfirrmann_given_correct_detection_and_level": {
            "samples": (
                level_correct_with_known_grade
            ),
            **conditional_grade_metrics,
        },
        "detection_level_and_grade_end_to_end": {
            "references": (
                lumbar_reference_count
            ),
            "all_three_correct": (
                correct_detection_level_and_grade
            ),
            "accuracy": divide(
                correct_detection_level_and_grade,
                lumbar_reference_count,
            ),
        },
        "complete_studies": {
            "eligible_with_five_reference_levels": (
                eligible_five_level_studies
            ),
            "all_five_levels_correct": (
                complete_level_studies
            ),
            "all_five_levels_and_grades_correct": (
                complete_grade_studies
            ),
        },
        "decoder_sources": dict(
            decoder_sources
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

    save_csv(
        args.output_dir
        / "per_disc_predictions.csv",
        per_disc,
        [
            "patient_id",
            "series_name",
            "true_level",
            "true_ivd_label",
            "true_grade",
            "matched",
            "mask_iou",
            "predicted_component_id",
            "predicted_level",
            "level_correct",
            "predicted_grade",
            "grade_correct",
            "grade_absolute_error",
            "end_to_end_correct",
        ],
    )

    save_csv(
        args.output_dir
        / "per_study_results.csv",
        per_study,
        [
            "patient_id",
            "series_name",
            "reference_lumbar_discs",
            "predicted_components",
            "matched_components",
            "decoder_source",
            "decoder_status",
            "correct_levels",
            "correct_grades_end_to_end",
            "extra_lumbar_predictions",
            "all_five_reference_levels",
            "all_annotated_levels_correct",
            "all_annotated_grades_correct",
        ],
    )

    print()
    print("Validation studies:", len(records))
    print(
        "Localization F1:",
        round(localization_f1, 4),
    )
    print(
        "End-to-end level accuracy:",
        round(
            summary["lumbar_level_end_to_end"][
                "accuracy"
            ],
            4,
        ),
    )
    print(
        "Pfirrmann accuracy on correctly "
        "detected and leveled discs:",
        round(
            conditional_grade_metrics["accuracy"],
            4,
        ),
    )
    print(
        "Pfirrmann macro F1:",
        round(
            conditional_grade_metrics["macro_f1"],
            4,
        ),
    )
    print(
        "Pfirrmann MAE:",
        conditional_grade_metrics["mae"],
    )
    print(
        "End-to-end detection + level + grade:",
        round(
            summary[
                "detection_level_and_grade_end_to_end"
            ]["accuracy"],
            4,
        ),
    )
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()