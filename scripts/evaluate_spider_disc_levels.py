import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from src.models.spider_disc_unet import SpiderDiscUNet
from src.postprocess.disc_level_assignment import (
    assign_lumbar_levels,
)


LEVELS = [
    "L1-L2",
    "L2-L3",
    "L3-L4",
    "L4-L5",
    "L5-S1",
]

# SPIDER disc IDs use the inferior-to-superior order.
DISC_ID_TO_LEVEL = {
    201: "L5-S1",
    202: "L4-L5",
    203: "L3-L4",
    204: "L2-L3",
    205: "L1-L2",
}

LEVEL_TO_DISC_ID = {
    level: disc_id
    for disc_id, level in DISC_ID_TO_LEVEL.items()
}


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"CSV is empty: {path}")

    return rows


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> SpiderDiscUNet:
    # Load only a checkpoint you created and trust.
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

    model = SpiderDiscUNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model


@torch.inference_mode()
def predict_probability(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if image.shape != (256, 256):
        raise ValueError(
            f"Expected image shape (256, 256), got {image.shape}"
        )

    if not np.isfinite(image).all():
        raise ValueError("Input image contains non-finite values")

    tensor = torch.from_numpy(
        image.astype(np.float32, copy=False)
    ).unsqueeze(0).unsqueeze(0).to(device)

    logits = model(tensor)

    if logits.shape != (1, 1, 256, 256):
        raise ValueError(
            f"Unexpected model output shape: {tuple(logits.shape)}"
        )

    probability = torch.sigmoid(logits)[0, 0]

    return probability.cpu().numpy()


def extract_predicted_components(
    probability: np.ndarray,
    probability_threshold: float,
    minimum_area: int,
) -> list[dict]:
    binary_mask = (
        probability >= probability_threshold
    ).astype(np.uint8)

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8,
        )
    )

    components = []

    for component_id in range(1, count):
        x = int(
            stats[component_id, cv2.CC_STAT_LEFT]
        )
        y = int(
            stats[component_id, cv2.CC_STAT_TOP]
        )
        width = int(
            stats[component_id, cv2.CC_STAT_WIDTH]
        )
        height = int(
            stats[component_id, cv2.CC_STAT_HEIGHT]
        )
        area = int(
            stats[component_id, cv2.CC_STAT_AREA]
        )

        if area < minimum_area:
            continue

        center_x = float(centroids[component_id, 0])
        center_y = float(centroids[component_id, 1])

        components.append({
            "component_id": component_id,
            "x1": x,
            "y1": y,
            "x2": x + width,
            "y2": y + height,
            "area": area,
            "center_x": center_x,
            "center_y": center_y,
        })

    return components


def create_match_lookup(
    matched_rows: list[dict],
) -> dict[str, dict[int, int]]:
    """
    Result:

    {
        "101_t2": {
            component_id: reference_disc_id,
            ...
        }
    }
    """

    lookup = defaultdict(dict)

    for row in matched_rows:
        series_name = row["series_name"]
        component_id = int(row["component_id"])
        disc_id = int(row["disc_id"])

        if component_id in lookup[series_name]:
            raise ValueError(
                f"Duplicate component {component_id} "
                f"in {series_name}"
            )

        lookup[series_name][component_id] = disc_id

    return dict(lookup)


def print_confusion_matrix(
    matrix: np.ndarray,
) -> None:
    print()
    print(
        "Level confusion matrix "
        "[true level row, predicted level column]"
    )
    print("Level order:", LEVELS)

    for level, row in zip(LEVELS, matrix):
        print(f"  {level}: {row.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/processed/spider_disc_localization/"
            "manifest.csv"
        ),
    )

    parser.add_argument(
        "--matched-discs",
        type=Path,
        default=Path(
            "data/processed/spider_disc_localization/"
            "validation_matched_discs.csv"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/spider_disc_localization_softdice_v1/"
            "best_model.pt"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/spider_disc_level_assignment/"
            "validation_level_predictions.csv"
        ),
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
        "--exclude-patients",
        type=str,
        default="107,152",
        help=(
            "Comma-separated patients with uncertain "
            "SPIDER disc numbering"
        ),
    )

    args = parser.parse_args()

    if not 0.0 < args.probability_threshold < 1.0:
        raise ValueError(
            "Probability threshold must be between 0 and 1"
        )

    excluded_patients = {
        patient.strip()
        for patient in args.exclude_patients.split(",")
        if patient.strip()
    }

    manifest_rows = load_csv(args.manifest)
    matched_rows = load_csv(args.matched_discs)

    validation_rows = [
        row
        for row in manifest_rows
        if row["subset"] == "validation"
        and row["patient_id"] not in excluded_patients
    ]

    if not validation_rows:
        raise ValueError(
            "No validation images remain after exclusions"
        )

    matched_lookup = create_match_lookup(matched_rows)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(
        args.checkpoint,
        device,
    )

    level_to_index = {
        level: index
        for index, level in enumerate(LEVELS)
    }

    confusion = np.zeros(
        (len(LEVELS), len(LEVELS)),
        dtype=np.int64,
    )

    status_counts = Counter()
    component_count_distribution = Counter()

    output_rows = []
    review_rows = []

    correct_levels = 0
    expected_levels = 0
    assigned_components = 0
    false_assignments = 0
    matched_lumbar_discs = 0
    complete_studies = 0

    for manifest_row in validation_rows:
        patient_id = manifest_row["patient_id"]
        series_name = manifest_row["series_name"]
        example_path = Path(
            manifest_row["example_path"]
        )

        with np.load(example_path) as data:
            image = data["image"].astype(np.float32)

        probability = predict_probability(
            model,
            image,
            device,
        )

        components = extract_predicted_components(
            probability,
            args.probability_threshold,
            args.minimum_area,
        )

        component_count_distribution[
            len(components)
        ] += 1

        assignment_result = assign_lumbar_levels(
            components,
            minimum_area=args.minimum_area,
        )

        status_counts[
            assignment_result["status"]
        ] += 1

        match_by_component = matched_lookup.get(
            series_name,
            {},
        )

        assigned_by_predicted_level = {}
        correct_in_study = 0

        for assignment in assignment_result["assignments"]:
            component_index = int(
                assignment["component_index"]
            )

            if not 0 <= component_index < len(components):
                raise IndexError(
                    f"{series_name}: invalid component index "
                    f"{component_index}"
                )

            component = components[component_index]
            component_id = int(
                component["component_id"]
            )

            predicted_level = assignment["position"]
            reference_disc_id = match_by_component.get(
                component_id
            )

            true_level = None

            if reference_disc_id is not None:
                true_level = DISC_ID_TO_LEVEL.get(
                    reference_disc_id
                )

            assigned_components += 1

            if true_level is None:
                false_assignments += 1
            else:
                true_index = level_to_index[true_level]
                predicted_index = level_to_index[
                    predicted_level
                ]

                confusion[
                    true_index,
                    predicted_index,
                ] += 1

                if predicted_level == true_level:
                    correct_levels += 1
                    correct_in_study += 1

            assigned_by_predicted_level[
                predicted_level
            ] = {
                "component_id": component_id,
                "true_level": true_level,
                "reference_disc_id": reference_disc_id,
                "bounding_box": assignment[
                    "bounding_box"
                ],
                "center_x": assignment["center_x"],
                "center_y": assignment["center_y"],
                "area": assignment["area"],
            }

        matched_reference_discs = {
            disc_id
            for disc_id in match_by_component.values()
            if disc_id in DISC_ID_TO_LEVEL
        }

        matched_lumbar_discs += len(
            matched_reference_discs
        )

        study_correct = 0

        for true_level in LEVELS:
            expected_levels += 1
            true_disc_id = LEVEL_TO_DISC_ID[
                true_level
            ]

            matched_component_id = None

            for component_id, disc_id in (
                match_by_component.items()
            ):
                if disc_id == true_disc_id:
                    matched_component_id = component_id
                    break

            predicted_level_for_true_disc = None
            predicted_component_id = None
            predicted_box = None

            for predicted_level, information in (
                assigned_by_predicted_level.items()
            ):
                if (
                    information["reference_disc_id"]
                    == true_disc_id
                ):
                    predicted_level_for_true_disc = (
                        predicted_level
                    )
                    predicted_component_id = (
                        information["component_id"]
                    )
                    predicted_box = information[
                        "bounding_box"
                    ]
                    break

            correct = (
                predicted_level_for_true_disc
                == true_level
            )

            if correct:
                study_correct += 1

            output_rows.append({
                "patient_id": patient_id,
                "series_name": series_name,
                "true_level": true_level,
                "true_disc_id": true_disc_id,
                "matched_component_id": (
                    matched_component_id
                    if matched_component_id is not None
                    else ""
                ),
                "predicted_level": (
                    predicted_level_for_true_disc
                    if predicted_level_for_true_disc
                    is not None
                    else ""
                ),
                "predicted_component_id": (
                    predicted_component_id
                    if predicted_component_id is not None
                    else ""
                ),
                "predicted_box": (
                    predicted_box
                    if predicted_box is not None
                    else ""
                ),
                "correct": int(correct),
                "assignment_status": (
                    assignment_result["status"]
                ),
                "assignment_reasons": " | ".join(
                    assignment_result["reasons"]
                ),
                "detected_components": len(components),
                "spacing_cv": assignment_result.get(
                    "spacing_cv",
                    "",
                ),
            })

        if study_correct == 5:
            complete_studies += 1
        else:
            review_rows.append({
                "patient_id": patient_id,
                "series_name": series_name,
                "correct_levels": study_correct,
                "detected_components": len(components),
                "status": assignment_result["status"],
                "reasons": assignment_result["reasons"],
            })

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(output_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    image_count = len(validation_rows)

    end_to_end_accuracy = (
        correct_levels / expected_levels
    )

    assignment_precision = (
        correct_levels / assigned_components
        if assigned_components
        else 0.0
    )

    matched_localization_recall = (
        matched_lumbar_discs / expected_levels
    )

    complete_study_accuracy = (
        complete_studies / image_count
    )

    print("Checkpoint:", args.checkpoint)
    print("Device:", device)
    print("Validation images:", image_count)
    print(
        "Excluded patients:",
        sorted(excluded_patients),
    )
    print(
        "Probability threshold:",
        args.probability_threshold,
    )
    print("Minimum component area:", args.minimum_area)

    print()
    print(
        "Predicted component counts:",
        dict(sorted(component_count_distribution.items())),
    )
    print(
        "Assignment status counts:",
        dict(status_counts),
    )

    print()
    print("Expected lumbar levels:", expected_levels)
    print(
        "Reference levels with matched components:",
        matched_lumbar_discs,
    )
    print(
        "Matched localization recall:",
        f"{matched_localization_recall:.4f}",
    )
    print("Assigned components:", assigned_components)
    print("False or non-lumbar assignments:", false_assignments)
    print("Correct level assignments:", correct_levels)
    print(
        "End-to-end level accuracy:",
        f"{end_to_end_accuracy:.4f}",
    )
    print(
        "Assignment precision:",
        f"{assignment_precision:.4f}",
    )
    print(
        "Studies with all five levels correct:",
        f"{complete_studies}/{image_count}",
    )
    print(
        "Complete-study accuracy:",
        f"{complete_study_accuracy:.4f}",
    )

    print_confusion_matrix(confusion)

    if review_rows:
        print()
        print("First 12 studies requiring review:")

        for row in review_rows[:12]:
            print(row)

    print()
    print("Saved:", args.output)


if __name__ == "__main__":
    main()