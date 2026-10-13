import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import SimpleITK as sitk

from scripts.build_spider_paired_crops import (
    normalize_slice,
)


LEVEL_NAMES = (
    "L1-L2",
    "L2-L3",
    "L3-L4",
    "L4-L5",
    "L5-S1",
)

LEVEL_COLORS = {
    "L1-L2": "#1f77b4",
    "L2-L3": "#ff7f0e",
    "L3-L4": "#2ca02c",
    "L4-L5": "#d62728",
    "L5-S1": "#9467bd",
}

# SPIDER IVD labels are ordered inferior to superior.
IVD_TO_LEVEL = {
    5: "L1-L2",
    4: "L2-L3",
    3: "L3-L4",
    2: "L4-L5",
    1: "L5-S1",
}

ROMAN_TO_INT = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
}

INT_TO_ROMAN = {
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
    5: "V",
}


def load_oriented_slice(
    path: Path,
    slice_index: int,
) -> np.ndarray:
    """
    Load one sagittal slice using the same orientation used by the
    localization and classification pipeline.
    """

    if not path.is_file():
        raise FileNotFoundError(path)

    source = sitk.DICOMOrient(
        sitk.ReadImage(str(path)),
        "LPS",
    )

    volume = np.flip(
        sitk.GetArrayFromImage(source),
        axis=0,
    )

    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D volume, got {volume.shape}"
        )

    if slice_index < 0 or slice_index >= volume.shape[2]:
        raise IndexError(
            f"Slice {slice_index} is outside sagittal axis "
            f"with size {volume.shape[2]}"
        )

    return np.take(
        volume,
        slice_index,
        axis=2,
    )


def mask_box(
    mask: np.ndarray,
) -> tuple[int, int, int, int] | None:
    """
    Convert a binary mask into an exclusive-corner bounding box.
    """

    ys, xs = np.nonzero(mask)

    if len(xs) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def load_ground_truth(
    grading_path: Path,
    patient_id: str,
) -> dict[str, dict]:
    """
    Load Pfirrmann, bulging, and herniation labels for one patient.
    """

    with grading_path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    ground_truth = {}

    for row in rows:
        if row["Patient"].strip() != patient_id:
            continue

        ivd_label = int(row["IVD label"])

        if ivd_label not in IVD_TO_LEVEL:
            continue

        level = IVD_TO_LEVEL[ivd_label]

        ground_truth[level] = {
            "ivd_label": ivd_label,
            "mask_value": 200 + ivd_label,
            "pfirrmann_grade": int(
                row["Pfirrman grade"]
            ),
            "disc_bulging": int(
                row["Disc bulging"]
            ),
            "disc_herniation": int(
                row["Disc herniation"]
            ),
        }

    return ground_truth


def predicted_grade_to_int(
    value: str,
) -> int | None:
    """
    Convert 'Pfirrmann IV' into integer grade 4.
    """

    if not isinstance(value, str):
        return None

    parts = value.strip().split()

    if not parts:
        return None

    return ROMAN_TO_INT.get(parts[-1])


def box_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection = intersection_width * intersection_height

    first_area = (
        max(0, first[2] - first[0])
        * max(0, first[3] - first[1])
    )

    second_area = (
        max(0, second[2] - second[0])
        * max(0, second[3] - second[1])
    )

    union = first_area + second_area - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def configure_axis(
    axis,
    image: np.ndarray,
    title: str,
) -> None:
    axis.imshow(
        image,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )

    axis.set_title(
        title,
        fontsize=11,
        fontweight="bold",
    )

    axis.axis("off")


def draw_ground_truth(
    axis,
    image: np.ndarray,
    mask_slice: np.ndarray,
    ground_truth: dict[str, dict],
) -> dict[str, tuple[int, int, int, int]]:
    """
    Draw transparent ground-truth masks with solid boxes.
    """

    configure_axis(
        axis,
        image,
        "Ground Truth",
    )

    height, width = image.shape

    rgba_overlay = np.zeros(
        (height, width, 4),
        dtype=np.float32,
    )

    boxes = {}

    for level in LEVEL_NAMES:
        if level not in ground_truth:
            continue

        record = ground_truth[level]

        binary_mask = (
            mask_slice == record["mask_value"]
        )

        box = mask_box(binary_mask)

        if box is None:
            continue

        boxes[level] = box

        color = matplotlib.colors.to_rgba(
            LEVEL_COLORS[level]
        )

        rgba_overlay[
            binary_mask,
            0,
        ] = color[0]

        rgba_overlay[
            binary_mask,
            1,
        ] = color[1]

        rgba_overlay[
            binary_mask,
            2,
        ] = color[2]

        rgba_overlay[
            binary_mask,
            3,
        ] = 0.35

        x1, y1, x2, y2 = box

        rectangle = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1.8,
            edgecolor=LEVEL_COLORS[level],
            facecolor="none",
        )

        axis.add_patch(rectangle)

        grade = record["pfirrmann_grade"]
        bulging = record["disc_bulging"]
        herniation = record["disc_herniation"]

        axis.text(
            x1,
            max(3, y1 - 4),
            (
                f"{level}  GT {INT_TO_ROMAN[grade]}\n"
                f"B={bulging} H={herniation}"
            ),
            fontsize=7,
            color="white",
            bbox={
                "facecolor": LEVEL_COLORS[level],
                "alpha": 0.85,
                "pad": 2,
                "edgecolor": "none",
            },
        )

    axis.imshow(rgba_overlay)

    return boxes


def draw_predictions(
    axis,
    image: np.ndarray,
    predictions: dict,
) -> dict[str, tuple[int, int, int, int]]:
    """
    Draw predicted boxes and Pfirrmann grades.
    """

    configure_axis(
        axis,
        image,
        "Pipeline Predictions",
    )

    boxes = {}

    for level_record in predictions["levels"]:
        level = level_record["position"]

        if level_record["status"] != "available":
            continue

        box = tuple(
            int(value)
            for value in level_record[
                "bounding_box_original"
            ]
        )

        boxes[level] = box

        x1, y1, x2, y2 = box

        rectangle = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1.8,
            edgecolor=LEVEL_COLORS[level],
            facecolor="none",
            linestyle="--",
        )

        axis.add_patch(rectangle)

        grade = level_record[
            "disc_degeneration"
        ]

        confidence = level_record[
            "disc_degeneration_confidence"
        ]

        level_probability = level_record[
            "level_model_probability"
        ]

        bulging_status = level_record.get("disc_bulging", "unavailable")
        bulging_probability = level_record.get("disc_bulging_probability")

        if bulging_status == "unavailable" or bulging_probability is None:
            bulging_label = "B=NA"
        else:
            bulging_label = (
                f"B={int(bulging_status == 'detected')} "
                f"p={float(bulging_probability):.2f}"
            )

        axis.text(
            x1,
            min(image.shape[0] - 5, y2 + 5),
            (
                f"{level}  Pred {grade.split()[-1]}\n"
                f"G={confidence:.2f} L={level_probability:.2f}\n"
                f"{bulging_label}"
            ),
            fontsize=7,
            color="white",
            va="top",
            bbox={
                "facecolor": LEVEL_COLORS[level],
                "alpha": 0.85,
                "pad": 2,
                "edgecolor": "none",
            },
        )

    return boxes


def draw_comparison(
    axis,
    image: np.ndarray,
    ground_truth_boxes: dict,
    prediction_boxes: dict,
    ground_truth: dict,
    predictions_by_level: dict,
) -> None:
    """
    Draw ground-truth boxes in green and predictions in red.
    """

    configure_axis(
        axis,
        image,
        "Spatial Comparison",
    )

    for level in LEVEL_NAMES:
        if level in ground_truth_boxes:
            x1, y1, x2, y2 = ground_truth_boxes[
                level
            ]

            axis.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2.0,
                    edgecolor="lime",
                    facecolor="none",
                    linestyle="-",
                )
            )

        if level in prediction_boxes:
            x1, y1, x2, y2 = prediction_boxes[
                level
            ]

            axis.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=1.8,
                    edgecolor="red",
                    facecolor="none",
                    linestyle="--",
                )
            )

        if (
            level in ground_truth
            and level in predictions_by_level
        ):
            truth_grade = ground_truth[
                level
            ]["pfirrmann_grade"]

            predicted_grade = predicted_grade_to_int(
                predictions_by_level[
                    level
                ]["disc_degeneration"]
            )

            marker = (
                "✓"
                if predicted_grade == truth_grade
                else "✗"
            )

            box = (
                prediction_boxes.get(level)
                or ground_truth_boxes.get(level)
            )

            if box is not None:
                axis.text(
                    box[2] + 3,
                    (box[1] + box[3]) / 2,
                    (
                        f"{level} {marker}\n"
                        f"{truth_grade}→{predicted_grade}"
                    ),
                    color=(
                        "lime"
                        if predicted_grade == truth_grade
                        else "yellow"
                    ),
                    fontsize=7,
                    va="center",
                    bbox={
                        "facecolor": "black",
                        "alpha": 0.65,
                        "pad": 2,
                        "edgecolor": "none",
                    },
                )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="lime",
            linewidth=2,
            linestyle="-",
            label="Ground truth",
        ),
        Line2D(
            [0],
            [0],
            color="red",
            linewidth=2,
            linestyle="--",
            label="Prediction",
        ),
    ]

    axis.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=8,
        framealpha=0.75,
    )


def calculate_study_metrics(
    ground_truth: dict[str, dict],
    predictions_by_level: dict[str, dict],
    ground_truth_boxes: dict,
    prediction_boxes: dict,
) -> dict:
    exact_grades = 0
    absolute_error = 0
    available_predictions = 0
    ious = []

    for level in LEVEL_NAMES:
        if level not in ground_truth:
            continue

        if level not in predictions_by_level:
            continue

        prediction = predictions_by_level[level]

        if prediction["status"] != "available":
            continue

        predicted_grade = predicted_grade_to_int(
            prediction["disc_degeneration"]
        )

        if predicted_grade is None:
            continue

        true_grade = ground_truth[
            level
        ]["pfirrmann_grade"]

        available_predictions += 1
        absolute_error += abs(
            predicted_grade - true_grade
        )

        if predicted_grade == true_grade:
            exact_grades += 1

        if (
            level in ground_truth_boxes
            and level in prediction_boxes
        ):
            ious.append(
                box_iou(
                    ground_truth_boxes[level],
                    prediction_boxes[level],
                )
            )

    total_ground_truth = sum(
        level in ground_truth
        for level in LEVEL_NAMES
    )

    return {
        "total_ground_truth": total_ground_truth,
        "available_predictions": available_predictions,
        "correct_grades": exact_grades,
        "end_to_end_grade_accuracy": (
            exact_grades / total_ground_truth
            if total_ground_truth
            else 0.0
        ),
        "grade_mae": (
            absolute_error / available_predictions
            if available_predictions
            else None
        ),
        "mean_same_level_box_iou": (
            float(np.mean(ious))
            if ious
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--series-name",
        required=True,
        help="For example: 101_t2",
    )

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/spider"),
    )

    parser.add_argument(
        "--prediction",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if args.prediction is None:
        args.prediction = (
            Path("outputs/sagittal_inference")
            / f"{args.series_name}.json"
        )

    if args.output is None:
        args.output = (
            Path("outputs/sagittal_inference")
            / f"{args.series_name}_comparison.png"
        )

    if not args.prediction.is_file():
        raise FileNotFoundError(
            f"Prediction JSON not found: "
            f"{args.prediction}"
        )

    with args.prediction.open() as file:
        predictions = json.load(file)

    if predictions["series_name"] != args.series_name:
        raise ValueError(
            "Prediction JSON series does not match "
            "--series-name"
        )

    patient_id = str(
        predictions["study_id"]
    )

    slice_index = int(
        predictions["sagittal_slice"]
    )

    image_path = (
        args.raw_root
        / "images"
        / f"{args.series_name}.mha"
    )

    mask_path = (
        args.raw_root
        / "masks"
        / f"{args.series_name}.mha"
    )

    grading_path = (
        args.raw_root
        / "radiological_gradings.csv"
    )

    image_slice = load_oriented_slice(
        image_path,
        slice_index,
    ).astype(np.float32)

    mask_slice = load_oriented_slice(
        mask_path,
        slice_index,
    )

    if image_slice.shape != mask_slice.shape:
        raise ValueError(
            "Image and mask slice shapes differ: "
            f"{image_slice.shape} versus {mask_slice.shape}"
        )

    display_image = normalize_slice(
        image_slice
    )

    ground_truth = load_ground_truth(
        grading_path,
        patient_id,
    )

    predictions_by_level = {
        row["position"]: row
        for row in predictions["levels"]
    }

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(18, 7),
    )

    configure_axis(
        axes[0],
        display_image,
        "Original Sagittal T2 MRI",
    )

    ground_truth_boxes = draw_ground_truth(
        axes[1],
        display_image,
        mask_slice,
        ground_truth,
    )

    prediction_boxes = draw_predictions(
        axes[2],
        display_image,
        predictions,
    )

    draw_comparison(
        axes[3],
        display_image,
        ground_truth_boxes,
        prediction_boxes,
        ground_truth,
        predictions_by_level,
    )

    metrics = calculate_study_metrics(
        ground_truth,
        predictions_by_level,
        ground_truth_boxes,
        prediction_boxes,
    )

    mean_iou = metrics[
        "mean_same_level_box_iou"
    ]

    mean_iou_text = (
        f"{mean_iou:.3f}"
        if mean_iou is not None
        else "N/A"
    )

    grade_mae = metrics["grade_mae"]

    grade_mae_text = (
        f"{grade_mae:.3f}"
        if grade_mae is not None
        else "N/A"
    )

    figure.suptitle(
        (
            f"SPIDER Sagittal Result: {args.series_name}\n"
            f"Correct grades: "
            f"{metrics['correct_grades']}/"
            f"{metrics['total_ground_truth']} | "
            f"End-to-end grade accuracy: "
            f"{metrics['end_to_end_grade_accuracy']:.3f} | "
            f"Grade MAE: {grade_mae_text} | "
            f"Same-level box IoU: {mean_iou_text}"
        ),
        fontsize=13,
        fontweight="bold",
    )

    figure.text(
        0.5,
        0.02,
        (
            "GT = ground truth, Pred = prediction, "
            "G = grade confidence, L = learned level probability. "
            "Green solid boxes are ground truth; "
            "red dashed boxes are predictions."
        ),
        ha="center",
        fontsize=9,
    )

    figure.tight_layout(
        rect=(0, 0.05, 1, 0.92)
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        args.output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("Series:", args.series_name)
    print("Slice:", slice_index)
    print(
        "Correct grades:",
        f"{metrics['correct_grades']}/"
        f"{metrics['total_ground_truth']}",
    )
    print(
        "End-to-end grade accuracy:",
        round(
            metrics[
                "end_to_end_grade_accuracy"
            ],
            4,
        ),
    )
    print("Grade MAE:", grade_mae_text)
    print("Same-level mean box IoU:", mean_iou_text)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()