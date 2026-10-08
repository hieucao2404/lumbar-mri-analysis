# scripts/evaluate_spider_disc_instances.py

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import SimpleITK as sitk
import torch

from src.models.spider_disc_unet import SpiderDiscUNet


def load_reference_instances(
    mask_path: Path,
    slice_index: int,
    output_size: int,
) -> np.ndarray:
    source = sitk.DICOMOrient(
        sitk.ReadImage(str(mask_path)),
        "LPS",
    )

    volume = np.flip(
        sitk.GetArrayFromImage(source),
        axis=0,
    )

    mask_slice = volume[:, :, slice_index].astype(np.int32)
    height, width = mask_slice.shape

    scale = min(output_size / width, output_size / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    resized = np.asarray(
        Image.fromarray(mask_slice).resize(
            (new_width, new_height),
            Image.Resampling.NEAREST,
        ),
        dtype=np.int32,
    )

    canvas = np.zeros(
        (output_size, output_size),
        dtype=np.int32,
    )

    offset_x = (output_size - new_width) // 2
    offset_y = (output_size - new_height) // 2

    canvas[
        offset_y:offset_y + new_height,
        offset_x:offset_x + new_width,
    ] = resized

    canvas[(canvas < 201) | (canvas >= 300)] = 0
    return canvas


def reference_objects(
    instance_map: np.ndarray,
    min_area: int,
) -> list[dict]:
    objects = []

    for disc_id in np.unique(instance_map):
        if disc_id == 0:
            continue

        pixels = instance_map == disc_id

        if int(pixels.sum()) >= min_area:
            objects.append({
                "id": int(disc_id),
                "pixels": pixels,
            })

    return objects


def predicted_objects(
    probability: np.ndarray,
    threshold: float,
    min_area: int,
) -> list[dict]:
    binary = (probability >= threshold).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    objects = []

    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])

        if area >= min_area:
            objects.append({
                "id": component_id,
                "pixels": labels == component_id,
            })

    return objects


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else 0.0


def pixel_box(
    pixels: np.ndarray,
) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(pixels)

    if len(xs) == 0:
        raise ValueError("Empty component has no bounding box")

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def match_objects(
    references: list[dict],
    predictions: list[dict],
    iou_threshold: float,
) -> tuple[list[dict], list[int], list[int]]:
    candidates = []

    for reference_index, reference in enumerate(references):
        for prediction_index, prediction in enumerate(predictions):
            iou = mask_iou(
                reference["pixels"],
                prediction["pixels"],
            )

            if iou >= iou_threshold:
                candidates.append(
                    (iou, reference_index, prediction_index)
                )

    candidates.sort(reverse=True)

    used_references = set()
    used_predictions = set()
    matches = []

    for iou, reference_index, prediction_index in candidates:
        if (
            reference_index in used_references
            or prediction_index in used_predictions
        ):
            continue

        used_references.add(reference_index)
        used_predictions.add(prediction_index)

        matches.append({
            "disc_id": references[reference_index]["id"],
            "component_id": predictions[prediction_index]["id"],
            "iou": iou,
        })

    missed = [
        reference["id"]
        for index, reference in enumerate(references)
        if index not in used_references
    ]

    extra = [
        prediction["id"]
        for index, prediction in enumerate(predictions)
        if index not in used_predictions
    ]

    return matches, missed, extra


@torch.no_grad()
def predict_probability(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    tensor = torch.from_numpy(
        image.astype(np.float32)
    )[None, None].to(device)

    logits = model(tensor)
    return torch.sigmoid(logits)[0, 0].cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/spider"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/processed/spider_disc_localization/manifest.csv"
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
        "--matches-output",
        type=Path,
        default=Path(
            "data/processed/spider_disc_localization/"
            "validation_matched_discs.csv"
        ),
    )
    parser.add_argument("--subset", default="validation")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=25)

    args = parser.parse_args()

    with args.manifest.open(newline="") as file:
        rows = [
            row
            for row in csv.DictReader(file)
            if row["subset"] == args.subset
        ]

    if not rows:
        raise ValueError(
            f"No examples for subset {args.subset!r}"
        )

    series_names = [row["series_name"] for row in rows]
    if len(series_names) != len(set(series_names)):
        raise ValueError("Manifest contains duplicate series")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Use only a checkpoint you created and trust.
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    model = SpiderDiscUNet().to(device)
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    total_references = 0
    total_predictions = 0
    total_matches = 0
    matched_ious = []
    exact_count_images = 0
    complete_match_images = 0
    problems = []
    matched_rows = []
    processed_images = 0

    # Everything in this loop runs ONCE for each MRI image.
    for row in rows:
        processed_images += 1
        series_name = row["series_name"]
        example_path = Path(row["example_path"])

        with np.load(example_path) as example:
            image = example["image"]
            exported_binary_mask = example["mask"]

        instance_map = load_reference_instances(
            args.root / "masks" / f"{series_name}.mha",
            int(row["sagittal_slice"]),
            image.shape[0],
        )

        if not np.array_equal(
            instance_map > 0,
            exported_binary_mask.astype(bool),
        ):
            raise ValueError(
                f"{series_name}: reference instances do not "
                "match the exported binary mask"
            )

        probability = predict_probability(
            model,
            image,
            device,
        )

        references = reference_objects(
            instance_map,
            args.min_area,
        )
        predictions = predicted_objects(
            probability,
            args.threshold,
            args.min_area,
        )

        matches, missed, extra = match_objects(
            references,
            predictions,
            args.iou_threshold,
        )

        prediction_by_id = {
            item["id"]: item for item in predictions
        }

        # This inner loop creates CSV records. It does NOT update
        # image-level evaluation totals.
        for match in matches:
            component = prediction_by_id[match["component_id"]]
            x1, y1, x2, y2 = pixel_box(
                component["pixels"]
            )

            matched_rows.append({
                "patient_id": row["patient_id"],
                "series_name": series_name,
                "subset": row["subset"],
                "sagittal_slice": row["sagittal_slice"],
                "example_path": str(example_path),
                "disc_id": match["disc_id"],
                "component_id": match["component_id"],
                "mask_iou": match["iou"],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "original_height": row["height"],
                "original_width": row["width"],
            })

        # Back at image-loop indentation: run exactly once per MRI.
        total_references += len(references)
        total_predictions += len(predictions)
        total_matches += len(matches)
        matched_ious.extend(
            match["iou"] for match in matches
        )

        if len(references) == len(predictions):
            exact_count_images += 1

        if not missed and not extra:
            complete_match_images += 1
        else:
            problems.append({
                "series": series_name,
                "reference_count": len(references),
                "predicted_count": len(predictions),
                "matched": len(matches),
                "missed_ids": missed,
                "extra_components": extra,
            })

    # These checks run AFTER all MRI images have been processed.
    if processed_images != len(rows):
        raise AssertionError("Not every manifest image was processed")

    if len(matched_rows) != total_matches:
        raise AssertionError(
            "CSV row count differs from match count"
        )

    if exact_count_images > processed_images:
        raise AssertionError("Invalid equal-count image total")

    if complete_match_images > processed_images:
        raise AssertionError("Invalid complete-match image total")

    precision = (
        total_matches / total_predictions
        if total_predictions
        else 0.0
    )
    recall = (
        total_matches / total_references
        if total_references
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    print("Subset:", args.subset)
    print("Images:", processed_images)
    print("Probability threshold:", args.threshold)
    print("Matching mask IoU threshold:", args.iou_threshold)
    print("Minimum component area:", args.min_area)
    print("Reference disc instances:", total_references)
    print("Predicted components:", total_predictions)
    print("Matched discs:", total_matches)
    print(f"Disc detection precision: {precision:.4f}")
    print(f"Disc detection recall:    {recall:.4f}")
    print(f"Disc detection F1:        {f1:.4f}")

    if matched_ious:
        print(
            "Mean mask IoU of matched discs:",
            f"{np.mean(matched_ious):.4f}",
        )

    print(
        "Images with equal counts:",
        f"{exact_count_images}/{processed_images}",
    )
    print(
        "Images with every disc matched:",
        f"{complete_match_images}/{processed_images}",
    )

    print("\nFirst 12 images requiring review:")
    for problem in problems[:12]:
        print(problem)

    if not matched_rows:
        raise ValueError("No matched discs to export")

    args.matches_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.matches_output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(matched_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(matched_rows)

    print("\nMatched-disc records saved:", len(matched_rows))
    print("CSV:", args.matches_output)


if __name__ == "__main__":
    main()