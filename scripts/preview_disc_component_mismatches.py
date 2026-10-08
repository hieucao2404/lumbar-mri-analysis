import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

from src.data.spider_disc_localization_dataset import (
    SpiderDiscLocalizationDataset,
)
from src.models.spider_disc_unet import SpiderDiscUNet
from src.postprocess.disc_components import extract_disc_boxes


def draw_boxes(ax, boxes, color: str) -> None:
    for number, box in enumerate(boxes, start=1):
        x1, y1 = box["x1"], box["y1"]
        width = box["x2"] - x1
        height = box["y2"] - y1

        ax.add_patch(
            Rectangle(
                (x1, y1),
                width,
                height,
                fill=False,
                edgecolor=color,
                linewidth=1.5,
            )
        )
        ax.text(
            x1,
            max(0, y1 - 2),
            str(number),
            color=color,
            fontsize=8,
            weight="bold",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
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
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/spider_disc_localization_softdice_v1/"
            "component_mismatches"
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=25)
    args = parser.parse_args()

    dataset = SpiderDiscLocalizationDataset(
        args.manifest,
        subset="validation",
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    model = SpiderDiscUNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    with torch.no_grad():
        for index in range(len(dataset)):
            image_tensor, target_tensor = dataset[index]

            logits = model(
                image_tensor.unsqueeze(0).to(device)
            )
            probability = (
                torch.sigmoid(logits)[0, 0]
                .cpu()
                .numpy()
            )

            image = image_tensor[0].numpy()
            target = target_tensor[0].numpy()

            reference_boxes = extract_disc_boxes(
                target,
                threshold=0.5,
                min_area=args.min_area,
            )
            predicted_boxes = extract_disc_boxes(
                probability,
                threshold=args.threshold,
                min_area=args.min_area,
            )

            if len(reference_boxes) == len(predicted_boxes):
                continue

            patient_id = dataset.rows[index]["patient_id"]

            fig, axes = plt.subplots(
                1, 2, figsize=(10, 6), squeeze=False
            )
            reference_ax, prediction_ax = axes[0]

            reference_ax.imshow(image, cmap="gray")
            draw_boxes(reference_ax, reference_boxes, "lime")
            reference_ax.set_title(
                f"Reference: {len(reference_boxes)} components"
            )

            prediction_ax.imshow(image, cmap="gray")
            draw_boxes(prediction_ax, predicted_boxes, "red")
            prediction_ax.set_title(
                f"Prediction: {len(predicted_boxes)} components"
            )

            for ax in (reference_ax, prediction_ax):
                ax.axis("off")

            fig.suptitle(
                f"Validation index {index} | Patient {patient_id}"
            )
            fig.tight_layout()

            output = args.output_dir / (
                f"index_{index:02d}_patient_{patient_id}.png"
            )
            fig.savefig(output, dpi=160)
            plt.close(fig)

            print(
                f"Index {index}, patient {patient_id}: "
                f"reference={len(reference_boxes)}, "
                f"prediction={len(predicted_boxes)} "
                f"-> {output}"
            )
            saved += 1

    print(f"\nSaved {saved} mismatch previews to {args.output_dir}")


if __name__ == "__main__":
    main()