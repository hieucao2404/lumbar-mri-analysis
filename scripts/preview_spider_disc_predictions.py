import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.spider_disc_localization_dataset import (
    SpiderDiscLocalizationDataset,
)
from src.models.spider_disc_unet import SpiderDiscUNet


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
        "--output",
        type=Path,
        default=Path(
            "outputs/spider_disc_localization_softdice_v1/"
            "validation_preview.png"
        ),
    )
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()

    dataset = SpiderDiscLocalizationDataset(
        args.manifest,
        subset="validation",
    )

    if args.count < 1:
        raise ValueError("--count must be at least 1")

    count = min(args.count, len(dataset))
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

    # Spread the examples across the validation dataset.
    indices = np.linspace(
        0, len(dataset) - 1, count, dtype=int
    )

    fig, axes = plt.subplots(
        count,
        3,
        figsize=(10, 3 * count),
        squeeze=False,
    )

    for plot_row, dataset_index in enumerate(indices):
        image_tensor, mask_tensor = dataset[int(dataset_index)]

        with torch.no_grad():
            logits = model(
                image_tensor.unsqueeze(0).to(device)
            )
            probability = (
                torch.sigmoid(logits)[0, 0]
                .cpu()
                .numpy()
            )

        image = image_tensor[0].numpy()
        truth = mask_tensor[0].numpy() >= 0.5
        prediction = probability >= 0.5

        intersection = np.logical_and(
            prediction, truth
        ).sum()

        dice = (
            (2 * intersection + 1)
            / (prediction.sum() + truth.sum() + 1)
        )

        row = dataset.rows[int(dataset_index)]
        patient_id = row["patient_id"]

        axes[plot_row, 0].imshow(image, cmap="gray")
        axes[plot_row, 0].set_title(
            f"Patient {patient_id} | MRI"
        )

        axes[plot_row, 1].imshow(image, cmap="gray")
        axes[plot_row, 1].imshow(
            np.ma.masked_where(~truth, truth),
            cmap="Greens",
            alpha=0.55,
            vmin=0,
            vmax=1,
        )
        axes[plot_row, 1].set_title("Ground truth: green")

        axes[plot_row, 2].imshow(image, cmap="gray")
        axes[plot_row, 2].imshow(
            np.ma.masked_where(
                ~prediction, prediction
            ),
            cmap="Reds",
            alpha=0.55,
            vmin=0,
            vmax=1,
        )
        axes[plot_row, 2].set_title(
            f"Prediction: red | Dice {dice:.3f}"
        )

        for ax in axes[plot_row]:
            ax.axis("off")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Saved preview:", args.output)


if __name__ == "__main__":
    main()