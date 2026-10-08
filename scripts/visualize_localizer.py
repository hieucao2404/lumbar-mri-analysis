import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Patch, Rectangle
from PIL import Image

from src.inference.lumbar_localizer import (
    load_lumbar_localizer,
    localize_lumbar,
)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a Stage 1 lumbar localization."
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=0.08,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.20,
    )

    return parser.parse_args()


def add_box(
        axes,
        box,
        color: str,
        linewidth: float,
) -> None:
    x1, y1, x2, y2 = box

    rectangle = Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
    )

    axes.add_patch(rectangle)

def main() -> None:
    arguments = parse_arguments()

    if not arguments.image.exists():
        raise FileNotFoundError(
            f"Checking not found: {arguments.checkpoint}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_lumbar_localizer(
        checkpoint_path=arguments.checkpoint,
        device=device,
    )

    with Image.open(arguments.image) as opened_image:
        image = opened_image.convert("L")

    result = localize_lumbar(
        image=image,
        model=model,
        device=device,
        target_size=(128, 128),
        margin_fraction=arguments.margin,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10, 6),
    )

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original MRI with localization")

    add_box(
        axes=axes[0],
        box=result.box_original,
        color="red",
        linewidth=2,
    )

    add_box(
        axes=axes[0],
        box=result.crop_box_original,
        color="deepskyblue",
        linewidth=2,
    )

    axes[0].legend(
        handles=[
            Patch(
                facecolor="none",
                edgecolor="red",
                label="CNN prediction",
            ),
            Patch(
                facecolor="none",
                edgecolor="deepskyblue",
                label="Expanded crop",
            ),
        ],
        loc="upper right"
    )

    axes[0].axis("off")

    axes[1].imshow(result.crop, cmap="gray")
    axes[1].set_title(
        f"Lumbar crop: {result.crop.size[0]}"
        f"{result.crop.size[1]}"
    )
    axes[1].axis("off")

    figure.suptitle(
        f"Stage 1 Lumbar Localization - {arguments.image.name}"
    )

    figure.tight_layout()

    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        arguments.output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Device: {device}")
    print(f"Image: {arguments.image}")
    print(f"Original image size: {image.size}")
    print(f"Normalized box: {result.box_normalized}")
    print(f"Original box: {result.box_original}")
    print(f"Expanded crop box: {result.crop_box_original}")
    print(f"Crop size: {result.crop.size}")
    print(f"Saved visualization: {arguments.output}")


if __name__ == "__main__":
    main()