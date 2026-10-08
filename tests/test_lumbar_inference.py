import pytest
import torch
from PIL import Image
from torch import nn

from src.inference.lumbar_localizer import (
    expand_box,
    localize_lumbar,
    normalized_cxcywh_to_xyxy,
)


class FixedPredictionModel(nn.Module):
    """
    Test model that always returns the same normalized box.
    """

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        prediction = torch.tensor(
            [0.5, 0.5, 0.5, 0.5],
            dtype=torch.float32,
            device=images.device,
        )

        return prediction.unsqueeze(0).repeat(
            images.shape[0],
            1,
        )


def test_normalized_box_conversion() -> None:
    box = normalized_cxcywh_to_xyxy(
        box=(0.5, 0.5, 0.5, 0.5),
        image_size=(128, 128),
    )

    assert box == pytest.approx(
        (32.0, 32.0, 96.0, 96.0)
    )


def test_expand_box() -> None:
    expanded = expand_box(
        box=(50.0, 20.0, 150.0, 80.0),
        image_size=(200, 100),
        margin_fraction=0.1,
    )

    assert expanded == (40, 14, 160, 86)


def test_localization_maps_back_to_original_image() -> None:
    image = Image.new(
        mode="L",
        size=(200, 100),
        color=128,
    )

    model = FixedPredictionModel()
    device = torch.device("cpu")

    result = localize_lumbar(
        image=image,
        model=model,
        device=device,
        target_size=(128, 128),
        margin_fraction=0.0,
    )

    assert result.box_normalized == pytest.approx(
        (0.5, 0.5, 0.5, 0.5)
    )

    assert result.box_letterboxed == pytest.approx(
        (32.0, 32.0, 96.0, 96.0)
    )

    # Original 200×100 image becomes 128×64 with 32 pixels
    # of vertical padding. The predicted box maps to the
    # center 100×100 region of the original image.
    assert result.box_original == pytest.approx(
        (50.0, 0.0, 150.0, 100.0)
    )

    assert result.crop_box_original == (
        50,
        0,
        150,
        100,
    )

    assert result.crop.size == (100, 100)


def test_negative_margin_is_rejected() -> None:
    with pytest.raises(ValueError):
        expand_box(
            box=(10.0, 10.0, 20.0, 20.0),
            image_size=(100, 100),
            margin_fraction=-0.1,
        )