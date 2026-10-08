from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from src.models.lumbar_localizer import LumbarLocalizer
from src.preprocessing.resize import (
    letterbox_resize,
    map_box_to_original,
)
from src.preprocessing.tensor import image_to_tensor


Box = tuple[float, float, float, float]
IntegerBox = tuple[int, int, int, int]

@dataclass(frozen=True)
class LumbarLocalizationResult:
    """
    Complete result produced by the Stage 1 lumbar localizer
    """

    box_normalized: Box
    box_letterboxed: Box
    box_original: Box
    crop_box_original: IntegerBox
    crop: Image.Image

def load_lumbar_localizer(
    checkpoint_path: str | Path,
    device: torch.device,
) -> LumbarLocalizer:
    """
    Load the frozen Stage 1 model checkpoint.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model = LumbarLocalizer()
    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    return model

def normalized_cxcywh_to_xyxy(
    box: Box,
    image_size: tuple[int, int],
) -> Box:
    """
    Convert a normalized [cx, cy, width, height] box into
    pixel-based [x1, y1, x2, y2] coordinates
    """
    center_x, center_y, width, height = box
    image_width, image_height = image_size

    center_x *= image_width
    center_y *= image_height
    width *= image_width
    height *= image_height

    x1 = center_x - width / 2
    y1 = center_y - height / 2
    x2 = center_x + width / 2
    y2 = center_y + height / 2

    x1 = max(0.0, min(x1, image_width))
    y1 = max(0.0, min(y1, image_height))
    x2 = max(0.0, min(x2, image_width))
    y2 = max(0.0, min(y2, image_width))

    return x1, y1, x2, y2


def expand_box(
    box: Box,
    image_size: tuple[int, int],
    margin_fraction: float,
) -> IntegerBox:
    """
    Expand a box using a fraction of its own width and height.

    The expanded box is cliped to the image boundaries.
    """
    if margin_fraction < 0:
        raise ValueError("margin_fraction cannot be negative.")

    x1, y1, x2, y2 = box
    image_width, image_height = image_size

    box_width = x2 - x1
    box_height = y2 - y1

    margin_x = box_width * margin_fraction
    margin_y = box_height * margin_fraction

    expanded_x1 = round(max(0.0, x1 - margin_x))
    expanded_y1 = round(max(0.0, y1 - margin_y))
    expanded_x2 = round(min(image_width, x2 + margin_x))
    expanded_y2 = round(min(image_height, y2 + margin_y))

    if expanded_x2 <= expanded_x1:
        raise ValueError("Predicted crop has invalid width.")

    if expanded_y2 <= expanded_y1:
        raise ValueError("Predicted crop has invalid height.")

    return (
        expanded_x1,
        expanded_y1,
        expanded_x2,
        expanded_y2,
    )

def localize_lumbar(
    image: Image.Image,
    model: nn.Module,
    device: torch.device,
    target_size: tuple[int, int] = (128, 128),
    margin_fraction: float = 0.20,
) -> LumbarLocalizationResult:
    """
    Locate and crop the whole lumbar region from one MRI image.
    """
    if image.width <= 0 or image.height <= 0:
        raise ValueError("The input image must not be empty.")

    letterboxed_image, metadata = letterbox_resize(
        image=image,
        target_size=target_size,
        fill=0,
    )

    input_tensor = image_to_tensor(letterboxed_image)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    model.eval()

    with torch.inference_mode():
        prediction = model(input_tensor)[0].detach().cpu()

    normalized_box: Box = tuple(
        float(value) for value in prediction
    )

    letterboxed_box = normalized_cxcywh_to_xyxy(
        normalized_box,
        target_size,
    )

    original_box = map_box_to_original(
        letterboxed_box,
        metadata,
    )

    crop_box = expand_box(
        box=original_box,
        image_size = image.size,
        margin_fraction=margin_fraction,
    )

    crop = image.crop(crop_box)

    return LumbarLocalizationResult(
        box_normalized=normalized_box,
        box_letterboxed=letterboxed_box,
        box_original=original_box,
        crop_box_original=crop_box,
        crop=crop,
    )


