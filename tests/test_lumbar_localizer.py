import torch

from src.models.lumbar_localizer import LumbarLocalizer


def test_lumbar_localizer_output():
    model = LumbarLocalizer()

    # Two grayscale MRI images.
    images = torch.rand(2, 1, 128, 128)

    boxes = model(images)

    assert boxes.shape == (2, 4)
    assert boxes.dtype == torch.float32

    # Sigmoid should keep every coordinate between 0 and 1.
    assert torch.all(boxes >= 0.0)
    assert torch.all(boxes <= 1.0)