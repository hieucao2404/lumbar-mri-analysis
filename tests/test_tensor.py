import torch
from PIL import Image

from src.preprocessing.tensor import image_to_tensor


def test_image_to_tensor():
    image = Image.new(
        mode="L",
        size=(128, 128),
        color=255,
    )

    tensor = image_to_tensor(image)

    assert tensor.shape == (1, 128, 128)
    assert tensor.dtype == torch.float32
    assert tensor.min().item() == 1.0
    assert tensor.max().item() == 1.0