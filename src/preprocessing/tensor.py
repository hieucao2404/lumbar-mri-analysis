import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert an MRI image into a PyTorch tensor.

    Output format:
        [channels, height, width]

    Greyscale MRI output:
        [1, height, width]

    Pixel values are scaled from [0, 255] to [0.0, 1.0]
    """
    grayscale_image = image.convert("L")

    tensor = pil_to_tensor(grayscale_image)
    tensor = tensor.to(dtype=torch.float32)
    tensor = tensor / 255.0

    return tensor