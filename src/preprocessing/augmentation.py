import math
import random

from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as functional_transform


class RandomBoxTranslation:
    """
    Randomly translate an image and its bounding box together.

    Translation is limited so the complete box remains inside
    the image.
    """

    def __init__(
        self,
        horizontal_fraction: float = 0.15,
        vertical_fraction: float = 0.10,
        probability: float = 1.0,
    ) -> None:
        if not 0.0 <= horizontal_fraction <= 1.0:
            raise ValueError("Invalid horizontal fraction")

        if not 0.0 <= vertical_fraction <= 1.0:
            raise ValueError("Invalid vertical fraction")

        if not 0.0 <= probability <= 1.0:
            raise ValueError("Invalid probability")

        self.horizontal_fraction = horizontal_fraction
        self.vertical_fraction = vertical_fraction
        self.probability = probability

    def __call__(
        self,
        image: Image.Image,
        box: tuple[float, float, float, float],
    ) -> tuple[Image.Image, tuple[float, float, float, float]]:
        if random.random() > self.probability:
            return image, box

        image_width, image_height = image.size
        x1, y1, x2, y2 = box

        maximum_dx = round(
            image_width * self.horizontal_fraction
        )
        maximum_dy = round(
            image_height * self.vertical_fraction
        )

        minimum_allowed_dx = max(
            -maximum_dx,
            math.ceil(-x1),
        )
        maximum_allowed_dx = min(
            maximum_dx,
            math.floor(image_width - x2),
        )

        minimum_allowed_dy = max(
            -maximum_dy,
            math.ceil(-y1),
        )
        maximum_allowed_dy = min(
            maximum_dy,
            math.floor(image_height - y2),
        )

        dx = random.randint(
            minimum_allowed_dx,
            maximum_allowed_dx,
        )
        dy = random.randint(
            minimum_allowed_dy,
            maximum_allowed_dy,
        )

        translated_image = functional_transform.affine(
            image,
            angle=0.0,
            translate=[dx, dy],
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )

        translated_box = (
            x1 + dx,
            y1 + dy,
            x2 + dx,
            y2 + dy,
        )

        return translated_image, translated_box