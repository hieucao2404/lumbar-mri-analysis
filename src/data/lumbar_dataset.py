from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from collections.abc import Callable

from src.data.lumbar_bbox import lumbar_bbox_from_xml
from src.preprocessing.resize import (
    letterbox_resize,
    map_box_to_letterbox,
)
from src.preprocessing.tensor import image_to_tensor

class LumbarLocalizationDataset(Dataset):
    """
    Dataset for whole-lumbar localization using middle sagittal slices.

    Image output:
        Float tensor [1, height, width], range[0, 1]

    Target output: 
        Normalized [center_x, center_y, width, height]
    """

    def __init__(
        self,
        dataset_root: str | Path,
        target_size: tuple[int, int] = (128, 128),
        augmentation: Callable | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.target_size = target_size
        self.augmentation = augmentation

        segmentation_root = self.dataset_root / "Segmentation"
        self.image_dir = segmentation_root / "Middle_Slice"
        self.annotation_dir = segmentation_root / "Masks"

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )

        if not self.annotation_dir.is_dir():
            raise FileNotFoundError(
                f"Annotation  directory not found: {self.annotation_dir}"
            )

        self.samples: list[tuple[Path, Path]] = []

        for xml_path in sorted(self.annotation_dir.glob("*.xml")):
            image_path = self.image_dir / f"{xml_path.stem}.png"

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"No matching image for {xml_path.name}: {image_path}"
                )

            self.samples.append((image_path, xml_path))

        if not self.samples:
            raise RuntimeError(
                f"No XML annotations found in {self.annotation_dir}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int,) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, xml_path = self.samples[index]

        with Image.open(image_path) as source:
            image = source.convert("L")

        original_box = lumbar_bbox_from_xml(xml_path)

        resized_image, metadata = letterbox_resize(
            image, 
            target_size=self.target_size,
        )

        letterbox_box = map_box_to_letterbox(
            original_box,
            metadata,
        )

        if self.augmentation is not None:
            resized_image, letterbox_box = self.augmentation(
                resized_image,
                letterbox_box,
            )

        x1, y1, x2, y2 = letterbox_box

        target_width, target_height = self.target_size

        center_x = ((x1 + x2) / 2.0) / target_width
        center_y = ((y1 + y2) / 2.0) / target_height
        box_width = (x2 - x1) / target_width
        box_height = (y2 - y1) / target_height

        target = torch.tensor(
            [center_x, center_y, box_width, box_height],
            dtype=torch.float32,
        )

        image_tensor = image_to_tensor(resized_image)

        return image_tensor, target