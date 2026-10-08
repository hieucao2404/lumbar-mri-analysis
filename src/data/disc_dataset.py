from PIL import Image, ImageOps
import numpy as np
import torch
from torch.utils.data import Dataset

class DiscDataset(Dataset):
    def __init__(self, rows, image_size=128):
        self.rows = rows
        self.image_size = image_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        with Image.open(row["image_path"]) as image:
            image = image.convert("L")
            image = image.crop(row["box"])
            image = ImageOps.pad(
                image,
                (self.image_size, self.image_size),
                method=Image.Resampling.BILINEAR,
            )

        pixels = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).unsqueeze(0)

        # PyTorch class indices are 0..4; the clinical grades are I..V
        grade = (
            row["pfirrman"] - 1
            if row["pfirrman"] != -1
            else -1
        )

        return (
            tensor,
            torch.tensor(grade, dtype=torch.long),
            torch.tensor(row["bulge"], dtype=torch.float32),
            torch.tensor(row["herniation"], dtype=torch.float32),
        )
