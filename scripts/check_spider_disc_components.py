from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.data.spider_disc_localization_dataset import (
    SpiderDiscLocalizationDataset,
)
from src.models.spider_disc_unet import SpiderDiscUNet
from src.postprocess.disc_components import extract_disc_boxes


manifest = Path(
    "data/processed/spider_disc_localization/manifest.csv"
)
checkpoint_path = Path(
    "outputs/spider_disc_localization_softdice_v1/best_model.pt"
)

dataset = SpiderDiscLocalizationDataset(
    manifest,
    subset="validation",
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False,
)

model = SpiderDiscUNet().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

predicted_counts = Counter()
reference_counts = Counter()
mismatches = []

with torch.no_grad():
    for index in range(len(dataset)):
        image, target = dataset[index]

        logits = model(image.unsqueeze(0).to(device))
        probability = (
            torch.sigmoid(logits)[0, 0]
            .cpu()
            .numpy()
        )

        predicted_boxes = extract_disc_boxes(
            probability,
            threshold=0.5,
            min_area=25,
        )

        # Diagnostic only: count regions in the exported binary mask.
        reference_boxes = extract_disc_boxes(
            target[0].numpy(),
            threshold=0.5,
            min_area=25,
        )

        predicted_count = len(predicted_boxes)
        reference_count = len(reference_boxes)

        predicted_counts[predicted_count] += 1
        reference_counts[reference_count] += 1

        if predicted_count != reference_count:
            mismatches.append({
                "index": index,
                "patient_id": dataset.rows[index]["patient_id"],
                "reference": reference_count,
                "predicted": predicted_count,
            })

print("Validation images:", len(dataset))
print("Predicted component counts:", dict(sorted(predicted_counts.items())))
print("Reference component counts:", dict(sorted(reference_counts.items())))
print("Images with different counts:", len(mismatches))

for item in mismatches:
    print(item)