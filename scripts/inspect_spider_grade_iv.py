from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.data.spider_dataset import SpiderPfirrmannDataset
from src.models.spider_model import SpiderPfirrmannCNN


manifest = Path("data/processed/spider_crops/crops.csv")
checkpoint = Path("outputs/spider_pfirrmann_baseline/best_model.pt")
output = Path("outputs/spider_pfirrmann_baseline/grade_iv_review.png")

dataset = SpiderPfirrmannDataset(manifest, "validation")

model = SpiderPfirrmannCNN()
model.load_state_dict(
    torch.load(checkpoint, map_location="cpu", weights_only=True)
)
model.eval()

# Dataset target 3 means Pfirrmann Grade IV.
groups = {2: [], 3: [], 4: []}  # Predicted III, IV, V
all_predictions = Counter()

with torch.inference_mode():
    for start in range(0, len(dataset), 16):
        indices = range(start, min(start + 16, len(dataset)))
        samples = [dataset[i] for i in indices]

        images = torch.stack([image for image, _ in samples])
        targets = [int(target) for _, target in samples]

        probabilities = model(images).softmax(dim=1)
        predictions = probabilities.argmax(dim=1)

        for batch_index, dataset_index in enumerate(indices):
            if targets[batch_index] != 3:
                continue

            predicted = int(predictions[batch_index])
            all_predictions[predicted] += 1

            if predicted in groups:
                groups[predicted].append({
                    "index": dataset_index,
                    "image": images[batch_index, 0].numpy(),
                    "confidence": float(
                        probabilities[batch_index, predicted]
                    ),
                })

print("True Grade IV predictions (IDs 0–4):")
print(dict(sorted(all_predictions.items())))

# Show six confident examples from each outcome.
fig, axes = plt.subplots(3, 6, figsize=(16, 9))
grade_names = {2: "III", 3: "IV", 4: "V"}

for row, predicted in enumerate((2, 3, 4)):
    examples = sorted(
        groups[predicted],
        key=lambda item: item["confidence"],
        reverse=True,
    )[:6]

    for column, ax in enumerate(axes[row]):
        ax.axis("off")
        if column >= len(examples):
            continue

        item = examples[column]
        ax.imshow(item["image"], cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"True IV → {grade_names[predicted]}\n"
            f"index {item['index']}, score {item['confidence']:.2f}",
            fontsize=9,
        )

fig.suptitle("SPIDER validation: Pfirrmann Grade IV crops")
fig.tight_layout()
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=160)
plt.close(fig)

print("Saved:", output)