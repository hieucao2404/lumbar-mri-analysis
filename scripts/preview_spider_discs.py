import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import SimpleITK as sitk


manifest = Path("data/processed/spider_discs.csv")
output = Path("outputs/spider/disc_preview.png")
output.parent.mkdir(parents=True, exist_ok=True)

with manifest.open(newline="") as file:
    rows = list(csv.DictReader(file))

selected = []
seen = set()

def add_first(predicate):
    for row in rows:
        key = (row["series_name"], row["ivd_label"])
        if predicate(row) and key not in seen:
            selected.append(row)
            seen.add(key)
            return

# Check both NumPy slice axes and examples of the target labels.
add_first(lambda r: r["series_name"] == "1_t2")
add_first(lambda r: r["series_name"] == "107_t2")
add_first(lambda r: r["series_name"] == "35_t2")
add_first(lambda r: r["series_name"] == "64_t2")
add_first(lambda r: r["pfirrmann_grade"] == "1")
add_first(lambda r: r["pfirrmann_grade"] == "5")
add_first(lambda r: r["disc_herniation"] == "1")
add_first(
    lambda r: r["disc_bulging"] == "1"
    and r["disc_herniation"] == "1"
)

fig, axes = plt.subplots(
    len(selected), 2,
    figsize=(10, 3 * len(selected)),
    squeeze=False,
)

for index, row in enumerate(selected):
    image_volume = np.flip(
        sitk.GetArrayFromImage(
            sitk.DICOMOrient(
                sitk.ReadImage(row["image_path"]),
                "LPS",
            )
        ),
        axis=0,
    )

    mask_volume = np.flip(
        sitk.GetArrayFromImage(
            sitk.DICOMOrient(
                sitk.ReadImage(row["mask_path"]),
                "LPS",
            )
        ),
        axis=0,
    )

    axis = int(row["sagittal_axis"])
    slice_index = int(row["sagittal_slice"])
    image = np.take(image_volume, slice_index, axis=axis)
    disc_mask = (
        np.take(mask_volume, slice_index, axis=axis)
        == int(row["mask_value"])
    )

    x1, y1, x2, y2 = (
        int(row[name]) for name in ("x1", "y1", "x2", "y2")
    )
    height, width = image.shape

    # Show tissue around the disc, including the posterior boundary.
    pad_x = x2 - x1
    pad_y = y2 - y1
    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(width, x2 + pad_x)
    crop_y2 = min(height, y2 + pad_y)

    positive = image[image > 0]
    if positive.size:
        low, high = np.percentile(positive, (1, 99))
    else:
        low, high = float(image.min()), float(image.max())

    if high <= low:
        high = low + 1

    shown = np.clip((image - low) / (high - low), 0, 1)

    full_ax, crop_ax = axes[index]
    full_ax.imshow(shown, cmap="gray")
    full_ax.contour(disc_mask, levels=[0.5], colors="lime", linewidths=0.7)
    full_ax.add_patch(
        Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor="red", linewidth=1.5,
        )
    )
    full_ax.set_title(
        f'{row["series_name"]}, disc {row["ivd_label"]}, '
        f'axis {axis}, slice {slice_index}'
    )

    crop_ax.imshow(
        shown[crop_y1:crop_y2, crop_x1:crop_x2],
        cmap="gray",
    )
    crop_ax.set_title(
        f'Pfirrmann {row["pfirrmann_grade"]} | '
        f'bulge {row["disc_bulging"]} | '
        f'herniation {row["disc_herniation"]}'
    )

    full_ax.axis("off")
    crop_ax.axis("off")

fig.tight_layout()
fig.savefig(output, dpi=150)
plt.close(fig)
print("Saved:", output)