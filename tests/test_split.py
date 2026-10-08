from pathlib import Path

from torch.utils.data import DataLoader, Subset

from src.data.lumbar_dataset import LumbarLocalizationDataset
from src.data.splits import (
    create_split_indices,
    save_split_manifest,
)

SEED = 42

dataset = LumbarLocalizationDataset(
    "data/raw/lss_mri_dataset",
    target_size=(128, 128),
)

splits = create_split_indices(
    number_of_samples=len(dataset),
    train_fraction=0.8,
    validation_fraction=0.1,
    seed=SEED,
)

sample_names = [
    image_path.stem
    for image_path, _ in dataset.samples
]

save_split_manifest(
    splits=splits,
    sample_names=sample_names,
    output_path="data/splits/lumbar_localizer.json",
    seed=SEED,
)

train_set = Subset(dataset, splits["train"])
validation_set = Subset(dataset, splits["validation"])
test_set = Subset(dataset, splits["test"])

print("Training samples:  ", len(train_set))
print("Validation samples:", len(validation_set))
print("Test samples:      ", len(test_set))

train_indices = set(splits["train"])
validation_indices = set(splits["validation"])
test_indices = set(splits["test"])

assert len(train_set) == 400
assert len(validation_set) == 50
assert len(test_set) == 50

assert train_indices.isdisjoint(validation_indices)
assert train_indices.isdisjoint(test_indices)
assert validation_indices.isdisjoint(test_indices)

all_indices = train_indices | validation_indices | test_indices
assert len(all_indices) == len(dataset)

train_loader = DataLoader(
    train_set,
    batch_size=16,
    shuffle=True,
)

validation_loader = DataLoader(
    validation_set,
    batch_size=16,
    shuffle=False,
)

test_loader = DataLoader(
    test_set,
    batch_size=16,
    shuffle=False,
)

images, targets = next(iter(train_loader))

print("Training batch:", images.shape, targets.shape)
print("Manifest: data/splits/lumbar_localizer.json")
print("\nPatient-level split passed.")

