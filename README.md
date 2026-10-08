Lumbar MRI Coarse-to-Fine Detection

This project implements the proposed two-stage pipeline:

flowchart TD
    A["Original MRI"] --> B["Letterbox to 128×128"]
    B --> C["Stage 1: whole-lumbar localizer"]
    C --> D["Map box to original coordinates"]
    D --> E["Crop original MRI with margin"]
    E --> F["Letterbox crop to 320×320"]
    F --> G["Stage 2: anatomical-region detector"]
    G --> H["Map regional boxes back to original MRI"]

Stage 1 answers only: where is the whole lumbar spine? Stage 2 examines the
high-resolution crop and detects smaller annotated regions such as neural
foramina. Later ROI heads can classify lumbar level, side, condition, and
severity.

Important dataset requirement

The first model requires a whole-lumbar bounding box for every training image.
The LSS foraminal-stenosis boxes cannot automatically serve as whole-lumbar
labels. If whole-lumbar boxes are unavailable, create them manually, derive a
coarse box from trustworthy anatomical masks, or skip Stage 1 and use a
deterministic anatomical crop.

Do not train Stage 2 from Stage 1 predictions at first. Train it using the
ground-truth lumbar box so its errors remain independent of Stage 1 errors.

Input-size decision

Start the localizer at 128×128.

Treat 64×64 as an ablation experiment, not an assumption.

Keep 128×128 only if lumbar-localization recall and crop coverage remain
acceptable on the validation set.

Stage 2 starts at 320×320 because its targets are smaller.

The final choice must be supported by an experiment comparing at least
64, 128, and optionally 192 pixels. Measure accuracy and latency together.

Project layout

lumbar-cascade/
├── configs/
│   └── default.json
├── data/
│   └── README.md
├── scripts/
│   ├── inspect_sample.py
│   └── train_detector.py
├── src/lumbar_cascade/
│   ├── datasets.py
│   ├── geometry.py
│   ├── models.py
│   └── pipeline.py
└── tests/
    └── test_geometry.py

Coordinate spaces

The code keeps these spaces separate:

original: coordinates in the source MRI.

stage1: coordinates in the letterboxed 128×128 image.

crop: coordinates relative to the high-resolution lumbar crop.

stage2: coordinates in the letterboxed 320×320 crop.

Every resize returns metadata that can transform boxes in both directions.
Never multiply a Stage 1 box by original_size / 128 unless the image was
stretched without padding. Letterboxing requires both scale removal and padding
removal.

Manifest format

Create data/manifest.json:

[
  {
    "image": "data/images/patient_001_slice_01.png",
    "patient_id": "patient_001",
    "lumbar_box": [80, 40, 420, 470],
    "regions": [
      {"box": [130, 150, 180, 205], "label": 1},
      {"box": [315, 150, 365, 205], "label": 1}
    ]
  }
]

All manifest boxes use absolute original-image xyxy coordinates:
[x1, y1, x2, y2].

Setup

Python 3.11 or 3.12 is recommended because the scientific Python ecosystem may
lag behind Python 3.14.

cd lumbar-cascade
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest

First milestones

Fill the manifest for 20 samples.

Run the inspection script and verify every box visually.

Pass all geometry tests.

Split data by patient_id, never by individual slice.

Train Stage 1 using whole-lumbar boxes.

Train Stage 2 using ground-truth lumbar crops.

Evaluate Stage 2 using predicted Stage 1 crops.

Compare Stage 1 input sizes using the same patient split.

Commands

Inspect the coordinate pipeline:

python scripts/inspect_sample.py \
  --manifest data/manifest.json \
  --index 0 \
  --output runs/inspection.png

Train the whole-lumbar localizer:

python scripts/train_detector.py \
  --manifest data/manifest.json \
  --stage 1 \
  --epochs 20 \
  --output checkpoints/stage1.pt

Train the detailed region detector from ground-truth lumbar crops:

python scripts/train_detector.py \
  --manifest data/manifest.json \
  --stage 2 \
  --epochs 30 \
  --output checkpoints/stage2.pt

What to study alongside the code

Letterboxing: aspect-ratio-preserving resize, padding, affine coordinate
transformations.

Stage 1/2 models: CNN backbone, feature maps, region proposals, box regression.

Evaluation: IoU, recall, precision, mAP, and error propagation in cascaded
models.