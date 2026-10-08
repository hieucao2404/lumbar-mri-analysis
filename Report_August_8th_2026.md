# Lumbar MRI CNN Progress Report

## 1. Project Goal

The current sagittal MRI pipeline is being developed to produce findings for each lumbar disc level from **L1–L2 to L5–S1**.

The intended output for each level includes:

- Lumbar position
- Pfirrmann degeneration grade
- Spinal-canal stenosis severity
- Posterior disc finding

---

## 2. Dataset Preparation

The **SPIDER lumbar MRI dataset** is currently used for sagittal processing.

Completed preprocessing includes:

- Selecting sagittal T2 MRI series
- Correcting image orientation
- Normalizing MRI intensity
- Resizing images to \(256 \times 256\)
- Creating disc segmentation masks
- Creating lumbar-level masks
- Generating \(160 \times 160\) disc crops
- Separating patients into training and validation sets

### Dataset split

| Item | Count |
|---|---:|
| Sagittal T2 studies | 210 |
| Training patients | 172 |
| Validation patients | 38 |
| Disc samples | 1,452 |
| Training disc samples | 1,187 |
| Validation disc samples | 265 |
| Patient overlap | 0 |

---

## 3. Disc Localization

A lightweight U-Net was trained to segment intervertebral discs from sagittal MRI images.

### Validation results

| Metric | Result |
|---|---:|
| Dice score | 0.8525 |
| Precision | 0.8206 |
| Recall | 0.9003 |
| Instance-detection F1 | 0.9205 |
| Mean matched-mask IoU | 0.7758 |

The model can:

- Identify disc pixels
- Separate the mask into individual disc components
- Produce a bounding box for each detected disc
- Generate disc regions for later classification

---

## 4. Lumbar-Level Identification

The SPIDER anatomical ordering was verified as:

| SPIDER label | Lumbar level |
|---:|---|
| 1 | L5–S1 |
| 2 | L4–L5 |
| 3 | L3–L4 |
| 4 | L2–L3 |
| 5 | L1–L2 |
| 6 and above | Upper discs |

All **1,242 adjacent disc pairs** followed the expected anatomical order.

### Position-based assignment results

Detected discs are ordered vertically, and the five lowest valid discs are assigned to the lumbar levels.

| Metric | Result |
|---|---:|
| End-to-end level accuracy | 0.8444 |
| Assignment precision | 0.8686 |
| All five levels correct | 26/36 studies |
| Complete-study accuracy | 0.7222 |

This is currently the best-performing lumbar-level assignment method.

---

## 5. Coordinate-Aware Multitask U-Net

A multitask U-Net was developed with:

1. A binary disc-segmentation head
2. A lumbar-level classification head
3. Normalized \(x\)- and \(y\)-coordinate inputs

### Pixel-level validation results

| Metric | Result |
|---|---:|
| Binary Dice | 0.8589 |
| Binary precision | 0.8316 |
| Binary recall | 0.8880 |
| Level accuracy | 0.6945 |
| Level macro F1 | 0.6880 |

### Component-level results

| Metric | Result |
|---|---:|
| Disc localization F1 | 0.9271 |
| Mean matched IoU | 0.7771 |
| Matched-disc level accuracy | 0.7074 |
| Matched-disc level macro F1 | 0.6839 |
| End-to-end lumbar accuracy | 0.6045 |

The learned level head provides useful class probabilities, but the position-based method currently produces better final level assignments.

---

## 6. Pfirrmann Grade Classification

A CNN was trained to classify each detected disc into:

- Grade I
- Grade II
- Grade III
- Grade IV
- Grade V

### Results on 228 paired validation discs

| Crop source | Accuracy | Macro F1 | Grade MAE |
|---|---:|---:|---:|
| Expert-defined crops | 0.4649 | 0.4451 | 0.6360 |
| Automatically detected crops | 0.4605 | 0.4342 | 0.6316 |

The small difference between the two crop sources shows that automatic localization does not significantly reduce classification performance. However, the Pfirrmann classifier still requires improvement.

---

## 7. Foraminal-Stenosis Classification

A separate CNN was trained using the LSS/AISSLab dataset to classify:

- Normal
- Mild
- Moderate
- Severe

### Test results

| Metric | Result |
|---|---:|
| Accuracy | 0.6963 |
| Macro F1 | 0.5081 |
| Normal F1 | 0.859 |
| Mild F1 | 0.345 |
| Moderate F1 | 0.313 |
| Severe F1 | 0.515 |

This model is an auxiliary experiment and has not yet been integrated into the sagittal SPIDER pipeline.

---

## 8. Current Sagittal Pipeline

```text
Sagittal T2 MRI
      ↓
Orientation correction
      ↓
Intensity normalization
      ↓
Resize to 256 × 256
      ↓
U-Net disc segmentation
      ↓
Individual disc components
      ↓
Bounding-box generation
      ↓
Lumbar-level assignment
      ↓
160 × 160 disc crops
      ↓
Pfirrmann classification
      ↓
Per-level result
```

---

## 9. Current Project Status

### Completed

- Sagittal T2 MRI preprocessing
- Patient-level dataset splitting
- Disc semantic segmentation
- Disc-component extraction
- Bounding-box generation
- Lumbar-level mapping verification
- Position-based level assignment
- Coordinate-aware level prediction
- Pfirrmann classification baseline
- Expert-crop versus detected-crop evaluation
- Foraminal-severity classification experiment

### Not yet completed

- Posterior disc classification:
  - None
  - Bulge
  - Suspected herniation
- Sagittal spinal-canal stenosis classification
- Sequence-constrained level correction
- Complete JSON result generation
- Axial MRI processing
- Laterality prediction
- Model quantization
- Verilog CNN acceleration
- FPGA deployment and performance measurement

---

## 10. Next Step

The next recommended step is to combine:

- The strong position-based lumbar assignment
- The probabilities from the coordinate-aware level head
- The required anatomical order

After level assignment is stabilized, the detected disc crops can be passed to the Pfirrmann and posterior-disc classifiers to generate output such as:

```json
{
  "study_id": "example_001",
  "view": "sagittal",
  "levels": [
    {
      "position": "L4-L5",
      "disc_degeneration": "Pfirrmann IV",
      "spinal_canal_stenosis": "unavailable",
      "posterior_disc_finding": "unavailable"
    }
  ]
}
```

At the current stage, the system can reliably localize discs and assign lumbar levels, while condition classification and final pipeline integration remain under development.