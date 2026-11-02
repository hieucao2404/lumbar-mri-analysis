# Teammate guide: paired sagittal–axial inference

## Code structure

| Location | Responsibility |
|---|---|
| `src/models/attention_vgg16_disc.py` | Shared VGG16 and attention over eight sagittal slice logits |
| `src/models/trainable_vgg16_disc.py` | VGG16 encoder, grayscale conversion, normalization |
| `src/models/lstm_attention_vgg16_disc.py` | Optional bidirectional LSTM variant |
| `src/inference/mri_slice_selector.py` | MRI-only middle-window selection with guarded confidence switching |
| `src/inference/herniation_levels.py` | Localization, lumbar levels, bands, and eight-slice crops |
| `src/inference/attention_vgg16_stack.py` | Classifier interface for already prepared stacks |
| `MRI_Lumbar/` | Axial inference source snapshot; see `AXIAL_SOURCE.md` |
| `scripts/run_paired_sagittal_axial.py` | Mendeley adapter, sagittal inference, axial matching, routing, fusion, overlays |
| `scripts/evaluate_lsma_paired_pipeline.py` | Bulk paired evaluation, attaching labels after prediction |
| `scripts/evaluate_mendeley_multislice_transfer.py` | External sagittal attention/LSTM comparison |
| `scripts/evaluate_rsna_five_level_axial_matching.py` | Five-level axial matching evaluation |
| `scripts/train_spider_abnormal_vgg16_attention.py` | Attention training and OOF predictions |
| `scripts/calibrate_spider_abnormal_disc.py` | Threshold selection from training OOF predictions |
| `tests/` | Unit tests, including failure accounting and same-disc comparisons |
| `docs/handoff/` | Stack examples, calibration copy, artifact hashes |
| `outputs/` | Selected runnable checkpoints and evaluation records |

Historical modules may still exist in Git. The active paired entry point is
`scripts.run_paired_sagittal_axial`. The older
`src.inference.abnormal_disc_review` command loads the lightweight CNN.
Three-class experiments do not change the paired runner's binary task.

## Environment and installation

The tested environment is Python 3.14.4, PyTorch 2.11.0 + CUDA 12.8, and
torchvision 0.26.0. Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-dev.txt
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -m pytest -q tests/test_spider_abnormal_disc_cnn.py \
  tests/test_herniation_level_bands.py tests/test_lsma_paired_evaluation.py \
  tests/test_paired_pipeline.py
```

For CPU inference, install the CPU PyTorch build and use `--device cpu`.
Requirements use headless OpenCV; no desktop image viewer is required. Choose
compatible Python and PyTorch builds for other platforms.

The historical `tests/test_split.py` loads a local LSS dataset during collection.
It requires that dataset; the focused command above does not. To run the other
historical tests without it, use `python -m pytest -q tests --ignore=tests/test_split.py`.

## Included checkpoints

| Role | Path |
|---|---|
| Sagittal localizer | `outputs/spider_disc_multitask_coord_v2/best_model.pt` |
| Sagittal attention classifier | `outputs/spider_abnormal_vgg16_attention_guarded_v1/fixed.pt` |
| Fixed threshold | `outputs/spider_abnormal_vgg16_attention_guarded_v1/calibration.json` |
| Axial segmentation | `MRI_Lumbar/cnn_segmentation/models/checkpoints_4cls/lenet_unet_best.pt` |
| Axial ROI detector | `MRI_Lumbar/cnn/cnn_roi_detection/outputs_256/best.pt` |

Verify the exact files after cloning:

```bash
python -m scripts.check_pipeline_artifacts
```

All five entries must print `ok`. SHA-256 hashes are versioned in
`docs/handoff/paired_artifacts.json`. Threshold **0.23432311415672302** was selected
from SPIDER training OOF predictions. Keep it fixed for evaluation. Inference
loads saved VGG16 weights with `pretrained=False`. The optional LSTM and legacy
lightweight CNN outputs are also included; see `outputs/README.md`.

## Smoke test without MRI

This checks installation, weights, and JSON wiring. A synthetic score has no
diagnostic meaning. Use a fresh output filename each time:

```bash
mkdir -p outputs/handoff_smoke
python -c 'import numpy as np; np.save("outputs/handoff_smoke/blank.npy", np.zeros((8, 1, 160, 160), dtype=np.uint8))'
python -m src.inference.attention_vgg16_stack \
  --input docs/handoff/smoke_input.example.json \
  --checkpoint outputs/spider_abnormal_vgg16_attention_guarded_v1/fixed.pt \
  --calibration outputs/spider_abnormal_vgg16_attention_guarded_v1/calibration.json \
  --output outputs/handoff_smoke/result.json --device cpu
```

Each stack is `[8, 1, 160, 160]` `uint8`, with eight actual slices in scan order.
The runner divides by 255; RGB conversion and ImageNet normalization happen
inside the model. See the [stack contract](handoff/README.md) for crop details.

## Run a paired examination

Obtain the original 515-patient Mendeley archive and place it at
`data/raw/mendeley_lumbar/MRI_Data.zip`, preserving its directory structure and
`.ima` files. LSMA's selected PNGs alone cannot supply eight sagittal slices or
DICOM geometry.

```bash
# Sequential axial analysis of sagittal-flagged levels.
python -m scripts.run_paired_sagittal_axial \
  --patient-id 1 --device cuda --serial \
  --output outputs/team_serial/patient_0001

# Same routing, with up to three axial level workers.
python -m scripts.run_paired_sagittal_axial \
  --patient-id 1 --device cuda --concurrent \
  --output outputs/team_concurrent/patient_0001

# Audit all supported levels that have an axial match.
python -m scripts.run_paired_sagittal_axial \
  --patient-id 1 --device cuda \
  --output outputs/team_all_levels/patient_0001
```

`--serial` and `--concurrent` are mutually exclusive. The default mode,
`all_levels_pilot`, is sequential. Concurrent mode uses host threads, shared
GPU models, and a lock for Matplotlib. CUDA operations can still serialize on
the shared default stream. All modes first finish sagittal inference because
disc coordinates are needed for axial matching. Saved timing repeats showed
only a small concurrency difference, not a substantial speed improvement.

Outputs include `result.json`, axial input PNGs, and overlays. Use unique
directories: the pilot can overwrite existing files. Bulk evaluators refuse
an existing output directory.

## Fusion and output fields

Join records by the same examination and exact lumbar level. Preserve
`study_uid`, the sagittal series, and selected axial SOP instance UIDs.
Current paired JSON is a research interface without a production schema
version; consumers should check the named fields and `status`.

| Condition | `final_label` | `final_grade` |
|---|---|---|
| Sagittal below threshold | `normal` | `null` |
| Sagittal flagged, valid axial grade 0 | `normal` | `0` |
| Sagittal flagged, valid axial grade 1–3 | `abnormal` | `1`–`3` |
| Sagittal unavailable or axial matching/analysis fails | `indeterminate` | `null` |

Axial support currently covers L3–L4, L4–L5, and L5–S1. A sagittal positive at
either upper level remains indeterminate. Extending the support set alone does
not validate upper-level grading. A sagittal pixel box must be transformed
through DICOM patient coordinates; it is not an axial pixel box. Axial grades
are geometric research outputs, not expert-confirmed bulge/herniation subtypes.

## LSMA evaluation

The default label path is
`data/raw/LSMA-PQR Lumbar Spine Multi-view Annotations with/LSMA-PQR Lumbar Spine Multi-view Annotations with/structured_notes.csv`.
Pass `--notes` or `--archive` for other locations:

```bash
python -m scripts.evaluate_lsma_paired_pipeline --device cuda \
  --output outputs/team_lsma_pilot --limit 2
python -m scripts.evaluate_lsma_paired_pipeline --device cuda \
  --output outputs/team_lsma_full
```

The evaluator reuses models, grades the two labelled lower levels, and skips
overlay drawing. It exports per-patient JSON, `predictions.csv`, and `report.json`.
It attaches labels after prediction: `None` → normal; `Diffuse Bulge` and
`Central Protrusion` → abnormal. The historical None/Diffuse Bulge subset is
also reported.

The saved run decided 945/1,030 discs: F1 0.709, precision 0.584, recall 0.902.
There were 85 indeterminate discs; full-cohort abnormal detection was 431/509 =
0.847. The report compares sagittal and fused decisions on the same discs.
Fused AP is not calculated for the categorical fusion rule. These exploratory
report-derived labels do not validate clinical grades. Upper-level `Involvement`
fields cannot substitute for per-level disc-type labels.

## Adapting another dataset

Implement the acquisition adapter around the existing models: select sagittal
T2 and same-study axial T2, verify orientation and spacing, create the expected
stacks, then match disc centres to axial planes. Current matching takes up to
three in-image planes within 8 mm. It chooses the largest qualifying axial
series; acquisitions with multiple axial series may require a new selection
adapter.

Keep labels out of inference. Verify lumbar numbering and visibility. Preserve
missing results as unavailable/indeterminate. Reserve evaluation patients absent
from both models' training and threshold selection. Report classification,
level assignment, boxes, axial matching, and coverage separately. RSNA stenosis
references can evaluate matching but cannot provide bulge/herniation truth.

## Git and teammate updates

`.gitignore` allows selected shared outputs listed in `outputs/README.md`.
Datasets, local clinical cases, crops, overlays, and new runs stay local. Ignore
rules do not remove already tracked historical files. Stage explicit paths and
inspect `git diff --cached --stat`. The axial directory in this revision is a
source snapshot, not a submodule. Preserve its upstream attribution when updating
it, and rerun tests plus known-case checks after integration changes.
