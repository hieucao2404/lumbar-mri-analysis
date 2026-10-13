# Whole-MRI, Multi-Spot Herniation Project — Flow Specification

**Date:** 13 October 2026  
**Status:** Proposed new project design  
**Starting point:** Sagittal T2 MRI and reusable components from the current lumbar MRI repository

## 1. Goal

Given one **full sagittal T2 MRI image**, decide whether disc herniation is present. If present, detect **every separate herniation spot**, retain one result per spot, and assign each spot a lumbar level. A study with two spots must return two findings, including when both spots map to the same lumbar level.

For this first design, “whole MRI” means the **whole selected sagittal T2 image**, rather than a crop of one disc. The current project selects one sagittal slice per series. If the intended input is an entire multi-slice or 3D MRI volume, the input definition, labels, and evaluation must be revised before training.

## 2. Required behavior

| Input situation | Expected output |
|---|---|
| No herniation | Whole-image presence result negative; zero spot findings |
| One herniation spot | Presence positive; one spot location and its lumbar level |
| Two herniation spots at different levels | Presence positive; two separate locations and two assigned levels |
| Two separate spots at the same level | Presence positive; two separate findings with the same level |
| Presence positive but detector finds no spot | Mark as positive but unlocalized for review; do not invent a location |
| Spot found but level uncertain | Retain the spot and mark its level unavailable or uncertain |

A global positive/negative classifier cannot count or localize spots by itself. The separate spot detector is a required part of this project.

## 3. Proposed inference flow

1. **Load the full sagittal T2 image.** Record study ID, series ID, selected slice, orientation, and original image size.
2. **Preprocess the image.** Apply the established orientation correction, intensity normalization, and aspect-ratio-preserving letterbox transform. Preserve the transform needed to map coordinates back to the original image.
3. **Segment intervertebral discs.** Run the existing disc U-Net or its multitask successor on the full image. Extract disc masks, components, and boxes.
4. **Classify whole-image herniation presence.** Feed the full preprocessed MRI to a new binary classifier. A disc-mask channel can be tested as an input variant, but the prediction remains one yes/no result for the full image.
5. **If presence is positive, detect all herniation spots.** A separate multi-instance detector returns zero, one, or more boxes or masks with scores. Use duplicate suppression or connected-component rules that preserve genuinely separate spots.
6. **Assign lumbar levels.** Run the current level model and anatomical decoder on the segmented discs. This may run after a positive presence decision, as in the proposed diagram, or in parallel when latency permits.
7. **Associate each spot with a disc and level.** Match by spatial overlap and distance to the segmented disc region. A spot may extend beyond a disc boundary, so the association rule must allow a documented tolerance. Do not collapse two spots merely because they map to the same level.
8. **Return structured results and an overlay.** Express spot coordinates in the original MRI coordinate system and draw each spot separately, with its own ID, score, and level.

The branch relationship is:

Full MRI → preprocessing → disc segmentation → whole-image binary presence decision → multi-spot detector.  
Full MRI and disc masks → level assignment.  
Spot locations + assigned disc levels → one localized output record per spot.

### Gating rule

The proposed runtime detector runs only when the whole-image classifier is positive. This saves work but makes a false negative from the classifier an end-to-end missed case. Evaluation must count these misses. An ungated detector run is a useful comparison to measure the cost of this gate.

## 4. Example output contract

The fields below are a proposed interface, not current production JSON:

~~~json
{
  "study_id": "example_001",
  "series_name": "example_001_t2",
  "view": "sagittal",
  "sequence": "T2",
  "selected_slice": 12,
  "herniation_presence": {
    "label": "positive",
    "probability": 0.91
  },
  "status": "localized",
  "herniation_count": 2,
  "herniations": [
    {
      "spot_id": 1,
      "bounding_box_original_xyxy": [95, 118, 137, 145],
      "spot_probability": 0.88,
      "lumbar_level": "L4-L5",
      "level_probability": 0.82
    },
    {
      "spot_id": 2,
      "bounding_box_original_xyxy": [101, 173, 142, 201],
      "spot_probability": 0.79,
      "lumbar_level": "L5-S1",
      "level_probability": 0.76
    }
  ]
}
~~~

The count equals the number of retained **spot instances**, not the number of positive discs or distinct lumbar levels. The two records may have the same lumbar_level. For a negative result, use herniation_count = 0 and an empty herniations list. For a positive but unlocalized result, preserve the positive presence result, use status = positive_unlocalized, and return an empty list.

Probabilities are model scores; they must not be described as clinical certainty without separate calibration and validation.

## 5. Data and annotations

### 5.1 Labels currently available

The existing [SPIDER disc table](data/processed/spider_discs.csv) contains a binary disc_herniation label for each annotated disc row, its disc box, and its patient/series identity. A whole-image presence target can be **derived** for an image or study by taking the logical OR of its disc_herniation labels, after checking that the labels correspond to the exact selected image or series used as input.

| Existing SPIDER split | Studies | Studies with at least one positive disc row | Studies with at least two positive disc rows | Positive disc rows |
|---|---:|---:|---:|---:|
| Training | 172 | 55 | 6 | 62 |
| Validation | 38 | 10 | 0 | 10 |

These counts come from the processed disc table. The posterior-crop training export has its own smaller sample count because it is a different task-specific export.

A positive **disc row is not a herniation-spot annotation**. The table does not provide a separate box or mask for each herniation focus. Two positive disc rows show two positive discs, but they do not prove two distinct spots in one disc. The existing validation split has no study with two or more positive disc rows, so it cannot demonstrate the requested two-spot behavior.

### 5.2 New annotations required for the spot detector

For each visible herniation spot on the input image, collect:

| Field | Purpose |
|---|---|
| Study/series/slice ID | Link the annotation to the exact image |
| Unique spot ID | Preserve multiple spots, including at the same level |
| Spot box or pixel mask | Train and evaluate localization |
| Associated disc/level | Evaluate the final level assignment |
| Review status | Record ambiguous or ungradable cases |

Before annotation, define whether a contiguous abnormal region counts as one spot, how separate foci are distinguished, and how far beyond the disc a box or mask should extend. Have the annotation protocol reviewed by a qualified imaging mentor. Include examples with zero, one, and multiple spots; multiple spots at the same level are essential.

If only boxes are available, train a multi-object box detector. If pixel masks are available, an instance-segmentation model or a semantic heatmap plus carefully validated component splitting is possible. A disc segmentation mask cannot serve as the ground-truth herniation-spot mask.

## 6. Model boundaries

| Stage | Input | Output | Possible reuse or new work |
|---|---|---|---|
| Preprocessing | Full sagittal T2 image | Normalized, oriented image and coordinate transform | Reuse current preprocessing |
| Disc segmentation | Full preprocessed image | Disc masks/components | Reuse [multitask U-Net](src/models/spider_disc_multitask_unet.py) as a baseline |
| Whole-image classifier | Full image, optionally plus disc mask | One herniation presence score | **New model/training target** |
| Multi-spot detector | Full image and optional disc mask | List of spot boxes/masks and scores | **New model; requires spot annotations** |
| Level assignment | Disc components and learned level evidence | Disc-to-level mapping | Reuse current gated hybrid decoder as a baseline |
| Spot-to-level association | Spot locations and disc-level mapping | Level for each spot | New deterministic matching and uncertainty logic |
| Visualization | Original image and all predictions | Labeled image showing every spot | Extend the current comparison view |

Pfirrmann grading, bulging classification, canal stenosis, and FPGA acceleration are outside the initial herniation-only project deliverable. They can be added later without changing the one-record-per-spot output rule.

## 7. Training plan

1. **Define the input unit and annotation protocol.** Confirm whether the target is one selected 2D sagittal image or a multi-slice study. Review the spot definition with the mentor.
2. **Create patient-level splits before tuning.** Keep patients from crossing training, validation, and test. The current 38-study SPIDER validation set has already influenced previous model and decoder choices; it is not a fresh independent test set.
3. **Prepare whole-image presence labels.** Aggregate verified disc labels per input image and audit positive and negative examples. Check whether the selected sagittal slice visibly supports each label.
4. **Train the whole-image binary classifier.** Establish an always-negative and an always-positive baseline, then a CNN baseline. Select thresholds on validation data with attention to false negatives.
5. **Acquire spot boxes or masks.** Prioritize multiple-spot cases and include a held-out evaluation set containing them.
6. **Train a multi-instance spot detector.** Train with the chosen annotation type. Tune confidence threshold, duplicate suppression, and spot-separation rules on validation data.
7. **Reuse and verify disc-level assignment.** Match each detected spot to a disc, then to the predicted lumbar level. Record uncertain associations.
8. **Evaluate the gated end-to-end system.** Run the whole flow from raw MRI to all returned spot records on every held-out study.
9. **Package the demo.** Save model checkpoints, configurations, per-study JSON, error lists, and visual overlays.

The current two-output posterior CNN is a disc-crop model, not the proposed whole-image binary classifier. Its selected checkpoint detected no herniation positives in the 188-crop validation evaluation at threshold 0.5; it should be treated as prior experimental evidence, not as a validated replacement for the new stages.

## 8. Evaluation plan and acceptance checks

Report metrics with the unit and denominator stated. Keep component metrics separate from end-to-end results.

| Stage | Required measures |
|---|---|
| Whole-image presence | TP, FP, FN, TN; precision, recall, specificity, F1; PR curve/average precision; positive-study support |
| Spot detector | Spot recall and precision at a documented box/mask IoU threshold; AP; false spots per image; count MAE; exact-count accuracy |
| Level association | Correct level among matched spots; unknown-level rate; per-level errors |
| Full pipeline | Number of true spots, spots returned, correctly localized spots, correctly localized **and** leveled spots; studies with every spot correct |
| Multiple-spot subset | Results for exactly two spots, three-or-more spots, and multiple spots at one level; count and location correctness |
| Reliability | False-negative whole-image gates, positive-but-unlocalized cases, failures, and uncertainty cases |

Evaluate the **entire held-out set**, including negative images and cases where the classifier blocks the detector. For a two-spot case, the result passes only when two distinct predictions correspond to the two reference spots at the required localization threshold; a single merged box is a miss. Level accuracy alone is insufficient if a spot is missing.

Minimum demonstration cases:

- One negative image returning no spots.
- One positive image returning one spot and one level.
- One image with two distinct spots at different levels returning two records.
- One image with two distinct spots at the same level returning two records.
- One positive-but-unlocalized or uncertain-level case shown transparently.

The current SPIDER validation set has no studies with two positive disc rows, so the multiple-spot acceptance cases require new held-out annotations or another suitably labeled dataset.

## 9. Suggested new-project structure

~~~text
hernia-spot-project/
  README.md
  docs/
    FLOW_SPEC.md
    ANNOTATION_GUIDE.md
    EVALUATION_PROTOCOL.md
  data/
    manifests/
    annotations/
  src/
    preprocessing/
    disc_segmentation/
    presence_classifier/
    spot_detector/
    level_assignment/
    inference/
    visualization/
  scripts/
    prepare_data.py
    train_presence.py
    train_spot_detector.py
    evaluate_components.py
    evaluate_end_to_end.py
    run_inference.py
  outputs/
    checkpoints/
    evaluations/
    visualizations/
~~~

Copy this specification into the new project's docs/FLOW_SPEC.md when creating it. Keep the original dataset and selected checkpoints traceable, and record exactly which components are reused or retrained.

## 10. Decisions to settle before implementation

1. Does “whole MRI” mean one selected sagittal 2D image, all sagittal slices, or a 3D volume? This document assumes one full selected image.
2. What precisely counts as one herniation spot when an abnormality is broad or split into parts?
3. Will spot annotations be boxes or pixel masks?
4. Is the target claim research detection of image findings, or a clinical diagnostic decision? The current system supports only a research prototype claim.
5. What independent patient cohort will contain enough multiple-spot examples for evaluation?

These decisions control the classifier input, detector type, annotation workload, and validity of the final metrics.

