# Disc-level axial-review flag

The product target is now **a flag for possible disc bulging or herniation**, with axial MRI recommended to verify the finding. The system does not classify the abnormality as herniation. It retains the CNN disc localization, exact lumbar level assignment, eight-slice banded sagittal T2 crop, and a lightweight four-layer CNN. The displayed box is the **localized disc box**, not a herniation lesion box.

The training target is `abnormal = (bulging_only OR herniation)`. The new binary CNN uses 850 SPIDER training discs (546 abnormal, 304 normal), five patient-separated OOF folds, a 50/50 patient-balanced class sampler, and 15 fixed epochs. A threshold of **0.264306** was chosen solely from OOF training predictions for at least 91% abnormal-disc sensitivity. The full model was trained for the same fixed 15 epochs; its reused 185-disc validation set was scored once at that threshold.

| Model / target | OOF disc AP | OOF sensitivity | OOF normal false flags | Reused validation AP | Validation sensitivity | Validation normal false flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing three-class CNN, collapsed abnormal score | 0.824 | 497/546 (91.0%) | 165/304 | 0.922 | 124/140 (88.6%) | 21/45 |
| New dedicated binary review CNN | 0.799 | 497/546 (91.0%) | 190/304 | 0.918 | 126/140 (90.0%) | 28/45 |

At its OOF-selected threshold, the new binary model flags 59/60 training herniations and 9/9 reused-validation herniations **as possible abnormal discs**. It does not identify them as herniations. Binary validation precision is 126/154 (81.8%). The high abnormal prevalence in SPIDER (546/850 training; 140/185 validation) makes this easier than the former herniation-only task and limits how these figures generalize to other populations.

The binary OOF folds stratify on the combined abnormal target, while the earlier three-class OOF folds stratified on herniation. Both keep patients separate, but their fold assignments differ; the table is a descriptive comparison, not a paired effect estimate.

The end-to-end validation run processed all 37 reused validation T2 series. It localized all 185 labelled disc levels. The exact assigned level had the greatest overlap with its matching source disc in 179/185 cases (96.8%). The predicted **disc** box reached IoU ≥0.5 against the source disc box in 165/185 (89.2%); median same-level IoU was 0.796. The runtime review scores matched the stored validation-crop scores within 3.6×10⁻⁷. Source boxes can be annotated on a different sagittal slice from the runtime selection, so box IoU is an approximate disc-localization check. No herniation lesion-box metric is claimed.

The end-to-end output uses `review_recommended`, `review_levels`, `abnormal_disc_score`, and `disc_bounding_box_original`. It states that axial MRI verification is recommended; it never reports `herniation_detected`. An unavailable level has no flag or box. [Example output for `31_t2`](example_31_t2_review.json) demonstrates the format.

**Status:** This is a working research review-flag pipeline, with patient-separated threshold calibration and an end-to-end check on the reused validation cases. It reaches 91.0% sensitivity OOF and 90.0% on that reused cohort. It does **not** establish more than 90% sensitivity on fresh patients; it still flags 28/45 normal validation discs. Independent expert-labelled T2 cases are required for a reliable performance claim. Axial verification itself is outside this sagittal-only pipeline.

Artifacts: [checkpoint](fixed.pt), [calibration](calibration.json), [OOF predictions](../spider_abnormal_disc_cnn_oof_v1/oof_predictions.csv), [full-model validation predictions](validation_predictions.csv), [end-to-end metrics](../spider_abnormal_review_pipeline_validation_v1/metrics.json), [per-disc level and box audit](../spider_abnormal_review_pipeline_validation_v1/per_disc.csv). Code: [training](../../scripts/train_spider_abnormal_disc_cnn.py), [calibration](../../scripts/calibrate_spider_abnormal_disc.py), [inference](../../src/inference/abnormal_disc_review.py).
