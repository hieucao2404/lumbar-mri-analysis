# Lumbar MRI Analysis — Machine Learning Project Report

**Report date:** 12 October 2026  
**Primary system:** SPIDER sagittal T2 lumbar MRI pipeline  
**Evaluation status:** validation checkpoint; no independent SPIDER test result is recorded

## 1. Executive summary

The current software pipeline takes a sagittal T2 MRI study, localizes intervertebral discs, assigns the five lumbar levels, classifies Pfirrmann degeneration, and produces per-level JSON and comparison images. A separate posterior CNN has two output heads for disc bulging and herniation. Bulging is integrated through a follow-up enrichment step; herniation remains unavailable in the user-facing JSON because its selected checkpoint did not detect any validation positives at the 0.5 threshold. The separate foraminal-severity experiment uses another dataset and is not part of this sagittal SPIDER pipeline.

The complete sagittal pipeline was evaluated on **38 validation studies**. It correctly detected and assigned the level for **166/187 lumbar reference discs (88.77%)**. Detection, level, and Pfirrmann grade were all correct for **74/187 (39.57%)**. The grade classifier is a functional research baseline, not a clinically reliable diagnostic model.

| Main result | Value | Denominator and interpretation |
|---|---:|---|
| All-disc component localization precision / recall / F1 | 0.9205 / 0.9205 / 0.9205 | 243 matches; 264 reference and 264 predicted components |
| Mean matched mask IoU | 0.7758 | Matched all-disc components |
| Lumbar spatial match recall | 174/187 = 93.05% | Derived from the saved per-disc CSV; mask IoU threshold 0.5 |
| Correct lumbar detection and level | 166/187 = 88.77% | All lumbar references |
| Pfirrmann accuracy after correct detection and level | 74/166 = 44.58% | Conditional on the first two stages succeeding |
| Pfirrmann macro F1 / grade MAE | 0.4061 / 0.6747 | Same 166 conditional discs |
| Detection + level + grade correct | 74/187 = 39.57% | Full end-to-end lumbar result |
| Studies with all five levels correct | 28/36 = 77.78% | Studies containing five reference lumbar levels |
| Studies with all five levels and grades correct | 2/36 = 5.56% | Same 36 eligible studies |
| Integrated bulging precision / recall / F1 | 0.8255 / 0.9179 / 0.8693 | 185 discs with an available prediction |
| Selected herniation-head F1 | 0.0000 | 9 positive validation examples; no positives predicted at threshold 0.5 |

The SPIDER metrics in this report are **validation results**. They must not be presented as independent test performance.

## 2. Scope and system architecture

The production sagittal path is:

Sagittal T2 MRI → orientation and intensity preprocessing → 256 × 256 letterboxed image → coordinate-aware multitask U-Net → disc components and boxes → gated hybrid level decoder → original-image 160 × 160 disc crops → five-class Pfirrmann CNN → per-level JSON.

The multitask U-Net has a binary disc head and a six-class pixel-level head for L1–L2 through L5–S1 plus upper discs. Its level head adds normalized X and Y coordinate channels. The decoder combines learned level probabilities, vertical position, and sequence constraints. The Pfirrmann CNN classifies grades I–V from a one-channel disc crop.

A separate posterior CNN accepts a two-channel 160 × 160 crop and emits two logits: bulging and herniation. The integrated bulging step adds a bulging probability and detected/not-detected status to the JSON. The base sagittal JSON still marks posterior_disc_finding unavailable; the current bulging enrichment does not make herniation available.

The source model definitions are [multitask U-Net](src/models/spider_disc_multitask_unet.py), [Pfirrmann CNN](src/models/spider_model.py), and [posterior CNN](src/models/spider_posterior_model.py).

## 3. Data and splits

### 3.1 SPIDER sagittal data

| Dataset item | Training | Validation | Total |
|---|---:|---:|---:|
| Sagittal T2 studies / patients | 172 | 38 | 210 |
| Disc annotation rows | 1,187 | 265 | 1,452 |
| Posterior classifier crop samples | 860 | 188 | 1,048 |

The study and disc counts were checked against [the localization manifest](data/processed/spider_disc_localization/manifest.csv) and [disc annotation table](data/processed/spider_discs.csv). The earlier project report records zero patient overlap between training and validation. The current posterior training history reports 860 training samples and 188 validation samples. These are task-specific crops; they are not interchangeable with the 1,452 general disc rows.

The full sagittal evaluator has **187 lumbar reference discs** across 38 studies. Thirty-six studies have all five lumbar reference levels; one has four and one has three. The all-disc localization evaluation contains **264 reference components**, including upper discs. The posterior validation evaluation contains **188 reference discs**. These task-specific denominators must remain separate.

The SPIDER development record does **not** contain an independent held-out test result. Model selection and the hybrid decoder spacing gate were performed with validation data.

### 3.2 Separate foraminal-severity experiment

The auxiliary LSS/AISSLab foraminal-severity experiment has 2,088 training, 452 validation, and 438 test samples in its recorded metrics. It predicts Normal, Mild, Moderate, or Severe. It is **foraminal** severity, not sagittal spinal-canal stenosis, and it is not integrated into the SPIDER pipeline.

## 4. Training and model-development progress

The table shows training and validation values at the saved selected epoch. Histories contain later epochs for several runs; a later final epoch is not automatically the selected model.

| Experiment | Selected epoch / recorded epochs | Training metric at selected epoch | Validation metric at selected epoch | Role |
|---|---:|---|---|---|
| Binary disc U-Net baseline | 25 / 30 | Dice 0.8631 | Dice 0.8525; precision 0.8206; recall 0.9003 | Earlier segmentation baseline |
| Coordinate multitask U-Net v2 | 11 / 19 | Binary Dice 0.8851; pixel-level accuracy 0.7514 | Binary Dice 0.8589; pixel-level accuracy 0.6945; level macro F1 0.6880 | Selected localization/level network |
| Pfirrmann CNN repro v1 | 11 / 19 | Accuracy 0.4945; macro F1 0.4525 | Accuracy 0.4642; macro F1 0.4438; grade MAE 0.6604 | Selected grade network |
| Posterior two-output CNN | 28 / 30 | Bulging F1 0.8046; herniation F1 0.0723 | Bulging F1 0.8622; herniation F1 0.0000 | Bulging is integrated separately; herniation head is not usable at current threshold |
| Foraminal severity CNN | 17 / 25 | Accuracy 0.8716; macro F1 0.8325 | Accuracy 0.7456; macro F1 0.5855 | Separate auxiliary experiment |

The coordinate multitask model improves the selected binary Dice slightly over the earlier U-Net baseline, while its raw pixel-level level head remains weaker than the final decoder. The Pfirrmann and foraminal training histories show lower performance on validation than training. In particular, the foraminal model has a substantial macro-F1 gap at its selected epoch. These gaps should be discussed as generalization concerns, without claiming a cause that has not been tested.

Training histories and selected metrics: [binary localization](outputs/spider_disc_localization_softdice_v1/history.json), [multitask model](outputs/spider_disc_multitask_coord_v2/history.json), [Pfirrmann](outputs/spider_pfirrmann_repro_v1/history.json), [posterior](outputs/spider_posterior_baseline/history.json), and [foraminal](outputs/foraminal_severity_baseline_v2/history.json).

### 4.1 Level-decoder development

All methods below use the same 38-study validation subset and the saved coordinate-aware multitask checkpoint.

| Decoder | Accuracy on matched components | End-to-end lumbar level accuracy | Five-level studies with all levels correct |
|---|---:|---:|---:|
| Independent learned label | 0.7037 | 0.5989 | 12/36 |
| Position | 0.9342 | 0.8556 | 28/36 |
| Fused sequence | 0.9259 | 0.8610 | 26/36 |
| Gated hybrid, selected | **0.9588** | **0.8877** | **28/36** |

The gated hybrid has matched-component level macro F1 **0.9591**. It selected the fused-sequence source for 30 studies and position for 8. The configured hybrid spacing-CV gate is **0.20**, selected using validation data. Consequently these results are development/validation estimates, not a fully independent assessment. Source: [decoder comparison](outputs/spider_disc_multitask_coord_v2/level_decoder_hybrid_all/summary.json).

An earlier direct pixel-level disc-class model is not a valid substitute for this decoder: its reported pixel accuracy of 0.9880 was dominated by background, while macro lumbar-level Dice was only 0.0667. Source: [direct level-model metrics](outputs/spider_disc_levels_v1/best_validation_metrics.json).

## 5. Full sagittal validation evaluation

The production evaluator used the coordinate multitask checkpoint and Pfirrmann repro-v1 checkpoint. The segmentation probability threshold was 0.5, minimum component area was 25 pixels, and component matching required mask IoU at least 0.5. The decoder weights were learned 1.0, position 0.75, upper 0.20; maximum spacing CV 0.55 and hybrid spacing CV 0.20. No patients were excluded from the primary 38-study run.

### 5.1 Localization and level assignment

| Measure | Count or score |
|---|---:|
| All-disc reference components | 264 |
| All-disc predicted components | 264 |
| Spatially matched all-disc components | 243 |
| Component precision | 243/264 = 0.9205 |
| Component recall | 243/264 = 0.9205 |
| Component F1 | 0.9205 |
| Mean matched mask IoU | 0.7758 |
| Lumbar reference discs | 187 |
| Matched lumbar reference discs | 174/187 = 0.9305 |
| Correct detection and level | 166/187 = 0.8877 |
| Matched lumbar discs with an incorrect level | 8 |
| Unmatched lumbar reference discs | 13 |
| Extra predictions assigned to a lumbar level | 15 |

The 15 extra lumbar predictions are reported by the per-study evaluator. They include predictions that match an upper disc or no reference disc. Do not infer lumbar precision by dividing 174 by 174 + 15: the evaluator counts and assignments have different matching scopes.

### 5.2 Per-level end-to-end performance

This table was calculated from [per-disc predictions](outputs/spider_sagittal_pipeline_v1/per_disc_predictions.csv). “Matched” uses mask IoU at least 0.5. “Correct grade” requires a correct level first.

| Reference level | References | Matched | Correct detection + level | Correct detection + level + grade | Combined accuracy |
|---|---:|---:|---:|---:|---:|
| L1–L2 | 36 | 34 | 31 | 17 | 47.22% |
| L2–L3 | 37 | 35 | 34 | 18 | 48.65% |
| L3–L4 | 38 | 37 | 35 | 13 | 34.21% |
| L4–L5 | 38 | 34 | 33 | 15 | 39.47% |
| L5–S1 | 38 | 34 | 33 | 11 | 28.95% |
| **Total** | **187** | **174** | **166** | **74** | **39.57%** |

L5–S1 has the lowest combined accuracy in this run. The table is descriptive; no confidence intervals or independent-test estimates are available.

### 5.3 Pfirrmann grade classification

Among the 166 discs correctly detected **and** leveled, the grade classifier achieved accuracy **0.4458**, macro F1 **0.4061**, and mean absolute grade error **0.6747**. These conditional figures differ from the full-pipeline 74/187 accuracy and from the standalone crop-validation metrics in Section 4.

| Reference grade | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| I | 16 | 0.7500 | 0.1875 | 0.3000 |
| II | 17 | 0.4194 | 0.7647 | 0.5417 |
| III | 58 | 0.4828 | 0.2414 | 0.3218 |
| IV | 50 | 0.4222 | 0.7600 | 0.5429 |
| V | 25 | 0.5000 | 0.2400 | 0.3243 |

Confusion matrix; **rows are reference grades** and **columns are predicted grades**:

| True \ Predicted | I | II | III | IV | V |
|---|---:|---:|---:|---:|---:|
| I | 3 | 7 | 4 | 1 | 1 |
| II | 1 | 13 | 2 | 1 | 0 |
| III | 0 | 7 | 14 | 34 | 3 |
| IV | 0 | 3 | 7 | 38 | 2 |
| V | 0 | 1 | 2 | 16 | 6 |

The classifier often predicts grade IV: 34 of 58 true grade-III discs and 16 of 25 true grade-V discs were called IV. Grade I, III, and V recall is low. In the earlier 228-disc paired-crop analysis, expert-defined crops had accuracy 0.4649, macro F1 0.4451, MAE 0.6360; detected crops had accuracy 0.4605, macro F1 0.4342, MAE 0.6316. That comparison suggests the grade classifier itself is a major limitation, although the paired experiment and current 166-disc conditional run have different evaluation populations. The paired results were recorded in [the earlier report](Report_August_8th_2026.md).

### 5.4 Complete-study performance

| Study-level measure | Result |
|---|---:|
| Validation studies evaluated | 38 |
| Studies with all five reference lumbar levels | 36 |
| Five-level studies with all five levels correct | 28/36 = 77.78% |
| Five-level studies with all five levels and grades correct | 2/36 = 5.56% |

The comparison logs identify 101_t2 and 118_t2 as the two five-level studies with all grades correct. Individual examples must not be used as substitutes for the 38-study result. Source: [full sagittal summary](outputs/spider_sagittal_pipeline_v1/summary.json) and [per-study results](outputs/spider_sagittal_pipeline_v1/per_study_results.csv).

A secondary “reliable” analysis excluded patients 107 and 152: 36 studies, 177 lumbar references, 158 correct levels (89.27%), and 71 complete disc results (40.11%). It is a sensitivity analysis, not the primary result. Source: [reliable-subset summary](outputs/spider_sagittal_pipeline_v1_reliable/summary.json).

## 6. Posterior disc findings

### 6.1 Selected two-output classifier

The posterior CNN was trained for **two simultaneous binary labels**, disc bulging and disc herniation, rather than a mutually exclusive three-class label. At its selected epoch, on 188 validation crop samples and threshold 0.5:

| Label | Positive support | TP | FP | FN | TN | Precision | Recall | F1 | Average precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bulging | 137 | 122 | 24 | 15 | 27 | 0.8356 | 0.8905 | 0.8622 | 0.8724 |
| Herniation | 9 | 0 | 0 | 9 | 179 | 0.0000 | 0.0000 | 0.0000 | 0.2773 |

The herniation set has only nine positives, and the selected checkpoint predicted zero positives at 0.5. It cannot support a reliable “none / bulge / suspected herniation” clinical-style output yet. Source: [selected posterior metrics](outputs/spider_posterior_baseline/best_validation_metrics.json).

### 6.2 Integrated bulging evaluation

The separate enrichment pipeline ran over all 38 validation studies. Three of 188 reference discs had no available prediction; all three carried a positive bulging label.

| Bulging measure | All 38 studies |
|---|---:|
| Reference discs / positive references | 188 / 137 |
| Available / missing predictions | 185 / 3 |
| TP / FP / FN among available / TN | 123 / 26 / 11 / 25 |
| Precision among available | 0.8255 |
| Recall among available | 0.9179 |
| F1 among available | 0.8693 |
| Average precision among available | 0.8658 |
| Positive recall over **all** references | 123/137 = 0.8978 |
| F1 when the three missing positives count as false negatives | 0.8601, derived |
| Always-positive F1 over all references | 0.8431 |

Excluding two studies with uncertain numbering yields 178 references, 175 available predictions, and available-case precision 0.8380, recall 0.9370, F1 0.8848. This is a sensitivity analysis. The bulging task has a high positive prevalence (137/188), so the always-positive baseline is important context. Source: [integrated bulging summary](outputs/spider_bulging_pipeline_validation/summary.json).

No saved result here measures a joint detection + level + grade + bulging success rate. The bulging F1 should not be added to or multiplied by the sagittal grade accuracy to invent one.

## 7. Separate foraminal-severity test result

The auxiliary four-class foraminal CNN was evaluated on a **438-sample test split**. Its test accuracy was **0.6963** and macro F1 **0.5081**. This is the only independent test result covered in this report; it belongs to a different dataset and task.

| Test class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Normal | 300 | 0.8865 | 0.8333 | 0.8591 |
| Mild | 78 | 0.3731 | 0.3205 | 0.3448 |
| Moderate | 35 | 0.2708 | 0.3714 | 0.3133 |
| Severe | 25 | 0.4146 | 0.6800 | 0.5152 |

Normal performance is much stronger than the middle-severity classes. The separate model does **not** establish spinal-canal stenosis performance in the SPIDER sagittal pipeline. Source: [foraminal test metrics](outputs/foraminal_severity_baseline_v2/test_metrics.json).

## 8. Visual evidence and outputs

| Artifact | What it shows |
|---|---|
| [Overview chart](outputs/spider_sagittal_pipeline_v1/overview.png) | Aggregate pipeline and grade metrics |
| [101_t2 comparison with bulging](outputs/sagittal_inference/101_t2_comparison_with_bulging.png) | Five of five grades correct; grade MAE 0; same-level mean box IoU 0.713 |
| [112_t2 comparison with bulging](outputs/sagittal_inference/112_t2_comparison_with_bulging.png) | Zero of five grades correct; grade MAE 1.000; same-level mean box IoU 0.775 |
| [Per-disc validation CSV](outputs/spider_sagittal_pipeline_v1/per_disc_predictions.csv) | Reference/predicted levels and grades for audit |
| [Per-study validation CSV](outputs/spider_sagittal_pipeline_v1/per_study_results.csv) | Study totals and complete-study flags |

The 101 and 112 images illustrate variability. The report’s main conclusions use all 38 validation studies.

## 9. Metric definitions and interpretation

- **Precision** = TP / (TP + FP); **recall** = TP / (TP + FN); **F1** is their harmonic mean.
- **Dice** is the pixel-overlap form of F1 for binary segmentation. Pixel Dice and instance/component F1 evaluate different units.
- **IoU** measures intersection divided by union. Matching in the full pipeline requires mask IoU at least 0.5.
- **Macro F1** averages class F1 values equally, making weak minority-grade performance visible.
- **Grade MAE** averages the absolute difference between ordinal grade numbers I–V. It does not replace exact-grade accuracy.
- **Average precision** summarizes precision across score thresholds; it is not the same as precision at threshold 0.5.
- A grade result “given correct detection and level” excludes upstream failures. The 39.57% end-to-end result includes them.

## 10. Limitations, readiness, and next work

| Component | Current status | Evidence-based limitation or next step |
|---|---|---|
| Sagittal preprocessing | Implemented | Record orientation and image-selection assumptions in deployment documentation |
| Disc localization and level assignment | Working baseline | Validate on an untouched patient cohort; investigate 13 unmatched lumbar discs and 8 matched wrong-level discs |
| Pfirrmann | Integrated baseline | Exact grade remains weak; prioritize class balance, calibration, error review, and independent evaluation |
| Bulging | Trained and separately integrated | Audit three unavailable predictions and evaluate a joint four-stage pipeline |
| Herniation | Output head trained, not usable/integrated | Nine validation positives and zero predicted positives at 0.5; more positive data and threshold/calibration analysis required |
| Sagittal spinal-canal stenosis | Not implemented | Obtain suitable structured labels and train/evaluate a distinct head |
| Foraminal severity | Separate test-stage experiment | Keep separate from canal stenosis and SPIDER until domain and anatomy match |
| FPGA acceleration | Not demonstrated for this full pipeline | Quantize stable blocks and measure accuracy, latency, resource use, and power after software validation |

The validation set influenced model selection and the 0.20 hybrid decoder gate. A held-out patient test cohort or patient-level cross-validation is needed for a stronger performance claim. The available evaluation does not establish clinical diagnostic safety. The system should be described as a capstone engineering prototype.

Recommended next sequence:

1. Preserve this v1 report, checkpoint paths, source revision, and validation outputs.
2. Generate and review the per-study MRI comparisons for the full validation set, including bulging labels where available.
3. Audit Pfirrmann grade-III/IV/V confusions and missing bulging predictions.
4. Improve and independently validate herniation detection before exposing suspected herniation in the JSON.
5. Build a complete joint evaluation for detection, level, grade, and posterior findings.
6. Add a patient-level held-out SPIDER test cohort, then assess quantization and FPGA blocks.

## 11. Reproducibility index

| Purpose | File |
|---|---|
| Sagittal evaluation code | [scripts/evaluate_spider_sagittal_pipeline.py](scripts/evaluate_spider_sagittal_pipeline.py) |
| Full sagittal validation summary | [outputs/spider_sagittal_pipeline_v1/summary.json](outputs/spider_sagittal_pipeline_v1/summary.json) |
| Level decoder comparison | [outputs/spider_disc_multitask_coord_v2/level_decoder_hybrid_all/summary.json](outputs/spider_disc_multitask_coord_v2/level_decoder_hybrid_all/summary.json) |
| Bulging integrated evaluation code | [scripts/evaluate_spider_bulging_pipeline.py](scripts/evaluate_spider_bulging_pipeline.py) |
| Bulging integrated validation summary | [outputs/spider_bulging_pipeline_validation/summary.json](outputs/spider_bulging_pipeline_validation/summary.json) |
| Multitask localization/level checkpoint | [outputs/spider_disc_multitask_coord_v2/best_model.pt](outputs/spider_disc_multitask_coord_v2/best_model.pt) |
| Pfirrmann checkpoint | [outputs/spider_pfirrmann_repro_v1/best_model.pt](outputs/spider_pfirrmann_repro_v1/best_model.pt) |
| Posterior checkpoint | [outputs/spider_posterior_baseline/best_model.pt](outputs/spider_posterior_baseline/best_model.pt) |
| Earlier progress report | [Report_August_8th_2026.md](Report_August_8th_2026.md) |

