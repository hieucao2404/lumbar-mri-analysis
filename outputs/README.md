# Versioned model outputs

These selected outputs are committed so teammates can clone and run the current
pipeline. Run `python -m scripts.check_pipeline_artifacts` after cloning.
Required SHA-256 hashes are in `docs/handoff/paired_artifacts.json`.

| Directory | Contents and use |
|---|---|
| `spider_disc_multitask_coord_v2` | Default localizer, validation metrics, training history |
| `spider_disc_localization_softdice_v1` | Binary localization pretraining weights and history |
| `spider_abnormal_vgg16_attention_guarded_v1` | Default classifier, calibration, OOF/validation predictions, metrics |
| `spider_abnormal_vgg16_lstm_attention_guarded_v1` | Optional LSTM classifier and its corresponding calibration/results |
| `spider_abnormal_disc_cnn_fixed_v1` | Legacy lightweight CNN and results for the older SPIDER raw-MRI command |
| `lsma_paired_pipeline_full_v1` | Full 515-patient fused evaluation report and per-disc predictions |
| `mendeley_multislice_transfer_full_v1` | External sagittal attention/LSTM comparison and failure records |
| `rsna_five_level_axial_matching_full_v1` | Five-level matching report and predictions |
| `paired_pipeline_concurrent_timing_v1` | Repeated serial/concurrent/all-level timing measurements |

The axial weights live under
`MRI_Lumbar/cnn_segmentation/models/checkpoints_4cls/` and
`MRI_Lumbar/cnn/cnn_roi_detection/outputs_256/`. This handoff adds about 132 MB of
selected artifacts; each checkpoint is below 100 MB.

Evaluation CSVs use the public datasets' anonymous identifiers. Raw MRI, local
clinical cases, generated patient crops, and overlays are excluded. New output
directories are ignored unless an explicit exception is added. Historical files
already tracked by Git remain tracked until a separate cleanup removes them.
