# Axial inference source provenance

The `MRI_Lumbar/` inference modules are a small source snapshot from the teammate
project [LocPC2102/MRI_Lumbar](https://github.com/LocPC2102/MRI_Lumbar), revision
`0e0fa55741d6282f881d4a52a099a0b9becf92d7`.

Included components are `geometric_ldh/`, the LeNet segmentation backbone/U-Net,
and the ROI detector's backbone, model, decoding utilities, and config. Original
nested Git metadata, training datasets, and generated images are excluded.
Source authorship remains with the upstream project; this handoff does not
introduce a new license for that code.

Local integration changes rename upstream `src.*` imports to `MRI_Lumbar.*` and
resolve checkpoint paths relative to the backend source directory. The two
selected axial checkpoints are included at their expected paths and recorded
in `handoff/paired_artifacts.json`.

Upstream training scripts identify AISSLab Axial as the training source. The
saved state dictionaries do not independently establish training patient IDs.
Claims of fully independent external validation remain pending that audit.
