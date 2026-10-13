# CNN Architectures and Layer Specifications

**Project:** Lumbar MRI Analysis  
**Document date:** 12 October 2026  
**Scope:** Every CNN class defined in the current src/models directory. Shapes below omit the batch dimension unless shown.

This document describes model architecture. Measured accuracy, precision, recall, F1, and study-level results are in [the project ML report](PROJECT_ML_REPORT.md).

## 1. Model inventory and usage

| Model configuration | Input tensor | Output tensor | Trainable parameters | Project role |
|---|---|---|---:|---|
| SpiderDiscMultitaskUNet | B × 1 × 256 × 256 | Binary logits B × 1 × 256 × 256; level logits B × 6 × 256 × 256 | **488,551** | Active SPIDER localization and learned level head |
| SpiderPfirrmannCNN | B × 1 × 160 × 160 | B × 5 grade logits | **302,773** | Active SPIDER Pfirrmann classifier |
| SpiderPosteriorCNN | B × 2 × 160 × 160 | B × 2 finding logits | **215,538** | Active bulging enrichment; herniation head trained but not exposed |
| SpiderDiscUNet, binary | B × 1 × 256 × 256 | B × 1 mask logit map | 487,713 | Earlier disc segmentation baseline and multitask pretraining |
| SpiderDiscUNet, six-class | B × 1 × 256 × 256 | B × 6 class logit maps | 487,798 | Earlier direct level-segmentation experiment |
| ForaminalSeverityCNN | B × 1 × 128 × 128 | B × 4 severity logits | 310,740 | Separate LSS/AISSLab foraminal experiment |
| LumbarLocalizer | B × 1 × 128 × 128 | B × 4 normalized box coordinates | 359,940 | Separate coarse lumbar-box experiment |
| SagittalDiscCNN | B × 1 × 128 × 128 in its training script | Grade B × 5; bulge B; herniation B | 115,047 | Earlier experimental multihead model; not in active SPIDER inference |

Counts were checked by instantiating every class in the project Python environment. They include trainable convolution, linear, BatchNorm, and GroupNorm affine parameters. They exclude non-trainable running statistics and activation memory.

### Layer notation

- Conv k3 p1: 3 × 3 convolution, stride 1, padding 1; spatial height and width stay the same.
- Conv k1: 1 × 1 convolution for an output head or channel projection.
- Pool 2: 2 × 2 max-pooling with stride 2; height and width halve.
- BN: BatchNorm2d. GN4: GroupNorm with four groups.
- GAP 1: adaptive average pooling to one spatial value per channel.
- Logits are raw model outputs. The training loss or inference code applies softmax or sigmoid as appropriate; the final convolution or linear layer does not itself convert logits to probabilities unless stated.

## 2. Active SPIDER architecture

The active sagittal prediction path is:

256 × 256 sagittal T2 image → SpiderDiscMultitaskUNet → disc components and gated hybrid level decoder → original-image 160 × 160 crops → SpiderPfirrmannCNN → per-level JSON.

The bulging enrichment path uses the predicted disc location to create a two-channel posterior crop → SpiderPosteriorCNN → bulging probability and thresholded status. The level decoder, component extraction, crop geometry, and JSON formatting are algorithmic stages, not additional CNN layers.

### 2.1 SpiderDiscMultitaskUNet

**Source:** [spider_disc_multitask_unet.py](src/models/spider_disc_multitask_unet.py)  
**Selected checkpoint:** [coordinate multitask v2](outputs/spider_disc_multitask_coord_v2/best_model.pt)  
**Input:** one normalized, letterboxed sagittal T2 image, B × 1 × 256 × 256.  
**Shared block:** two Conv k3 p1 layers, each followed by GN4 and ReLU. Both convolutions have no bias.

| Stage | Operation and channel specification | Output per image | Parameters |
|---|---|---|---:|
| Encoder 1 | Shared block 1 → 16 | 16 × 256 × 256 | 2,512 |
| Pool 1 | Pool 2 | 16 × 128 × 128 | 0 |
| Encoder 2 | Shared block 16 → 32 | 32 × 128 × 128 | 13,952 |
| Pool 2 | Pool 2 | 32 × 64 × 64 | 0 |
| Encoder 3 | Shared block 32 → 64 | 64 × 64 × 64 | 55,552 |
| Pool 3 | Pool 2 | 64 × 32 × 32 | 0 |
| Bottleneck | Shared block 64 → 128 | 128 × 32 × 32 | 221,696 |
| Upsample + skip | Bilinear upsample; concatenate Encoder 3: 128 + 64 channels | 192 × 64 × 64 | 0 |
| Decoder 3 | Shared block 192 → 64 | 64 × 64 × 64 | 147,712 |
| Upsample + skip | Bilinear upsample; concatenate Encoder 2: 64 + 32 channels | 96 × 128 × 128 | 0 |
| Decoder 2 | Shared block 96 → 32 | 32 × 128 × 128 | 36,992 |
| Upsample + skip | Bilinear upsample; concatenate Encoder 1: 32 + 16 channels | 48 × 256 × 256 | 0 |
| Decoder 1 | Shared block 48 → 16 | 16 × 256 × 256 | 9,280 |
| Binary head | Conv k1, 16 → 1, with bias | 1 × 256 × 256 | 17 |
| Coordinate concat | Append X and Y maps, each normalized from −1 to +1 | 18 × 256 × 256 | 0 |
| Level head, projection | Conv k1, 18 → 32, no bias; GN4; ReLU | 32 × 256 × 256 | 640 |
| Level head, output | Conv k1, 32 → 6, with bias | 6 × 256 × 256 | 198 |
| **Total** | Shared trunk + two heads | Two raw-logit maps | **488,551** |

The binary logit map detects all intervertebral discs. The six level-logit channels mean **L1–L2, L2–L3, L3–L4, L4–L5, L5–S1, and Upper/non-lumbar**. These are pixel-level scores; a gated hybrid decoder later assigns one anatomical level to each detected component. Upsampling is bilinear with align_corners=False.

The model transfers the earlier binary U-Net weights into the shared trunk and binary head; the coordinate-aware level head starts new. Training first warms up the level head, then fine-tunes the full network. The objective combines binary BCE-with-logits and Dice losses with level cross-entropy. See [training code](scripts/train_spider_disc_multitask.py).

### 2.2 SpiderPfirrmannCNN

**Source:** [spider_model.py](src/models/spider_model.py)  
**Selected checkpoint:** [Pfirrmann repro v1](outputs/spider_pfirrmann_repro_v1/best_model.pt)  
**Input:** one normalized original-image disc crop, B × 1 × 160 × 160.  
**Repeated block:** Conv k3 p1 → BN → ReLU → Conv k3 p1 → BN → ReLU → Pool 2. The convolutions use bias.

| Stage | Channels | Output per image | Parameters |
|---|---|---|---:|
| Block 1 | 1 → 16 → 16 | 16 × 80 × 80 | 2,544 |
| Block 2 | 16 → 32 → 32 | 32 × 40 × 40 | 14,016 |
| Block 3 | 32 → 64 → 64 | 64 × 20 × 20 | 55,680 |
| Block 4 | 64 → 128 → 128 | 128 × 10 × 10 | 221,952 |
| GAP 1 + flatten | 128 channels → 128 features | 128 | 0 |
| Dense 1 + ReLU | 128 → 64 | 64 | 8,256 |
| Dropout | p = 0.30 | 64 | 0 |
| Dense output | 64 → 5 | Five grade logits | 325 |
| **Total** | | | **302,773** |

Output order is Pfirrmann I, II, III, IV, V. The model returns five raw logits and uses cross-entropy for training. It does not predict the disc level or locate the disc; both are upstream operations.

### 2.3 SpiderPosteriorCNN

**Source:** [spider_posterior_model.py](src/models/spider_posterior_model.py)  
**Selected checkpoint:** [posterior baseline](outputs/spider_posterior_baseline/best_model.pt)  
**Input:** B × 2 × 160 × 160. Channel 0 is the normalized MRI crop; channel 1 is a binary mask locating the target disc. Both channels refer to the same crop, as defined in [the posterior dataset](src/data/spider_posterior_dataset.py).  
**Repeated block:** Conv k3 p1 → GN4 → ReLU → Conv k3 p1 → GN4 → ReLU → Pool 2. Convolutions have no bias.

| Stage | Channels | Output per image | Parameters |
|---|---|---|---:|
| Block 1 | 2 → 16 → 16 | 16 × 80 × 80 | 2,656 |
| Block 2 | 16 → 32 → 32 | 32 × 40 × 40 | 13,952 |
| Block 3 | 32 → 64 → 64 | 64 × 20 × 20 | 55,552 |
| Block 4 | 64 → 96 → 96 | 96 × 10 × 10 | 138,624 |
| GAP 1 + flatten | 96 channels → 96 features | 96 | 0 |
| Dense 1 + ReLU | 96 → 48 | 48 | 4,656 |
| Dropout | p = 0.30 | 48 | 0 |
| Dense output | 48 → 2 | Two finding logits | 98 |
| **Total** | | | **215,538** |

Logit 0 is **disc bulging** and logit 1 is **disc herniation**. They are independent binary outputs, so both labels can be present on one disc. Training uses BCEWithLogitsLoss, with a larger positive weight for herniation in the current script. The selected checkpoint detected no validation herniations at threshold 0.5; only bulging is surfaced by [the enrichment script](scripts/add_spider_bulging.py).

## 3. Earlier SPIDER U-Net variants

### 3.1 SpiderDiscUNet, binary output

**Source:** [spider_disc_unet.py](src/models/spider_disc_unet.py)  
**Checkpoint:** [binary localization baseline](outputs/spider_disc_localization_softdice_v1/best_model.pt)

This is the same U-Net encoder, bottleneck, decoder, skip connections, and shared block listed in Section 2.1. It has one head: Conv k1, 16 → 1, with bias. Total: **487,713 trainable parameters**. Input is B × 1 × 256 × 256; output is B × 1 × 256 × 256 raw binary logits. Training uses BCE-with-logits plus Dice loss. Its weights initialize the active multitask U-Net.

### 3.2 SpiderDiscUNet, six-class direct level output

**Source:** [spider_disc_unet.py](src/models/spider_disc_unet.py)  
**Checkpoint:** [direct level experiment](outputs/spider_disc_levels_v1/best_model.pt)

The same trunk instead ends in Conv k1, 16 → 6, with bias. Total: **487,798 trainable parameters**. Input is B × 1 × 256 × 256; output is B × 6 × 256 × 256 raw class logits. Here the six channels are **Background plus the five lumbar levels**. This class order differs from the multitask U-Net's six level channels, which contain **Upper** instead of Background. The direct model uses cross-entropy and is not the chosen level-assignment method.

## 4. Separate and legacy CNNs

### 4.1 ForaminalSeverityCNN

**Source:** [foraminal_cnn.py](src/models/foraminal_cnn.py)  
**Checkpoint:** [foraminal baseline v2](outputs/foraminal_severity_baseline_v2/best_model.pt)  
**Input/output:** B × 1 × 128 × 128 → B × 4 logits.  
**Repeated block:** Conv k3 p1, no bias → BN → ReLU → Conv k3 p1, no bias → BN → ReLU → Pool 2.

| Stage | Channels | Output per image | Parameters |
|---|---|---|---:|
| Block 1 | 1 → 16 → 16 | 16 × 64 × 64 | 2,512 |
| Block 2 | 16 → 32 → 32 | 32 × 32 × 32 | 13,952 |
| Block 3 | 32 → 64 → 64 | 64 × 16 × 16 | 55,552 |
| Block 4 | 64 → 128 → 128 | 128 × 8 × 8 | 221,696 |
| GAP 1 + flatten | 128 → 128 features | 128 | 0 |
| Dense 1 + ReLU | 128 → 128 | 128 | 16,512 |
| Dropout | p = 0.25 | 128 | 0 |
| Dense output | 128 → 4 | Four class logits | 516 |
| **Total** | | | **310,740** |

Class order is **Normal, Mild, Moderate, Severe**. This model uses a separate LSS/AISSLab foraminal dataset. It is not a sagittal canal-stenosis head. Training uses class-weighted cross-entropy; validation and test loss are unweighted.

### 4.2 LumbarLocalizer

**Source:** [lumbar_localizer.py](src/models/lumbar_localizer.py)  
**Checkpoint:** [coarse localizer](artifacts/lumbar_localizer/best_model.pt)  
**Input/output:** B × 1 × 128 × 128 → B × 4 normalized box coordinates.  
**Repeated feature stage:** one Conv k3 p1 with bias → ReLU → Pool 2. No BN or GN.

| Stage | Channels | Output per image | Parameters |
|---|---|---|---:|
| Conv + pool 1 | 1 → 16 | 16 × 64 × 64 | 160 |
| Conv + pool 2 | 16 → 32 | 32 × 32 × 32 | 4,640 |
| Conv + pool 3 | 32 → 64 | 64 × 16 × 16 | 18,496 |
| Conv + pool 4 | 64 → 128 | 128 × 8 × 8 | 73,856 |
| Adaptive average pool | 128 × 8 × 8 → 128 × 4 × 4 | 128 × 4 × 4 | 0 |
| Flatten + dense + ReLU | 2,048 → 128 | 128 | 262,272 |
| Dense + sigmoid | 128 → 4 | Four values in [0, 1] | 516 |
| **Total** | | | **359,940** |

The outputs are normalized center_x, center_y, width, height for one whole-lumbar box. This older coarse localizer is separate from the SPIDER disc-component localizer used in the current demo.

### 4.3 SagittalDiscCNN, early multihead experiment

**Source:** [sagittal_disc_cnn.py](src/models/sagittal_disc_cnn.py)  
**Training entry point:** [train_sagittal_disc.py](scripts/train_sagittal_disc.py)  
**Training input size:** B × 1 × 128 × 128. The adaptive pool also permits other spatial sizes.  
**Repeated stage:** one Conv k3 p1 with bias → BN → ReLU → Pool 2.

| Stage | Channels | Output per image at 128 × 128 input | Parameters |
|---|---|---|---:|
| Conv stage 1 | 1 → 16 | 16 × 64 × 64 | 192 |
| Conv stage 2 | 16 → 32 | 32 × 32 × 32 | 4,704 |
| Conv stage 3 | 32 → 64 | 64 × 16 × 16 | 18,624 |
| Conv stage 4 | 64 → 128 | 128 × 8 × 8 | 74,112 |
| GAP 1 + flatten | 128 channels → 128 features | 128 | 0 |
| Shared dense + ReLU | 128 → 128 | 128 | 16,512 |
| Pfirrmann head | 128 → 5 | Five grade logits | 645 |
| Bulge head | 128 → 1 | One binary logit, squeezed to B | 129 |
| Herniation head | 128 → 1 | One binary logit, squeezed to B | 129 |
| **Total** | | | **115,047** |

This class defines a single shared backbone with three task heads. The active SPIDER implementation instead uses the separate Pfirrmann CNN and posterior CNN described above.

## 5. Output semantics and implementation notes

| Output | Shape per image | Probability conversion | Used for |
|---|---|---|---|
| Multitask binary head | 1 × 256 × 256 | Sigmoid, then component thresholding | Disc mask |
| Multitask level head | 6 × 256 × 256 | Softmax across level classes | Learned component-level evidence |
| Pfirrmann head | 5 | Softmax across grades | Grade I–V |
| Posterior head | 2 | Independent sigmoid per label | Bulging; experimental herniation |
| Foraminal head | 4 | Softmax across severity classes | Separate foraminal task |
| Lumbar localizer head | 4 | Sigmoid is inside model | Normalized whole-lumbar box |
| Early SagittalDiscCNN heads | 5, 1, 1 | Softmax for grade; sigmoid for each finding | Earlier multihead experiment |

The U-Net's large full-resolution feature maps and skip tensors can dominate inference memory even though its parameter count is only about 0.49 million. Parameter count does not equal runtime or FPGA resource use. A hardware report would also need activation sizes, operation counts, numerical precision, latency, memory traffic, and accuracy after quantization.

## 6. Verification

A synthetic, no-gradient forward pass in the project environment confirmed all parameter counts and output shapes:

| Configuration | Verified output |
|---|---|
| Binary SpiderDiscUNet | [1, 1, 256, 256] |
| Six-class SpiderDiscUNet | [1, 6, 256, 256] |
| SpiderDiscMultitaskUNet | [1, 1, 256, 256] and [1, 6, 256, 256] |
| SpiderPfirrmannCNN | [1, 5] |
| SpiderPosteriorCNN | [1, 2] |
| ForaminalSeverityCNN | [1, 4] |
| LumbarLocalizer | [1, 4] |
| SagittalDiscCNN | grade [1, 5], bulge [1], herniation [1] |

This shape check verifies the model graph with synthetic inputs. It does not measure predictive accuracy or hardware performance; those results are covered separately in [PROJECT_ML_REPORT.md](PROJECT_ML_REPORT.md).

