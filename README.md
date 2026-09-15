# Degradation Transmission Prior-Guided All-Weather Perception Network for Insulator Defects

[![Paper](https://img.shields.io/badge/Paper-PDF-red)]() <!-- TODO: replace with paper link -->
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)]()
[![License](https://img.shields.io/badge/License-MIT-blue)]() <!-- TODO: adjust license -->

Official PyTorch implementation of the paper:

**"Degradation Transmission Prior-Guided All-Weather Perception Network for Insulator Defects"**

*Xiaoyang Song, Liqun Liu\*, Qingfeng Wu, Kai Yang, Jiebao Yang, Jianfeng Zhang* (Taiyuan University of Science and Technology / Taiyuan Institute of Technology)

> **TL;DR**: A plug-and-play, zero-inference-overhead multi-task perception enhancement network for insulator defect detection, semantic segmentation, and instance segmentation under fog / rain / snow, guided by a dynamic Degradation Transmission Map (DT-Map) prior. The auxiliary restoration and weather-classification branches are used **only during training** and are completely removed at inference, so the deployed model keeps exactly the same parameters, FLOPs, and speed as the native baseline while gaining **+4.5% mAP@0.5 (object detection)**, ~**+5% Mask AP@0.5 (instance segmentation)**, and ~**+2% mIoU (semantic segmentation)**.

---

## 🔔 News

- **[2026-09]** Initial release. Code will be made publicly available upon paper acceptance. <!-- TODO: update -->

---

## Abstract

Adverse weather conditions such as fog, rain, and snow cause contrast reduction, detail blurring, and target occlusion in aerial images of high-voltage transmission lines, significantly degrading the accuracy of insulator defect detection and segmentation models. Existing enhancement methods suffer from three core limitations: (1) domain shift and high inference overhead in two-stage "restoration-then-perception" pipelines; (2) underutilization of the physical characteristics of degradation in pure perception schemes; and (3) narrow task adaptability with homogeneous constraints in joint learning frameworks.

This paper proposes a **multi-task general perception enhancement network guided by dynamic degradation transmission priors**. Following the paradigm of *"auxiliary supervision during training, pure deployment at inference"*, plug-and-play dual auxiliary branches (lightweight restoration and weather classification) are built on a shared backbone and can be completely removed during inference, achieving **zero additional computational overhead**. A **Degradation Transmission Map (DT-Map)** is introduced as the constraint carrier with differentiated priors for four weather types, and a lightweight **NAF-Mini** unit is designed to reduce training overhead. The network simultaneously supports **object detection, semantic segmentation, and instance segmentation**.

---

## Main Contributions

1. **Three-task general zero-overhead plug-and-play architecture.** For the first time, the joint enhancement scheme for adverse weather is extended from single object detection to three tasks: object detection, semantic segmentation, and instance segmentation. Adopting *"shared backbone + plug-and-play dual auxiliary branches"*, the restoration and weather-classification branches only participate in feature optimization during training and can be completely removed at inference — parameters and computation stay fully consistent with the baseline.

2. **DT-Map dynamic prior constraint mechanism based on physical characteristics.** The Degradation Transmission Map (DT-Map) serves as a latent-variable constraint carrier that uniformly characterizes pixel-level degradation severity. Four **differentiated prior constraints** are designed according to the physical degradation characteristics of four weather types:

   | Weather | Physical property | DT-Map prior constraint |
   |---|---|---|
   | Clear | No degradation | All-zero suppression (L1 norm) |
   | Fog | Global, gentle atmospheric scattering | Global smoothing (total variation) |
   | Rain | Sparse sharp streaks | Sparse sharp-edge constraints (L1 + gradient) |
   | Snow | Block-like diffusion | Block-like diffusion (large receptive-field TV via dilated convolution) |

   The constraint type is selected dynamically by the weather-classification branch, preserving strong adaptability even in **mixed degradation** scenarios.

3. **NAF-Mini lightweight degradation restoration unit.** A lightweight transformation of NAFNet via three optimizations — depthwise separable convolution replacement, simplified channel attention, and streamlined network depth — greatly reduces training memory overhead while retaining restoration guidance for rain-fog-snow composite degradation.

4. **Multi-dimensional sufficient experimental verification.** Systematic experiments on **five insulator defect datasets** across three tasks, covering synthetic single weather, real complex weather, extreme mixed weather, and public benchmarks, with quantitative comparison, qualitative visualization, module ablation, and efficiency analysis.

---

## Method

### Overall Architecture

The network follows the design paradigm of **"training-time auxiliary supervision, inference-time pure deployment"**:

- **Training stage** — four parts: shared feature encoder, native perception module, degradation restoration branch, and weather classification branch. All branches share the multi-scale encoder features and are jointly optimized via multi-loss supervision.
- **Inference stage** — only the perception model is retained; the two auxiliary branches are entirely removed. The deployed model has exactly the same parameter count, computational complexity, and inference speed as the native baseline.

```
                        ┌──────────────────────────── Training only ────────────────────────────┐
  Weather-degraded ──► │  Shared Feature Encoder ──► Perception Head (det / seg / ins-seg)      │
       image           │        │  │  │                                                          │
                       │        ▼  ▼  ▼                                                          │
                       │  NAF-Mini Restoration Branch ──► Restored image + DT-Map (MSE + priors) │
                       │  Weather Classification Branch ──► clear / fog / rain / snow (CE)       │
                       └─────────────────────────────────────────────────────────────────────────┘
                                     │ Inference: keep only the perception model
                                     ▼
                        Same params / FLOPs / speed as the native baseline
```

*Detailed architecture is shown in Figure 1 of the paper.*

### Key Components

- **Shared feature encoder**: backbone-agnostic; any hierarchical encoder (CNN / Transformer) that outputs multi-scale features (1/4, 1/8, 1/16, 1/32) can be seamlessly embedded. Downstream heads are used as-is without any modification.
- **NAF-Mini restoration branch**: multi-scale encoder-decoder built from NAF-Mini units; outputs (a) a pixel-level restored clear image (MSE loss) and (b) a single-channel DT-Map carrying differentiated structural priors. Its gradient signal is back-propagated to the shared encoder to learn degradation-invariant features.
- **Weather classification branch**: global average pooling + two fully connected layers, outputting a 4-class distribution (clear/fog/rain/snow) via cross-entropy; its predicted category switches the DT-Map prior constraint.
- **Total loss**: `L = L_task + λ_r · L_restoration + λ_c · L_classification` (task loss auto-matches detection / semantic segmentation / instance segmentation).

---

## Results

### Object Detection (mAP@0.5, mixed weather unless noted)

| Dataset | YOLOv11s (baseline) | Ours (YOLOv11s) | Gain |
|---|---|---|---|
| RealInsul-Pair | 73.5 | **78.0** | **+4.5** |
| SynInsul-Weather (mixed subset) | 68.6 | **76.2** | **+7.6** |
| IDID_weather (mixed subset) | 61.4 | **69.1** | **+7.7** |
| SFID (public, foggy) | 70.6 | **77.3** | **+6.7** |
| SYNTHIDIA (public, cross-domain) | 69.2 | **75.2** | **+6.0** |

- On RealInsul-Pair, ours outperforms the SOTA joint-learning method UniDet-D by **+1.2%** (78.0 vs 76.8).
- Under clear weather, accuracy stays consistent with the native baseline (84.2 vs 83.1), verifying the all-zero suppression constraint avoids over-restoration.

### Instance Segmentation (Mask AP@0.5, mixed weather)

| Dataset | YOLOv11-seg (baseline) | Ours | Gain |
|---|---|---|---|
| RealInsul-Pair | 71.3 | **76.1** | **+4.8** |
| SynInsul-Weather | 69.1 | **74.5** | **+5.4** |

### Semantic Segmentation (mIoU, mixed weather)

| Model | RealInsul-Pair | SynInsul-Weather |
|---|---|---|
| MaskFormer (baseline) | 74.3 | 71.0 |
| MaskFormer (Ours) | **76.8** | **73.6** |
| Mask2Former (baseline) | 75.7 | 71.2 |
| Mask2Former (Ours) | **77.2** | **73.8** |

### Inference Efficiency (640×640 input)

| Model | Params (M) | FLOPs (G) | FPS |
|---|---|---|---|
| YOLOv11 (baseline) | 5.7 | 12.1 | 112 |
| Restormer two-stage scheme | 32.5 | 115.3 | 41 |
| DTRDNet (inference mode) | 8.6 | 18.3 | 89 |
| **Ours (full training model)** | 9.2 | 18.7 | 76 |
| **Ours (inference model)** | **5.7** | **12.1** | **112** |

> The inference model achieves **"improved accuracy, unchanged speed"** — identical params/FLOPs/FPS to the native baseline, directly embeddable into existing deployed models without changing the inference framework or hardware.

### Ablation Study (SynInsul-Weather mixed subset)

| Variant | Det. mAP@0.5 | Det. mAP@0.5:0.95 | Sem. Seg. mIoU | Ins. Seg. Mask AP@0.5 |
|---|---|---|---|---|
| A: Baseline (YOLOv11s) | 68.6 | 44.7 | 71.8 | 69.1 |
| B: + NAFNet branch (MSE only) | 72.1 | 47.9 | 74.9 | 72.5 |
| C: + NAF-Mini branch (MSE only) | 73.2 | 49.0 | 75.7 | 73.5 |
| D: + DT-Map differentiated priors | 75.7 | 51.9 | 77.6 | 75.9 |
| E: Full model (D + weather classifier) | **76.2** | **52.8** | **78.0** | **76.5** |

---

## Installation

```bash
# Python >= 3.8, PyTorch 2.1, CUDA 12.1 (verified on Ubuntu 22.04)
conda create -n dtmap python=3.9 -y
conda activate dtmap
pip install torch==2.1.* torchvision --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/<your-org>/<your-repo>.git   # TODO: replace
cd <your-repo>
pip install -r requirements.txt
```

---

## Datasets

Experiments are conducted on **five insulator defect datasets**:

| Dataset | #Images | Resolution | Native Annotation | Weather |
|---|---|---|---|---|
| **RealInsul-Pair** (self-built, real) | 398 | 1024×1024 | Instance mask (convertible to det/seg) | Clear:Fog:Rain:Snow = 3:1:1:1 |
| **SynInsul-Weather** (self-built, synthetic) | 1,600 | 1024×1024 | Instance mask (convertible to det/seg) | Clear:Fog:Rain:Snow = 1:1:1:1 (4 subsets, incl. mixed) |
| **SFID** (public) | 13,718 | varying | Detection box | Clear:Fog = 1:1 |
| **SYNTHIDIA** (public, `domain_syn`) | 22,000 | varying | Detection box | Clear:Fog:Rain:Snow = 1:1:1:1 |
| **IDID_weather** (public, real) | 396 | 640×640 | Detection box | Clear:Fog:Rain:Snow = 1:1:1:1 |

- The self-built datasets (RealInsul-Pair, SynInsul-Weather) are not publicly available due to project constraints and will be made available **on reasonable request** (contact the corresponding author).
- Train/test split is **8:2** for the self-built datasets.

---

## Usage

### Training

```bash
# Example: object detection with YOLOv11s on SynInsul-Weather (mixed subset)
python train.py --task detect --model yolov11s --dataset syninsul --weather mixed \
                --batch-size 16 --epochs 150 --lr 0.0001 --wd 0.0005
```

### Inference (deployment mode)

```bash
# Auxiliary branches are automatically removed after training
python infer.py --task detect --model yolov11s --weights path/to/weights.pt \
                --source path/to/images --img-size 640
```

### Key training hyperparameters

| Setting | Value |
|---|---|
| Optimizer / LR / Scheduler | AdamW / 1e-4 / cosine annealing |
| Batch size / Epochs / Weight decay | 16 / 150 / 5e-4 |
| Data augmentation | random horizontal flip, random crop & scale, brightness/contrast perturbation, Gaussian noise |
| Disabled augmentation | Mosaic, Mixup (to avoid mixed-weather feature interference) |

---

## Repository Structure

```
├── configs/          # model & training configs
├── datasets/         # dataset loading / generation scripts
├── models/           # backbone + NAF-Mini branch + DT-Map constraints + heads
├── tools/
│   ├── train.py      # training entry
│   └── infer.py      # inference entry (auxiliary branches removed)
├── docs/figs/        # paper figures (architecture, results)  # TODO
├── requirements.txt
└── README.md
```

<!-- TODO: fill in the actual structure after the code is released -->

---

## TODO

- [ ] Release training / inference code
- [ ] Release pretrained weights
- [ ] Add reproduction instructions for each dataset
- [ ] Extend to extreme mixed degradation scenarios, unsupervised / semi-supervised paradigms, and multi-modal perception (as discussed in the paper's future work)

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{song2026degradation,
  title   = {Degradation Transmission Prior-Guided All-Weather Perception Network for Insulator Defects},
  author  = {Song, Xiaoyang and Liu, Liqun and Wu, Qingfeng and Yang, Kai and Yang, Jiebao and Zhang, Jianfeng},
  journal = {<!-- TODO: journal name -->},
  year    = {2026},
  note    = {<!-- TODO: volume/pages/DOI -->}
}
```

---

## Contact & Acknowledgement

- **Corresponding author**: Liqun Liu — Liulq_1976@163.com
- Affiliations: School of Electronic Information Engineering, Taiyuan University of Science and Technology; Department of Automation, Taiyuan Institute of Technology.

This work was supported by the National Natural Science Foundation of China (Youth Program, Grant No. 61703297), the General Project of Shanxi Basic Research Program (Grant No. 202203021221153), the Scientific and Technological Innovation Project of Higher Education Institutions of Shanxi Province (Grant No. 2023L185), and the Shanxi Provincial Project (Grant No. 202303021222164).

No conflicts of interest are reported by all contributing authors.
