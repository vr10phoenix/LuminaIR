# Restormer Initialization and Transfer Learning Framework

## Overview

This module implements a two-stage deep learning framework for Thermal Infrared (TIR) image super-resolution by integrating the complementary strengths of Restormer and SwinIR. The framework is designed to reconstruct high-resolution thermal imagery from low-resolution Landsat observations while preserving both radiometric consistency and fine structural details.

The proposed approach decomposes the reconstruction task into two sequential stages. In the first stage, a pre-trained Restormer serves as a feature restoration network, suppressing sensor noise, mitigating degradation artifacts, and enhancing the underlying thermal representations. Rather than training this network from scratch, transfer learning is employed by adapting weights pre-trained on large-scale image restoration datasets, after which the network is frozen to preserve its learned feature extraction capabilities.

The restored thermal representation is subsequently passed to a SwinIR-based super-resolution network, which performs hierarchical feature extraction using shifted-window self-attention and reconstructs the high-resolution thermal image. By receiving cleaner and semantically richer inputs from Restormer, SwinIR can concentrate exclusively on recovering high-frequency spatial information instead of simultaneously learning denoising and reconstruction.

---

## Architectural Components

![System_Architecture](https://github.com/vr10phoenix/LuminaIR/blob/main/assets/System_architecture.png)

The implementation consists of four principal modules:

* Multi-DConv Head Transposed Self-Attention (MDTA)
* Gated-Dconv Feed Forward Network (GDFN)
* Transformer Block
* Restormer Backbone

Each module closely follows the architecture proposed in the original Restormer publication while maintaining a lightweight configuration suitable for thermal image restoration.

---

### Multi-DConv Head Transposed Self-Attention (MDTA)

The attention module is responsible for modelling long-range spatial dependencies without incurring the computational cost of conventional self-attention.

Unlike standard Vision Transformers, MDTA performs attention across feature channels rather than directly over spatial tokens. Query, Key and Value representations are first generated through point-wise convolution followed by depth-wise convolution, allowing simultaneous extraction of local texture information and global contextual relationships.

The implementation contains a safety assertion requiring

```python
channels % num_heads == 0
```

to ensure that channel dimensions are evenly partitioned among all attention heads.

Learnable temperature parameters are introduced independently for every attention head, enabling adaptive scaling of attention scores during optimization.

---

### Gated-Dconv Feed Forward Network (GDFN)

The feed-forward network extends the conventional transformer MLP by introducing spatial awareness through depth-wise convolution and gating.

Instead of a simple two-layer perceptron, feature maps undergo

1. Channel expansion
2. Depth-wise convolution
3. Feature splitting
4. GELU activation
5. Multiplicative gating
6. Linear projection back to the original embedding dimension

This gated mechanism selectively suppresses irrelevant feature responses while emphasizing informative thermal structures.

The expansion factor is fixed at

```text
2.66
```

which follows the configuration reported in the original Restormer architecture.

---

### Transformer Block

Each transformer block consists of two residual submodules:

```
Layer Normalization
        ↓
MDTA
        ↓
Residual Addition
        ↓
Layer Normalization
        ↓
GDFN
        ↓
Residual Addition
```

Layer Normalization is applied across channel dimensions, requiring temporary permutation between

```
(B,C,H,W)
```

and

```
(B,H,W,C)
```

before normalization and restoration to PyTorch's native tensor format.

Residual learning is employed after both the attention and feed-forward stages to facilitate stable optimization in deep transformer architectures.

---

#### Restormer 

The implemented Restormer backbone follows a simplified encoder architecture composed of

* Input embedding convolution
* Four transformer blocks
* Output reconstruction convolution
* Global residual connection

Unlike the complete encoder-decoder Restormer architecture, this implementation functions as a compact restoration network suitable for feature enhancement prior to super-resolution.

The network configuration is summarized below.

| Parameter           | Value |
| ------------------- | ----: |
| Input Channels      |     1 |
| Output Channels     |     1 |
| Embedding Dimension |    48 |
| Transformer Blocks  |     4 |
| Attention Heads     |     8 |

---

### Transfer Learning Strategy

A pre-trained Restormer checkpoint trained on RGB denoising is used as the initialization source.

Since the original model expects three-channel input images, direct weight loading into a single-channel network is not possible.

Instead, the initialization proceeds in three stages.

#### Stage 1

Instantiate a temporary Restormer with

```
Input Channels = 3
```

to exactly match the pre-trained checkpoint.

---

#### Stage 2

Load all available weights using

```python
strict=False
```

allowing partial parameter matching while avoiding incompatibilities introduced by architectural modification.

---

#### Stage 3

Replace the RGB embedding layer with a newly constructed single-channel convolution.

The new convolutional kernel is initialized by averaging the original RGB filters

[
W_{TIR}=\frac{W_R+W_G+W_B}{3}
]

thereby preserving the learned edge detectors, low-frequency filters, and structural priors embedded within the original convolutional kernels.

Only the first convolutional layer is modified.

Every remaining parameter—including transformer attention layers, feed-forward networks, normalization layers, and reconstruction head—is retained from the pre-trained checkpoint.

---

#### Weight Initialization Pipeline

```
Official Restormer Checkpoint
            │
            ▼
Instantiate RGB Restormer
            │
            ▼
Load Pre-trained Parameters
            │
            ▼
Extract First Convolution
            │
            ▼
Average RGB Kernels
            │
            ▼
Construct 1-Channel Embedding Layer
            │
            ▼
Replace Original Layer
            │
            ▼
Move Model to GPU
```

---

#### Residual Learning

The network predicts a residual correction rather than reconstructing the thermal image directly.

Let

[I_{LR}]

represent the input image and

[F(I_{LR})]

denote the learned restoration function.

The final prediction is

[I_{out}=I_{LR}+F(I_{LR})]

This residual formulation encourages the network to learn only missing high-frequency information while preserving the original thermal intensity distribution.

---

# Design Considerations

Several implementation decisions distinguish this adaptation from the original RGB Restormer.

* Single-channel thermal imagery replaces RGB input.
* RGB embedding weights are transformed rather than randomly initialized.
* The remaining transformer hierarchy remains untouched.
* Weight loading is tolerant to architectural differences through non-strict parameter matching.
* The resulting model serves as a feature restoration stage within a larger thermal super-resolution framework.

---

# Integration within the Super-Resolution Pipeline

The initialized Restormer functions as the first stage of the restoration pipeline.

```
Low-Resolution Thermal Image
            │
            ▼
Pre-trained Restormer
            │
            ▼
Noise Suppression
Texture Enhancement
Feature Restoration
            │
            ▼
SwinIR
            │
            ▼
High-Resolution Thermal Reconstruction
```

The output generated by Restormer provides structurally cleaner and semantically richer feature maps, enabling the subsequent SwinIR module to concentrate on high-frequency reconstruction rather than denoising.

---

### Results:
Achieved global features preservations and global resolution with reasonalbly good accuracy.

![Restormer Output](https://github.com/vr10phoenix/LuminaIR/blob/main/assets/restormer_output.jpeg)


# SwinIR Training Framework for Thermal Infrared Super-Resolution

## Overview

This module implements the complete supervised training framework for thermal infrared (TIR) image super-resolution. The objective is to reconstruct high-resolution thermal observations from low-resolution Landsat imagery by combining a frozen Restormer feature restoration network with a trainable SwinIR reconstruction network.

The implementation incorporates mixed-precision training, gradient accumulation, adaptive learning-rate scheduling, checkpoint management, and an edge-aware reconstruction loss specifically designed for satellite thermal imagery.

---

## Methodological Motivation

Thermal infrared satellite imagery exhibits several characteristics that distinguish it from conventional natural images.

* Low spatial resolution.
* Weak local contrast.
* Sensor-induced noise.
* Missing observations (NoData regions).
* Interpolation artefacts introduced during preprocessing.

Directly training a super-resolution network on such data forces the model to simultaneously learn denoising, feature extraction, and spatial reconstruction.

Instead, the proposed framework first removes degradations using a pre-trained Restormer and subsequently performs spatial reconstruction through SwinIR. This staged optimization strategy allows the super-resolution network to focus exclusively on recovering high-frequency structural information.

---

### Overall Training Architecture

The complete training pipeline is illustrated below.

```text
Low-Resolution Thermal Image
                │
                ▼
      Multi-City Dataset Loader
                │
                ▼
      Frozen Restormer
 (Feature Restoration Stage)
                │
                ▼
 Enhanced Thermal Representation
                │
                ▼
      SwinIR Super-Resolution
                │
                ▼
 Predicted High-Resolution Image
                │
                ▼
 Edge-Aware Reconstruction Loss
                │
                ▼
        Backpropagation
                │
                ▼
 Update SwinIR Parameters Only
```

No gradients are propagated through Restormer during optimization.

---

### Dataset Preparation

Training samples are obtained from the `MasterMultiCityDataset`, which aggregates paired thermal observations from multiple geographic regions into a unified dataset.

The dataset is partitioned into training and validation subsets using an 80–20 split.

A fixed random seed is employed during dataset partitioning to guarantee reproducibility across experiments.

```text
Complete Dataset
        │
        ├──────────► Training Set (80%)
        │
        └──────────► Validation Set (20%)
```

---

### Data Loading Strategy

Mini-batches are generated using PyTorch's `DataLoader`.

The implementation employs

* mini-batch training,
* asynchronous memory pinning,
* persistent worker processes,
* randomized training batches, and
* deterministic validation ordering.

This configuration minimizes data-loading overhead while maintaining reproducibility during evaluation.

---

### Transfer Learning Strategy

The framework assumes that Restormer has already been pre-trained on large-scale image restoration tasks.

During initialization

1. the pretrained checkpoint is loaded,
2. parameters are transferred to the restoration network,
3. the network is switched to evaluation mode, and
4. every parameter is frozen.

Consequently,

[
\frac{\partial \mathcal{L}}{\partial \theta_{Restormer}} = 0
]

throughout optimization.

Only SwinIR participates in gradient updates.

This considerably reduces computational cost while preventing catastrophic forgetting of previously learned restoration representations.

---

### SwinIR Optimization

The SwinIR network constitutes the only trainable component of the framework.

For every mini-batch,

```text
Low Resolution Image
        │
        ▼
Restormer
        │
        ▼
Enhanced Image
        │
        ▼
SwinIR
        │
        ▼
Prediction
        │
        ▼
Loss Computation
        │
        ▼
Gradient Update
```

The optimization objective is therefore limited exclusively to learning spatial reconstruction.

---

### Loss Function Design

Accurate reconstruction of thermal imagery requires preservation of both radiometric fidelity and structural boundaries.

To address these complementary objectives, the proposed loss combines a robust pixel reconstruction term with an edge-preservation constraint.

The total optimization objective is defined as

[
\mathcal{L}
===========

\mathcal{L}*{Charbonnier}
+
\lambda
\mathcal{L}*{Edge}
]

where

[
\lambda = 0.2
]

controls the contribution of edge preservation.

---

#### Charbonnier Reconstruction Loss

Instead of Mean Squared Error, the framework employs the Charbonnier loss

[
\mathcal{L}_{Char}
==================

\sqrt{(I_p-I_t)^2+\epsilon}
]

The Charbonnier formulation behaves similarly to an L1 loss while remaining differentiable everywhere, resulting in improved numerical stability and reduced sensitivity to outliers.

---

#### Edge Preservation Loss

Thermal images frequently lose fine object boundaries during super-resolution.

To preserve structural information, Laplacian filtering is applied to both prediction and ground-truth images.

Edge consistency is then measured using the same Charbonnier formulation.

This encourages accurate reconstruction of

* building boundaries,
* road networks,
* water edges,
* vegetation transitions, and
* other high-frequency thermal structures.

---

#### Mask-Aware Optimization

Satellite imagery commonly contains regions without valid observations.

Pixels corresponding to

* image padding,
* NoData values,
* interpolation artefacts,

are excluded from loss computation through a binary validity mask.

Consequently,

only physically meaningful observations contribute to gradient estimation.

This significantly improves optimization robustness when training on geospatial datasets.

---

#### Mixed Precision Training

Training is performed using Automatic Mixed Precision (AMP).

The framework combines

* FP16 arithmetic for computationally intensive operations,
* FP32 accumulation where numerical precision is required, and
* dynamic gradient scaling to prevent underflow.

This substantially reduces GPU memory consumption while maintaining numerical stability.

---

#### Gradient Accumulation

Transformer architectures typically require large effective batch sizes.

To accommodate limited GPU memory, gradients are accumulated across multiple iterations before each optimization step.

With

```text
ACCUM_STEPS = 4
```

the effective batch size becomes

[
Batch_{effective}
=================

Batch_{physical}
\times
4
]

allowing stable optimization without increasing memory requirements.

---

#### Gradient Stabilization

Before every optimizer update,

gradient norms are clipped to

[
|g|_2 \le 1
]

to suppress exploding gradients that occasionally arise during transformer training.

Gradient clipping improves convergence stability, particularly during the early optimization stages.

---

#### Learning Rate Scheduling

Optimization employs the AdamW optimizer together with cosine annealing.

The learning rate gradually decreases following

[
\eta_t
======

\eta_{min}
+
\frac{1}{2}
(\eta_{max}-\eta_{min})
\left(
1+
\cos
\frac{\pi t}{T}
\right)
]

This scheduling strategy provides smoother convergence than abrupt step-based decay and is well suited to transformer optimization.

---


#### Checkpoint Management

The framework maintains two independent checkpoints.

##### Latest Checkpoint

Saved after every epoch.

Contains

* SwinIR parameters,
* optimizer state,
* scheduler state,
* gradient scaler state,
* current epoch,
* best validation score.

This checkpoint enables seamless continuation after interrupted training.

---

##### Best Model

Whenever validation loss improves,

the current SwinIR weights are stored separately.

Only the model with the lowest validation loss is retained as the final inference network.

---

#### Early Stopping

Training terminates automatically when validation performance fails to improve for a predefined number of epochs.

This prevents unnecessary optimization after convergence while reducing the risk of overfitting.

---

#### Experiment Logging

Training statistics are recorded after every epoch.

Each log entry contains

* epoch number,
* training loss,
* validation loss,
* learning rate,
* execution time.

The resulting CSV file provides a complete history of the optimization process and facilitates subsequent performance analysis.

---

## Results
the following Results were achieved :
![Raw data and Ground truth analysis](https://github.com/vr10phoenix/LuminaIR/blob/main/assets/tir_comparision.png)

``` 
            RESULTS
Interpolation time : 0.1315 seconds
Baseline PSNR      : 43.95 dB
Baseline SSIM      : 0.9971

```

Output after passing through swinIR via Restormer
![SwinIR_Output] (https://github.com/vr10phoenix/LuminaIR/blob/main/assets/tir_swin_result.png)

```
              RESULTS
Interpolation time : 0.1315 seconds
PSNR               : 28.29 dB
SSIM               : 0.9691

```
Hence verifing clinical precision in resolving Thermal Infra-Red Image Super Resolution.

# Summary

This training framework implements a modular two-stage learning strategy for thermal infrared image super-resolution. By decoupling feature restoration from spatial reconstruction, freezing the pretrained Restormer backbone, and optimizing only the SwinIR network using a mask-aware edge-preserving objective, the framework achieves stable convergence while preserving both radiometric accuracy and structural fidelity. The incorporation of mixed-precision computation, gradient accumulation, adaptive learning-rate scheduling, automatic checkpointing, and reproducible experimentation establishes a robust foundation for large-scale satellite image super-resolution research.



