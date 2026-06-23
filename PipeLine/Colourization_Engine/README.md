# 🔥 Semantic-Guided Thermal-to-RGB GAN

An advanced Generative Adversarial Network (GAN) pipeline designed to translate 1-channel thermal/IR imagery into fully realized 3-channel RGB images. 

Unlike standard image-to-image translation models, this architecture utilizes a **Semantic Guardrail** (powered by Segformer and DINOv2) to enforce structural integrity, and validates the realism of its generated outputs in real-time using a downstream **YOLOv11 Validator**.

---

## 🚀 Overview

Translating thermal inputs to RGB often results in hallucinated textures or loss of structural semantics. This repository solves that by combining **Pix2PixHD** principles with modern vision transformers. 

**Key Features:**
* **Zero-Shot Semantic Guardrails:** Uses pre-trained `Segformer-b0` and `DINOv2` to extract semantic masks and deep features from the clean IR input without requiring fine-tuning.
* **VRAM-Optimized Fusion:** A lightweight `SemanticFusionLayer` avoids massive one-hot encoded tensors by using shallow convolutions to fuse the 1-channel IR and 1-channel semantic mask.
* **Multi-Scale Criticism:** A Master Critic evaluates both micro-textures (512x512) and global coherence (256x256).
* **Real-Time Machine Vision Validation:** Integrates `YOLO11n` to actively test if the generated RGB images are readable by state-of-the-art object detectors.

---

## 🧠 Architecture

| Component | Class Name | Description |
| :--- | :--- | :--- |
| **Feature Extractor** | `SemanticGuardrail` | Freezes `segformer-b0` and `dinov2_vits14` to dynamically pad, process, and extract rich semantic masks and features from the IR input. |
| **Data Compressor** | `SemanticFusionLayer` | Combines thermal inputs and semantic masks using `Conv2d` and `InstanceNorm2d`, outputting a fused 64-channel tensor. |
| **The Painter** | `GlobalGenerator` | A heavy-lifting Pix2PixHD-style residual generator featuring downsampling, 6 residual blocks, and upsampling layers bounded by a `Tanh` activation `[-1, 1]`. |
| **Master Critic** | `MultiScaleDiscriminator` | Runs two identical PatchGAN discriminators at different resolutions (scales) to critique the generated outputs. |
| **Adjudicator** | `MasterLossEngine` | Balances Multi-scale LSGAN Loss (MSE) and Structural L1 Loss ($\lambda = 10.0$). |
| **Downstream QA** | `YOLOv11Validator` | A frozen YOLOv11-Nano model that evaluates the semantic readability of the generated fake RGB images during the training loop. |

---

## 📦 Prerequisites

Ensure you have a CUDA-capable GPU. The pipeline relies heavily on the Hugging Face ecosystem and Ultralytics.

```bash
pip install torch torchvision
pip install transformers
pip install ultralytics
```

## Training Configuration
The model is optimized using standard GAN hyperparameters:

- Optimizer: Adam
- Learning Rate: 0.0002
- Betas: (0.5, 0.999)
- Loss Weights: L1 Loss is scaled by a factor of 10.0 to encourage structural similarity to the ground truth.
- Epochs: 50 (Configurable in script)

## Logging Out example : 
```
[SUCCESS] Master Loss Engine initialized and deployed to GPU.
Initializing YOLO11-Nano Downstream Validator...
[SUCCESS] Optimizers initialized and locked to their respective networks.
[Epoch 0/50] [Batch 0/...] [D loss: 0.4821] [G loss: 12.3412] [G L1: 11.8591]
   -> YOLOv11 Validator spotted 3 objects in generated image.
End of Epoch 0 - Time taken: 142.50 seconds
```
## Checkpoints
To prevent data loss and allow for resuming, model weights are saved automatically every 5 epochs.
The outputs will appear in your root directory as:  
- semantic_generator_epoch_X.pth
- multiscale_critic_epoch_X.pth

## Generated output 
![output](assets/result.png)

Initial Prototype pipeline established and testing , moving towards refining the approach and fine tuning the pipeline with better loss functions and taking out errors.
