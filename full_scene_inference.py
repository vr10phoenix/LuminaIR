import os
import sys

# ---------- Add IR_SuperResolution folder to Python path ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
ir_folder = os.path.join(script_dir, 'IR_SuperResolution')
if os.path.exists(ir_folder):
    sys.path.insert(0, ir_folder)
    print(f"Added to path: {ir_folder}")
else:
    print(f"ERROR: {ir_folder} does not exist!")
    sys.exit(1)

import torch
import numpy as np
import time
from tqdm import tqdm
from swinir import SwinIR_Thermal
from restormer import pre_restormer

# ---------- SETUP ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------- PATHS ----------
MODEL_PATH = "swinir_thermal_v1.pth"
IR_ALIGNED_PATH = "Pipeline/Preprocessing/processed_data/ir_aligned.npy"
OUTPUT_PATH = "final_superres_thermal.npy"

# ---------- PATCH SIZE ----------
LR_PATCH_SIZE = 256
HR_PATCH_SIZE = LR_PATCH_SIZE * 2

# ---------- LOAD MODELS ----------
print("Loading models...")
swinir = SwinIR_Thermal(upscale=2, dim=32, depths=[4, 4, 4, 4], num_heads=8).to(device)
swinir.load_state_dict(torch.load(MODEL_PATH, map_location=device))
swinir.eval()

restormer = pre_restormer(pretrained_weights_path="real_denoising.pth", device=device)
restormer.to(device)
restormer.eval()

# ---------- LOAD & NORMALIZE FULL IR IMAGE ----------
print(f"Loading full IR image from {IR_ALIGNED_PATH}...")
ir_full = np.load(IR_ALIGNED_PATH)
print(f"Original size: {ir_full.shape}")

# Normalize exactly like training (ignore 0s)
valid = ir_full[ir_full > 0]
if len(valid) == 0:
    valid = ir_full.flatten()
ir_min, ir_max = valid.min(), valid.max()
print(f"Normalizing with min={ir_min:.2f}, max={ir_max:.2f}")

ir_full = (ir_full - ir_min) / (ir_max - ir_min + 1e-8)
ir_full = np.clip(ir_full, 0.0, 1.0)

ir_full_tensor = torch.from_numpy(ir_full).float().unsqueeze(0).unsqueeze(0).to(device)

# ---------- PAD ----------
H, W = ir_full.shape
pad_h = (LR_PATCH_SIZE - H % LR_PATCH_SIZE) % LR_PATCH_SIZE
pad_w = (LR_PATCH_SIZE - W % LR_PATCH_SIZE) % LR_PATCH_SIZE

if pad_h > 0 or pad_w > 0:
    print(f"Padding: +{pad_h} rows, +{pad_w} cols")
    ir_padded = torch.nn.functional.pad(ir_full_tensor, (0, pad_w, 0, pad_h), mode='reflect')
else:
    ir_padded = ir_full_tensor

H_pad, W_pad = ir_padded.shape[2], ir_padded.shape[3]
print(f"Padded size: {H_pad} x {W_pad}")

# ---------- SLIDING WINDOW ----------
num_h = H_pad // LR_PATCH_SIZE
num_w = W_pad // LR_PATCH_SIZE
print(f"Total patches: {num_h} x {num_w} = {num_h * num_w}")

out_h = H_pad * 2
out_w = W_pad * 2
output = np.zeros((out_h, out_w), dtype=np.float32)

print("Starting patch-wise inference...")
start_time = time.time()

with torch.no_grad():
    for i in tqdm(range(num_h), desc="Rows"):
        for j in range(num_w):
            y_start = i * LR_PATCH_SIZE
            y_end = y_start + LR_PATCH_SIZE
            x_start = j * LR_PATCH_SIZE
            x_end = x_start + LR_PATCH_SIZE

            lr_patch = ir_padded[:, :, y_start:y_end, x_start:x_end]

            clean_patch = restormer(lr_patch)
            sr_patch = swinir(clean_patch)

            out_y_start = i * HR_PATCH_SIZE
            out_y_end = out_y_start + HR_PATCH_SIZE
            out_x_start = j * HR_PATCH_SIZE
            out_x_end = out_x_start + HR_PATCH_SIZE

            output[out_y_start:out_y_end, out_x_start:out_x_end] = sr_patch.squeeze().cpu().numpy()

# ---------- REMOVE PADDING ----------
out_pad_h = pad_h * 2
out_pad_w = pad_w * 2
if out_pad_h > 0 or out_pad_w > 0:
    print(f"Removing padding: -{out_pad_h} rows, -{out_pad_w} cols")
    output = output[:H*2, :W*2]

# ---------- SAVE ----------
print(f"Saving final super-resolved image to {OUTPUT_PATH}...")
np.save(OUTPUT_PATH, output)
print(f"Final shape: {output.shape}")
print(f"Total time: {time.time() - start_time:.2f} seconds")

try:
    import tifffile
    tifffile.imwrite("final_superres_thermal.tif", output)
    print("Also saved as final_superres_thermal.tif")
except ImportError:
    print("Note: tifffile not installed. Only .npy saved.")