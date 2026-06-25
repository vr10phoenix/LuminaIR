import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from swinir import SwinIR_Thermal
from restormer import pre_restormer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ------------------------------------------------------------
# 1. Load the trained SwinIR model (now from the root folder)
# ------------------------------------------------------------
model_path = 'swinir_thermal_v1.pth'   # <-- It's in the root, not inside IR_SuperResolution
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please run training first.")

swinir = SwinIR_Thermal(upscale=2, dim=32, depths=[4, 4, 4, 4], num_heads=8).to(device)
swinir.load_state_dict(torch.load(model_path, map_location=device))
swinir.eval()

# 2. Load the frozen Restormer denoiser (also in the root)
restormer = pre_restormer(pretrained_weights_path="real_denoising.pth", device=device)
restormer.to(device)
restormer.eval()

# ------------------------------------------------------------
# 3. Pick a sample patch (adjust sample_idx as needed)
# ------------------------------------------------------------
sample_idx = 110  # This was in the validation set   
base_path = f"Pipeline/Preprocessing/patches/City_1/sample_{sample_idx}"
lr_path = os.path.join(base_path, "tir_200m.npy")
hr_path = os.path.join(base_path, "tir_100m_512.npy")

if not os.path.exists(lr_path) or not os.path.exists(hr_path):
    raise FileNotFoundError(f"Sample {sample_idx} not found. Check the path.")

lr_np = np.load(lr_path)
hr_np = np.load(hr_path)

print(f"Loaded sample_{sample_idx} -> Input: {lr_np.shape}, Target: {hr_np.shape}")

lr_tensor = torch.from_numpy(lr_np).float().unsqueeze(0).unsqueeze(0).to(device)

# ------------------------------------------------------------
# 4. Run Inference
# ------------------------------------------------------------
with torch.no_grad():
    clean_lr = restormer(lr_tensor)
    sr_output = swinir(clean_lr)

sr_np = sr_output.squeeze().cpu().numpy()

# ------------------------------------------------------------
# 5. Visualize
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(lr_np, cmap='viridis')
axes[0].set_title('Low-Res Input (256x256)')
axes[0].axis('off')

axes[1].imshow(sr_np, cmap='viridis')
axes[1].set_title('Super-Resolved Output (512x512)')
axes[1].axis('off')

axes[2].imshow(hr_np, cmap='viridis')
axes[2].set_title('Ground Truth (512x512)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('inference_result.png', dpi=150)
plt.show()

print("\n✅ Inference complete! Check 'inference_result.png'.")
print(f"PSNR (approx): {20 * np.log10(1.0 / np.mean((sr_np - hr_np)**2)):.2f} dB")