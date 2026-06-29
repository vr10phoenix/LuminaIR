import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def evaluate_super_resolution(test_ir_path, ground_truth_path, swinir_weights, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"[EVALUATION] Initializing testing suite on {device}")
    
    # 1. Load the trained SwinIR
    swinir = SwinIR_Thermal(upscale=2).to(device)
    swinir.load_state_dict(torch.load(swinir_weights, map_location=device))
    swinir.eval()
    
    # 2. Load the Raw Input (200m) and Ground Truth (100m)
    raw_matrix = np.load(test_ir_path)
    gt_matrix = np.load(ground_truth_path)
    
    # Normalize inputs to [0, 1] for accurate metric calculation
    raw_matrix = (raw_matrix - np.min(raw_matrix)) / (np.max(raw_matrix) - np.min(raw_matrix) + 1e-8)
    gt_matrix = (gt_matrix - np.min(gt_matrix)) / (np.max(gt_matrix) - np.min(gt_matrix) + 1e-8)
    
    # Convert input to PyTorch tensor
    tensor_in = torch.from_numpy(raw_matrix).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # 3. Measure Inference Speed
    with torch.no_grad():
        start_time = time.time()
        sr_tensor = swinir(tensor_in)
        inference_time = (time.time() - start_time) * 1000 # Convert to milliseconds
        
    # Extract prediction
    sr_matrix = sr_tensor.squeeze().cpu().numpy()
    sr_matrix = np.clip(sr_matrix, 0, 1) # Ensure values remain in standard bounds
    
    # 4. Calculate Mathematical Quality Metrics
    # Higher PSNR = Less Noise/Distortion
    calc_psnr = psnr(gt_matrix, sr_matrix, data_range=1.0)
    # Higher SSIM (closer to 1.0) = Perfectly matched structural edges
    calc_ssim = ssim(gt_matrix, sr_matrix, data_range=1.0)
    
    print("\n" + "="*40)
    print("      FINAL EVALUATION METRICS")
    print("="*40)
    print(f"PSNR (Peak Signal-to-Noise): {calc_psnr:.2f} dB")
    print(f"SSIM (Structural Similarity): {calc_ssim:.4f}")
    print(f"Inference Latency per Tile:  {inference_time:.2f} ms")
    print("="*40 + "\n")
    
    # 5. Visual Proof
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(raw_matrix, cmap='inferno')
    axes[0].set_title(f"Raw Input (256x256)")
    axes[0].axis('off')
    
    axes[1].imshow(sr_matrix, cmap='inferno')
    axes[1].set_title(f"SwinIR Prediction (512x512)\nPSNR: {calc_psnr:.2f} | SSIM: {calc_ssim:.4f}")
    axes[1].axis('off')
    
    axes[2].imshow(gt_matrix, cmap='inferno')
    axes[2].set_title(f"Ground Truth Target (512x512)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Execute Evaluation
# Replace with one of your sample paths
TEST_SAMPLE = "/content/drive/MyDrive/Projects/LuminaIR/output/patches/bangalore/sample_000/tir_200m.npy"
TEST_TARGET = "/content/drive/MyDrive/Projects/LuminaIR/output/patches/bangalore/sample_000/tir_100m_512.npy"
WEIGHTS = "swinir_thermal_direct_v1.pth"

# evaluate_super_resolution(TEST_SAMPLE, TEST_TARGET, WEIGHTS)