import numpy as np
import config

def calculate_global_stats(array):
    valid = array[array != config.NODATA_VALUE]
    if len(valid) == 0:
        return 0, 1
    return valid.min(), valid.max()

def normalize_patch(patch, g_min, g_max):
    patch_float = patch.astype(np.float32)
    norm = (patch_float - g_min) / (g_max - g_min + 1e-8)
    return np.clip(norm, 0.0, 1.0)

def run_slicer():
    print("PATCH SLICER (output: City_1/sample_X/tir_*.npy)...")
    
    rgb_master = np.load(config.PROCESSED_DATA_DIR / "rgb_aligned.npy")
    ir_master  = np.load(config.PROCESSED_DATA_DIR / "ir_aligned.npy")
    
    rgb_min, rgb_max = calculate_global_stats(rgb_master)
    ir_min, ir_max   = calculate_global_stats(ir_master)
    
    _, height, width = rgb_master.shape
    patch_size = config.PATCH_SIZE  # 512
    stride = patch_size
    
    # Output root: Luminair/patches/ (next to Data and Pipeline)
    output_root = config.PATCHES_DIR.parent / "patches"
    city_name = "City_1"
    
    saved = 0
    total = 0
    
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            total += 1
            rgb_patch = rgb_master[:, y:y+patch_size, x:x+patch_size]
            ir_patch  = ir_master[y:y+patch_size, x:x+patch_size]
            
            # Cloud check
            invalid = np.sum(ir_patch == config.NODATA_VALUE)
            if invalid / (patch_size * patch_size) > config.MAX_CLOUD_COVER:
                continue
            
            # Normalize
            ir_norm = normalize_patch(ir_patch, ir_min, ir_max)
            
            # High-Res (512x512) and Low-Res (256x256) via average pooling
            hr_patch = ir_norm
            lr_patch = hr_patch.reshape(256, 2, 256, 2).mean(axis=(1, 3))
            
            # Save in City_1/sample_XXX/
            sample_folder = output_root / city_name / f"sample_{saved+1}"
            sample_folder.mkdir(parents=True, exist_ok=True)
            
            np.save(sample_folder / "tir_100m_512.npy", hr_patch)
            np.save(sample_folder / "tir_200m.npy", lr_patch)
            
            saved += 1
            if saved % 50 == 0:
                print(f"   Saved {saved} patches...")
    
    print(f"--- DONE ---")
    print(f"Saved {saved} valid patches out of {total} raw slices.")
    print(f"Output folder: {output_root}")

if __name__ == "__main__":
    run_slicer()