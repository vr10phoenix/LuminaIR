import numpy as np
import config

def calculate_global_stats(array):
    """
    Calculates global minimum and maximum across the entire scene,
    strictly ignoring the NODATA_VALUE (0) so clouds/edges don't skew the math.
    """
    valid_pixels = array[array != config.NODATA_VALUE]
    if len(valid_pixels) == 0:
        return 0, 1 # Safety fallback
    return valid_pixels.min(), valid_pixels.max()

def normalize_patch(patch, g_min, g_max):
    """
    Deep learning models need inputs scaled between 0 and 1. 
    This applies our global physical scaling to a single patch.
    """
    patch_float = patch.astype(np.float32)
    # The small 1e-8 prevents division by zero
    normalized = (patch_float - g_min) / (g_max - g_min + 1e-8)
    
    # Clip strictly to [0.0, 1.0] to prevent anomalous spikes
    return np.clip(normalized, 0.0, 1.0)

def run_slicer():
    print("PATCH SLICER INITIATED..")
    
    # 1. Load the massive aligned arrays from the previous script
    print("Loading aligned arrays into memory...")
    rgb_master = np.load(config.PROCESSED_DATA_DIR / "rgb_aligned.npy")
    ir_master  = np.load(config.PROCESSED_DATA_DIR / "ir_aligned.npy")
    
    # 2. Calculate Global Normalization Stats
    # We MUST calculate min/max globally before slicing.
    print("Calculating Global Physical Scaling bounds...")
    rgb_min, rgb_max = calculate_global_stats(rgb_master)
    ir_min, ir_max   = calculate_global_stats(ir_master)
    
    # 3. Setup the Sliding Window
    _, height, width = rgb_master.shape
    patch_size = config.PATCH_SIZE
    stride = patch_size # Using a stride equal to patch_size ensures 0% overlap
    
    patch_count = 0
    saved_count = 0
    
    print(f"Slicing {height}x{width} grid into {patch_size}x{patch_size} patches...")
    
    #Slicing Loop (Y-axis then X-axis sliding window)
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch_count += 1
            
            # Extract the raw spatial block
            rgb_patch = rgb_master[:, y:y+patch_size, x:x+patch_size]
            ir_patch  = ir_master[y:y+patch_size, x:x+patch_size]
            
            #The Validation Gate (Check for Clouds/Black Edges)
            # Count how many pixels equal our NODATA_VALUE
            invalid_pixels = np.sum(ir_patch == config.NODATA_VALUE)
            total_pixels = patch_size * patch_size
            invalid_ratio = invalid_pixels / total_pixels
            
            # If the patch is too cloudy or sits on the black edge, skip it entirely
            if invalid_ratio > config.MAX_CLOUD_COVER:
                continue
                
            #Normalize valid patches to [0, 1]
            rgb_norm = normalize_patch(rgb_patch, rgb_min, rgb_max)
            ir_norm  = normalize_patch(ir_patch, ir_min, ir_max)
            
            #Save the perfect, clean pair to disk
            patch_name = f"patch_{saved_count:04d}"
            np.save(config.PATCHES_DIR / f"rgb_{patch_name}.npy", rgb_norm)
            np.save(config.PATCHES_DIR / f"ir_{patch_name}.npy", ir_norm)
            
            saved_count += 1
            
    # printing resultant metrics
    print(f"--- SLICING COMPLETE ---")
    print(f"Total Raw Patches Evaluated: {patch_count}")
    print(f"Valid SOTA Patches Saved:    {saved_count}")
    print(f"Garbage Discarded:           {patch_count - saved_count}")

if __name__ == "__main__":
    run_slicer()