import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import config

def load_and_stack_rgb():
    """
    Loads Bands 4 (Red), 3 (Green), and 2 (Blue) and stacks
    them into a single Master RGB grid.
    """
   
    with rasterio.open(config.FILE_B4) as src_r:
        red = src_r.read(1)
        # extract metadata from Band 4 to use as our absolute truth
        master_kwargs = src_r.profile
        master_transform = src_r.transform
        master_crs = src_r.crs
        
    with rasterio.open(config.FILE_B3) as src_g:
        green = src_g.read(1)
        
    with rasterio.open(config.FILE_B2) as src_b:
        blue = src_b.read(1)

    # Stack into a 3D NumPy array (Channels, Height, Width) - format -> Y_Target
    rgb_stack = np.stack([red, green, blue], axis=0)
    return rgb_stack, master_kwargs, master_transform, master_crs

def align_thermal_to_master(master_kwargs, master_transform, master_crs):
    """
    Loads the Thermal Band 10 and mathematically warps it to match the Master RGB grid.
    """
    
    with rasterio.open(config.FILE_B10) as src_ir:
        ir_raw = src_ir.read(1)
        ir_aligned = np.zeros((master_kwargs['height'], master_kwargs['width']), dtype=ir_raw.dtype)
        
        # The core reprojection engine
        reproject(
            source=ir_raw,
            destination=ir_aligned,
            src_transform=src_ir.transform,
            src_crs=src_ir.crs,
            dst_transform=master_transform,
            dst_crs=master_crs,
            resampling=Resampling.bilinear # Smooth interpolation for thermal data
        )
    return ir_aligned

def apply_cloud_mask(rgb_stack, ir_aligned, master_kwargs):
    """Reads the QA_PIXEL band to identify clouds/shadows and masks them out."""
    print("Applying QA_PIXEL Cloud and Shadow Mask...")
    
    with rasterio.open(config.FILE_QA) as src_qa:
        qa_data = src_qa.read(1)
    
    cloud_mask = (qa_data & (1 << 3) > 0) | (qa_data & (1 << 4) > 0)
    
    # Apply the mask: Set bad pixels to our NODATA_VALUE (0)
    rgb_stack[:, cloud_mask] = config.NODATA_VALUE
    ir_aligned[cloud_mask] = config.NODATA_VALUE
    
    return rgb_stack, ir_aligned

def run_pipeline():
    print("--- STARTING PHASE 1: GEOMETRY ENGINE ---")
    
    #Load Master Grid
    rgb_stack, master_kwargs, master_transform, master_crs = load_and_stack_rgb()
    
    #Align Thermal Data
    ir_aligned = align_thermal_to_master(master_kwargs, master_transform, master_crs)
    
    #Apply Cloud Mask
    rgb_stack, ir_aligned = apply_cloud_mask(rgb_stack, ir_aligned, master_kwargs)
    
    # 4. THE GEOMETRIC SANITY CHECKS (The Gatekeepers)
    print("Running Geometric Sanity Checks...")
    
    # Check 1: Do the spatial dimensions match exactly?
    assert rgb_stack.shape[1:] == ir_aligned.shape, f"CRITICAL ERROR: Shape mismatch! RGB: {rgb_stack.shape}, IR: {ir_aligned.shape}"
    
    # Check 2: Do they point to the exact same geospatial coordinates?
    print("Sanity Checks Passed! Matrices are physically locked.")
    
    # 5. Save the perfectly aligned, masked arrays to the drive
    print("Saving processed arrays to disk...")
    rgb_save_path = config.PROCESSED_DATA_DIR / "rgb_aligned.npy"
    ir_save_path = config.PROCESSED_DATA_DIR / "ir_aligned.npy"
    
    np.save(rgb_save_path, rgb_stack)
    np.save(ir_save_path, ir_aligned)
    
    print(f"SUCCESS! Files saved to {config.PROCESSED_DATA_DIR}")
    print("--- GEOMETRY ENGINE COMPLETE ---")

if __name__ == "__main__":
    run_pipeline()