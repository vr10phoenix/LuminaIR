import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import config

def load_and_stack_rgb():
    with rasterio.open(config.FILE_B4) as src_r:
        red = src_r.read(1)
        master_kwargs = src_r.profile
        master_transform = src_r.transform
        master_crs = src_r.crs
    with rasterio.open(config.FILE_B3) as src_g:
        green = src_g.read(1)
    with rasterio.open(config.FILE_B2) as src_b:
        blue = src_b.read(1)
    rgb_stack = np.stack([red, green, blue], axis=0)
    return rgb_stack, master_kwargs, master_transform, master_crs

def align_thermal_to_master(master_kwargs, master_transform, master_crs):
    with rasterio.open(config.FILE_B10) as src_ir:
        ir_raw = src_ir.read(1)
        ir_aligned = np.zeros((master_kwargs['height'], master_kwargs['width']), dtype=ir_raw.dtype)
        reproject(
            source=ir_raw,
            destination=ir_aligned,
            src_transform=src_ir.transform,
            src_crs=src_ir.crs,
            dst_transform=master_transform,
            dst_crs=master_crs,
            resampling=Resampling.bilinear
        )
    return ir_aligned

def apply_cloud_mask(rgb_stack, ir_aligned, master_kwargs):
    with rasterio.open(config.FILE_QA) as src_qa:
        qa_data = src_qa.read(1)
    cloud_mask = (qa_data & (1 << 3) > 0) | (qa_data & (1 << 4) > 0)
    rgb_stack[:, cloud_mask] = config.NODATA_VALUE
    ir_aligned[cloud_mask] = config.NODATA_VALUE
    return rgb_stack, ir_aligned

def run_pipeline():
    print("--- STARTING GEOMETRY ENGINE ---")
    rgb_stack, master_kwargs, master_transform, master_crs = load_and_stack_rgb()
    ir_aligned = align_thermal_to_master(master_kwargs, master_transform, master_crs)
    rgb_stack, ir_aligned = apply_cloud_mask(rgb_stack, ir_aligned, master_kwargs)

    assert rgb_stack.shape[1:] == ir_aligned.shape, f"Shape mismatch: RGB {rgb_stack.shape} vs IR {ir_aligned.shape}"
    print("Sanity checks passed – grids aligned.")

    np.save(config.PROCESSED_DATA_DIR / "rgb_aligned.npy", rgb_stack)
    np.save(config.PROCESSED_DATA_DIR / "ir_aligned.npy", ir_aligned)
    print(f"SUCCESS! Files saved to {config.PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    run_pipeline()