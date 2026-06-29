import tifffile
import numpy as np
import os
import glob
import argparse
import logging
import cv2
from utils.logging_utils import setup_logging
from utils.visualization import percentile_stretch
from utils.file_utils import find_file

def save_as_png(data, path):
    """Saves a numpy array as a normalized PNG for visual validation."""
    if data.ndim == 3:
        data = np.moveaxis(data, 0, -1)
    stretched = percentile_stretch(data)
    cv2.imwrite(path, stretched)

def create_patches(input_root, output_root, stride=128, patch_size=256, save_visuals=False):
    os.makedirs(output_root, exist_ok=True)
    logger = setup_logging('patch_slicer', output_root)
    
    if not os.path.exists(input_root):
        logger.error(f"Input directory {input_root} does not exist.")
        return

    all_files = glob.glob(os.path.join(input_root, '*'))
    products = set(os.path.basename(f).split('_')[0] for f in all_files if '_' in os.path.basename(f))

    logger.info(f"Found {len(products)} products to tile.")
    logger.info(f"Slicing Config: Size={patch_size}, Stride={stride}, Visuals Enabled={save_visuals}")

    for product_id in products:
        tir_200m_path = find_file(input_root, f'{product_id}*_tir_200m*')
        tir_100m_path = find_file(input_root, f'{product_id}*_tir_100m*')
        rgb_100m_path = find_file(input_root, f'{product_id}*_rgb_100m*')
        
        if not all([tir_200m_path, tir_100m_path, rgb_100m_path]):
            logger.warning(f"Skipping {product_id}: Resolution sets incomplete.")
            continue

        try:
            tir_200m = tifffile.imread(tir_200m_path)
            tir_100m = tifffile.imread(tir_100m_path)
            rgb_100m = tifffile.imread(rgb_100m_path)
        except Exception as e:
            logger.error(f"Error reading arrays for {product_id}: {e}")
            continue

        h200, w200 = tir_200m.shape[-2:]
        count = 0
        
        # Build coordinates dynamically via stride
        y_coords = list(range(0, h200 - patch_size + 1, stride))
        x_coords = list(range(0, w200 - patch_size + 1, stride))
        
        # Edge fallback validation pass
        if h200 >= patch_size and (h200 - patch_size) % stride != 0:
            y_coords.append(h200 - patch_size)
        if w200 >= patch_size and (w200 - patch_size) % stride != 0:
            x_coords.append(w200 - patch_size)

        for y in y_coords:
            for x in x_coords:
                patch_200m_tir = tir_200m[..., y:y+patch_size, x:x+patch_size]

                # Map coordinates out to matching 2x scale 100m layers
                y100, x100 = 2 * y, 2 * x
                t_size_100m = 2 * patch_size # 512
                
                patch_100m_tir_512 = tir_100m[..., y100:y100+t_size_100m, x100:x100+t_size_100m]
                patch_100m_rgb_512 = rgb_100m[..., y100:y100+t_size_100m, x100:x100+t_size_100m]

                if patch_100m_tir_512.shape[-2:] != (t_size_100m, t_size_100m) or \
                   patch_100m_rgb_512.shape[-2:] != (t_size_100m, t_size_100m):
                    continue

                sample_dir = os.path.join(output_root, product_id, f'sample_{count:04d}')
                os.makedirs(sample_dir, exist_ok=True)

                data_map = {
                    'tir_200m': patch_200m_tir,
                    'tir_100m_512': patch_100m_tir_512,
                    'rgb_100m_512': patch_100m_rgb_512
                }

                for name, data in data_map.items():
                    np.save(os.path.join(sample_dir, f'{name}.npy'), data)
                    if save_visuals:
                        save_as_png(data, os.path.join(sample_dir, f'{name}.png'))

                count += 1
        logger.info(f"Successfully constructed {count} patches for {product_id}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='High-Yield Patch Generation Engine.')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--save_visuals', action='store_true')
    args = parser.parse_args()
    
    create_patches(args.input_dir, args.output_dir, args.stride, args.patch_size, args.save_visuals)