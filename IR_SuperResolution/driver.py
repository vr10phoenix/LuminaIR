import os
import argparse
import subprocess
from utils.logging_utils import setup_logging
from utils.file_utils import find_file

def run_script(script_name, logger, *args):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(base_dir, 'scripts')
    script_path = os.path.join(scripts_dir, script_name)
    command = ['python', script_path] + list(args)
    logger.info(f"Executing: {' '.join(command)}")
    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = base_dir + os.pathsep + env.get('PYTHONPATH', '')
        result = subprocess.run(command, capture_output=True, text=True, check=True, env=env)
        if result.stdout:
            logger.info(f"STDOUT [{script_name}]:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failure in execution {script_name}: {e}")
        logger.error(f"STDERR Trace:\n{e.stderr}")
        raise e

def main():
    parser = argparse.ArgumentParser(description='Master Dataset Orchestrator Configuration Engine')
    parser.add_argument('--stride', type=str, default='64', help='Sliding window step size increment')
    parser.add_argument('--save_visuals', action='store_true', help='Toggle to save visual debugging png copies')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_root = os.path.join(base_dir, 'input')
    output_dir = os.path.join(base_dir, 'output')
    
    output_downscale_dir = os.path.join(output_dir, 'downscaled_data')
    output_rgb_dir = os.path.join(output_dir, 'rgb_images')
    output_patches_dir = os.path.join(output_dir, 'patches')

    for d in [output_downscale_dir, output_rgb_dir, output_patches_dir]:
        os.makedirs(d, exist_ok=True)

    logger = setup_logging('master_orchestrator', output_dir)

    if not os.path.isdir(input_root):
        logger.error(f"Input directory tree {input_root} missing.")
        exit(1)

    product_folders = [e for e in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, e))]

    for product_id in product_folders:
        input_dir = os.path.join(input_root, product_id)
        logger.info(f"Starting pipeline execution pass for: {product_id}")

        band2_path = find_file(input_dir, '_B2')
        band3_path = find_file(input_dir, '_B3')
        band4_path = find_file(input_dir, '_B4')
        band10_path = find_file(input_dir, '_B10')

        if not all([band2_path, band3_path, band4_path, band10_path]):
            logger.warning(f"Required spectral footprints missing. Skipping product ID: {product_id}")
            continue

        try:
            # 1. Merge RGB (30m)
            rgb_output_path = os.path.join(output_rgb_dir, f'{product_id}_rgb_30m.tif')
            run_script('merge_rgb.py', logger, band4_path, band3_path, band2_path, rgb_output_path)

            # 2. Downscale RGB to 100m
            downscaled_rgb_100m = os.path.join(output_downscale_dir, f'{product_id}_rgb_100m.tif')
            run_script('downscale.py', logger, rgb_output_path, downscaled_rgb_100m, '3.33')

            # 3. Downscale TIR to 100m
            downscaled_tir_100m = os.path.join(output_downscale_dir, f'{product_id}_tir_100m.tif')
            run_script('downscale.py', logger, band10_path, downscaled_tir_100m, '3.33')

            # 4. Downscale TIR to 200m
            downscaled_tir_200m = os.path.join(output_downscale_dir, f'{product_id}_tir_200m.tif')
            run_script('downscale.py', logger, band10_path, downscaled_tir_200m, '6.67')

            # 5. Overlapping High-Yield Matrix Patch Generation
            patch_args = ['--input_dir', output_downscale_dir, '--output_dir', output_patches_dir, '--stride', args.stride]
            if args.save_visuals:
                patch_args.append('--save_visuals')
                
            run_script('create_patches.py', logger, *patch_args)
            logger.info(f"Dataset matrix arrays ready for product footprint: {product_id}")

        except Exception as e:
            logger.error(f"Error compiling scene array target frames: {e}")

    logger.info("Dataset expansion complete. Arrays accessible within output/patches directory.")

if __name__ == '__main__':
    main()


# use case : python driver.py --stride 64
# python driver.py --stride 256 --save_visuals