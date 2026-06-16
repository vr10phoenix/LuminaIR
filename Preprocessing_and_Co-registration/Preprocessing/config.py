import os
from pathlib import Path


RAW_DATA_DIR = Path("../Data").resolve()

PROCESSED_DATA_DIR = Path("./processed_data").resolve()
PATCHES_DIR = Path("./training_patches").resolve()

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
PATCHES_DIR.mkdir(parents=True, exist_ok=True)

# 2. RAW FILE NAMES (From your USGS download)
PREFIX = "LC09_L2SP_144051_20260119_20260120_02_T1"

# Using pathlib to safely join directory paths and file names
FILE_B4  = RAW_DATA_DIR / f"{PREFIX}_SR_B4.TIF"     # Red Band
FILE_B3  = RAW_DATA_DIR / f"{PREFIX}_SR_B3.TIF"     # Green Band
FILE_B2  = RAW_DATA_DIR / f"{PREFIX}_SR_B2.TIF"     # Blue Band
FILE_B10 = RAW_DATA_DIR / f"{PREFIX}_ST_B10.TIF"    # Thermal IR Band
FILE_QA  = RAW_DATA_DIR / f"{PREFIX}_QA_PIXEL.TIF"  # Cloud Mask Band

# 3. PREPROCESSING CONSTANTS
PATCH_SIZE = 256          # The pixel grid size for the neural network
MAX_CLOUD_COVER = 0.10    # Maximum allowed cloud percentage (10%) per patch
NODATA_VALUE = 0          # Landsat's default "black edge" pixel value to ignore