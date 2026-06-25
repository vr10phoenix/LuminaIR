from pathlib import Path
import os

BASE_DIR = Path(__file__).parent.parent.parent  # LuminaIR/
RAW_DATA_DIR = BASE_DIR / "Data"
PROCESSED_DATA_DIR = BASE_DIR / "Pipeline" / "Preprocessing" / "processed_data"
PATCHES_DIR = BASE_DIR / "Pipeline" / "Preprocessing" / "training_patches"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
PATCHES_DIR.mkdir(parents=True, exist_ok=True)

# ---------- FIND FILES BY PATTERN (ignores date variations) ----------
try:
    FILE_B4 = list(RAW_DATA_DIR.glob("*SR_B4.TIF"))[0]
    FILE_B3 = list(RAW_DATA_DIR.glob("*SR_B3.TIF"))[0]
    FILE_B2 = list(RAW_DATA_DIR.glob("*SR_B2.TIF"))[0]
    FILE_B10 = list(RAW_DATA_DIR.glob("*ST_B10.TIF"))[0]
    FILE_QA = list(RAW_DATA_DIR.glob("*QA_PIXEL.TIF"))[0]
except IndexError:
    raise FileNotFoundError("One or more required .TIF files not found in Data folder. Check the file names.")

# ---------- PREPROCESSING CONSTANTS ----------
PATCH_SIZE = 512
MAX_CLOUD_COVER = 0.10
NODATA_VALUE = 0