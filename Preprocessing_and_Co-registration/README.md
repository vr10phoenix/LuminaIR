
# PIPELINE 
## config.py : 
purpose : Holds all your hardcoded variables. File paths to your USGS downloads, target patch sizes (e.g., $256 \times 256$), cloud cover thresholds, and global min/max scaling values.
why : If you want to change your patch size to $512 \times 512$ later, you change it in one place.

## Co-register.py (The Geometry Engine)
Purpose: Uses Rasterio to load the .tif files, warp the Thermal band to match the RGB master grid, apply the QA_PIXEL cloud mask to hide bad data, and run the two geometric assert sanity checks.

Output: A massive, perfectly aligned, cloud-masked .npy array saved to your drive.

## patch_slicer.py (The Butcher)
Purpose: Loads that massive .npy array and mathematically slices it into overlapping $256 \times 256$ grids. It includes the logic to automatically delete any patch that is 100% black (NoData) or too cloudy.
Output: Final, clean, normalized training folder.

## Main_pipeline.ipynb (The Colab Conductor)

Purpose: The clean, readable notebook where you simply import the modules above and run them step-by-step.

# TECH STACK
## GDAL & Rasterio: Libraries for the satellite imaging world.

Why chose it:
1. Understand the complex geographic metadata embedded in GeoTIFFs.
2. Rasterio translates GDAL's complex C-level operations into clean Python code.

Purpose:
 1. Reading .tif satellite formats.
 2. Handling coordinate reference systems (CRS).
 3. Performing the spatial reprojection to perfectly align the grids -> geospatial data directly into NumPy arrays.