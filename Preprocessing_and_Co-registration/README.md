GDAL & Rasterio: libraries for Satellite Imaging

Why chose it: Standard image libraries like OpenCV or Pillow cannot understand the complex geographic metadata embedded in GeoTIFFs. Rasterio translates GDAL's complex C-level operations into clean Python code.

Purpose: 
# reading .tif satellite formats
# handling coordinate reference systems (CRS)
# performing the spatial reprojection to perfectly align the grids. They read geospatial data directly into NumPy arrays.
