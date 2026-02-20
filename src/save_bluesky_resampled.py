"""
STEP 1 (Run Once): Resample full Bluesky DTM to WebODM resolution and save

Resamples 5m Bluesky DTM → 0.061m and saves as GeoTIFF
Applies geoid correction (+58m) at the same time

After this, batch processing just reads from the saved file - no per-tile resampling!

Time: ~5-10 minutes (once only)
Output size: ~same as WebODM orthophoto (~2-3 GB)
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import os

# =======================================================================
# YOUR FILES
# =======================================================================

# Input: Bluesky DTM in EPSG:32629 (from convert_bluesky_asc.py)
bluesky_dtm_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\5m_dtm\bluesky_dtm_utm29n.tif"

# Reference: WebODM orthophoto (defines target resolution + extent)
orthophoto_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_orthophoto\odm_orthophoto.tif"

# Output: Resampled DTM at WebODM resolution, geoid corrected
bluesky_resampled_output = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\5m_dtm\bluesky_dtm_resampled_0061m.tif"

# Geoid correction
GEOID_OFFSET = 58.0  # metres - Malin Head → WGS84 ellipsoid

# =======================================================================
# RESAMPLE AND SAVE
# =======================================================================

def save_bluesky_resampled():
    """
    Resample Bluesky DTM to match WebODM resolution and save as GeoTIFF
    
    Uses rasterio's reproject() which:
    - Handles large files efficiently (no full-file load into RAM)
    - Uses proper bicubic resampling
    - Applies geoid correction in one pass
    """

    print("="*70)
    print("SAVE BICUBIC-RESAMPLED BLUESKY DTM")
    print("="*70)

    # ------------------------------------------------------------------
    # Read WebODM orthophoto metadata (defines target grid)
    # ------------------------------------------------------------------

    with rasterio.open(orthophoto_file) as ortho:
        target_crs       = ortho.crs
        target_transform = ortho.transform
        target_width     = ortho.width
        target_height    = ortho.height
        target_bounds    = ortho.bounds
        pixel_size       = abs(ortho.transform.a)

    print(f"\nTarget grid (from WebODM orthophoto):")
    print(f"  CRS:        {target_crs}")
    print(f"  Resolution: {pixel_size:.4f}m")
    print(f"  Size:       {target_height} × {target_width} pixels")
    print(f"  Bounds:     {target_bounds}")

    # ------------------------------------------------------------------
    # Read Bluesky source info
    # ------------------------------------------------------------------

    with rasterio.open(bluesky_dtm_file) as src:
        src_crs        = src.crs
        src_resolution = abs(src.transform.a)
        src_bounds     = src.bounds
        src_nodata     = src.nodata if src.nodata is not None else -9999.0

    print(f"\nSource Bluesky DTM:")
    print(f"  CRS:        {src_crs}")
    print(f"  Resolution: {src_resolution:.1f}m")
    print(f"  Bounds:     {src_bounds}")
    print(f"  Nodata:     {src_nodata}")

    # Estimate output file size
    size_gb = (target_height * target_width * 4) / (1024**3)
    print(f"\nEstimated output size: {size_gb:.2f} GB (float32)")

    # ------------------------------------------------------------------
    # Resample and save
    # ------------------------------------------------------------------

    print(f"\nResampling (bicubic) + geoid correction (+{GEOID_OFFSET}m)...")
    print(f"This may take 5-10 minutes for large files...")

    output_nodata = -9999.0

    dst_meta = {
        'driver':    'GTiff',
        'dtype':     'float32',
        'nodata':    output_nodata,
        'crs':       target_crs,
        'transform': target_transform,
        'width':     target_width,
        'height':    target_height,
        'count':     1,
        'compress':  'deflate',   # ~50% size reduction
        'tiled':     True,
        'blockxsize': 256,
        'blockysize': 256,
        'predictor': 3,           # Floating-point predictor (better compression)
    }

    with rasterio.open(bluesky_resampled_output, 'w', **dst_meta) as dst:
        with rasterio.open(bluesky_dtm_file) as src:

            # Reproject/resample Bluesky → WebODM grid
            # rasterio handles this block by block internally
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.cubic,     # Bicubic
                src_nodata=src_nodata,
                dst_nodata=output_nodata,
            )

    # ------------------------------------------------------------------
    # Apply geoid correction to saved file
    # ------------------------------------------------------------------

    print(f"\nApplying +{GEOID_OFFSET}m geoid correction...")
    print(f"(Processing in blocks to avoid loading full file into RAM)")

    # Read, correct, overwrite in chunks
    with rasterio.open(bluesky_resampled_output, 'r+') as dst:
        
        block_size = 1024  # Process 1024 rows at a time
        
        for row_start in range(0, target_height, block_size):
            row_end = min(row_start + block_size, target_height)
            
            window = rasterio.windows.Window(
                col_off=0,
                row_off=row_start,
                width=target_width,
                height=row_end - row_start
            )
            
            block = dst.read(1, window=window)
            
            # Apply geoid offset only to valid pixels
            valid = block != output_nodata
            block[valid] += GEOID_OFFSET
            
            dst.write(block, 1, window=window)
            
            # Progress
            pct = row_end / target_height * 100
            if row_start % (block_size * 10) == 0 or row_end == target_height:
                print(f"  {pct:5.1f}% complete ({row_end:,} / {target_height:,} rows)")

    # ------------------------------------------------------------------
    # Verify output
    # ------------------------------------------------------------------

    print(f"\nVerifying output...")

    with rasterio.open(bluesky_resampled_output) as src:
        # Sample center of file
        center_row = target_height // 2
        center_col = target_width  // 2
        window = rasterio.windows.Window(center_col, center_row, 500, 500)
        sample = src.read(1, window=window)
        
        valid = sample[sample != output_nodata]

        print(f"\n  Output file:")
        print(f"    Path:       {bluesky_resampled_output}")
        print(f"    CRS:        {src.crs}")
        print(f"    Resolution: {abs(src.transform.a):.4f}m")
        print(f"    Size:       {src.height} × {src.width} pixels")
        
        file_size_gb = os.path.getsize(bluesky_resampled_output) / (1024**3)
        print(f"    File size:  {file_size_gb:.2f} GB (compressed)")
        
        if len(valid) > 0:
            print(f"\n  Sample elevation check (center 500×500 pixels):")
            print(f"    Min:  {np.min(valid):.2f}m")
            print(f"    Max:  {np.max(valid):.2f}m")
            print(f"    Mean: {np.mean(valid):.2f}m  ← Should be ~113m (WGS84 ellipsoid)")
            print(f"    (= Malin Head elevation + {GEOID_OFFSET}m geoid offset)")
        else:
            print(f"\n  ⚠️  Center sample has no valid data - check Bluesky coverage!")

    print(f"\n{'='*70}")
    print(f"✅ DONE - Bluesky DTM resampled and saved!")
    print(f"{'='*70}")
    print(f"\n  Output: {bluesky_resampled_output}")
    print(f"\n  This file is ready for batch_bluesky_depth.py")
    print(f"  Update batch script:")
    print(f"  bluesky_resampled_file = r\"{bluesky_resampled_output}\"")


if __name__ == "__main__":
    save_bluesky_resampled()
