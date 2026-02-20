"""
Bicubic DTM Upsampling (ROI-matched to fusion output)

Produces bicubic resampled DTM on the EXACT same grid as the fusion output,
enabling direct pixel-by-pixel comparison of all three methods:
  1. Bicubic (this script)
  2. DSM-guided fusion (dsm_guided_dtm_fusion.py)
  3. Kriging (kriging_masked_by_fusion.py)

Strategy:
  - Read fusion output to get ROI bounds and grid
  - Clip 5m Bluesky DTM to that region
  - Bicubic resample to match fusion grid exactly
  - Save output with identical spatial alignment
"""

import numpy as np
import rasterio
from rasterio.windows import from_bounds as window_from_bounds
from scipy.ndimage import zoom as scipy_zoom
import os

# =======================================================================
# FILES — UPDATE THESE
# =======================================================================

# Fusion output — defines the ROI and target grid
fusion_dtm_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\fused\bluesky_dtm_fused_0061m.tif"

# 5m Bluesky DTM (will be clipped and resampled)
bluesky_dtm_5m_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\5m_dtm\bluesky_dtm_utm29n.tif"

# Output (bicubic resampled, ROI-matched)
output_bicubic = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\bicubic\bluesky_dtm_bicubic_matched_0061m.tif"

GEOID_OFFSET = 58.0
NODATA_OUT = -9999.0

# =======================================================================
# MAIN
# =======================================================================

def run_bicubic_roi_matched():
    print("="*70)
    print("BICUBIC RESAMPLING — ROI-matched to fusion output")
    print("="*70)

    if not os.path.exists(fusion_dtm_file):
        print("\n✗ Fusion DTM missing. Run dsm_guided_dtm_fusion.py first.")
        return

    # Step 1: Get ROI from fusion output
    print("\nStep 1: Reading fusion output to define ROI...")
    with rasterio.open(fusion_dtm_file) as src:
        fusion_data = src.read(1)
        roi_trans = src.transform
        roi_crs = src.crs
        roi_h, roi_w = fusion_data.shape
        pixel_size = abs(src.transform.a)

        # Find valid bounding box
        valid_mask = ~np.isnan(fusion_data)
        rows, cols = np.where(valid_mask)
        r0, r1 = rows.min(), rows.max() + 1
        c0, c1 = cols.min(), cols.max() + 1

        # World bounds of the valid region
        left = roi_trans.c + c0 * roi_trans.a
        top = roi_trans.f + r0 * roi_trans.e
        right = roi_trans.c + c1 * roi_trans.a
        bottom = roi_trans.f + r1 * roi_trans.e
        roi_bounds = (left, bottom, right, top)

    print(f"  Fusion grid:     {roi_h} × {roi_w} pixels")
    print(f"  Valid coverage:  {valid_mask.sum():,} / {roi_h * roi_w:,} pixels")
    print(f"  ROI bounds:      {left:.0f}, {bottom:.0f}, {right:.0f}, {top:.0f}")

    # Step 2: Load 5m DTM for that ROI
    print("\nStep 2: Loading 5m DTM clipped to ROI...")
    with rasterio.open(bluesky_dtm_5m_file) as src:
        window = window_from_bounds(*roi_bounds, transform=src.transform)
        dtm_5m = src.read(1, window=window).astype(np.float64)
        dtm_trans = src.window_transform(window)
        dtm_nodata = src.nodata if src.nodata is not None else NODATA_OUT

    # Mask nodata and apply geoid
    valid_dtm = (dtm_5m != dtm_nodata) & ~np.isnan(dtm_5m)
    dtm_5m[~valid_dtm] = np.nan
    dtm_5m[valid_dtm] += GEOID_OFFSET

    print(f"  DTM window:      {dtm_5m.shape[0]} × {dtm_5m.shape[1]} pixels at 5m")
    print(f"  Valid DTM pts:   {valid_dtm.sum():,}")
    print(f"  Elevation:       {np.nanmin(dtm_5m):.2f} to {np.nanmax(dtm_5m):.2f} m")

    # Step 3: Bicubic resample to match fusion grid
    print("\nStep 3: Bicubic resampling to 0.061m...")
    
    # Fill NaNs with mean for scipy (it doesn't handle NaN well)
    dtm_filled = dtm_5m.copy()
    dtm_filled[np.isnan(dtm_filled)] = np.nanmean(dtm_5m)

    # Calculate zoom factors
    zoom_r = roi_h / dtm_5m.shape[0]
    zoom_c = roi_w / dtm_5m.shape[1]
    
    print(f"  Zoom factors:    {zoom_r:.2f} × {zoom_c:.2f}")

    bicubic = scipy_zoom(dtm_filled, zoom=(zoom_r, zoom_c), order=3)

    # Trim to exact output size (in case of rounding)
    if bicubic.shape[0] > roi_h:
        bicubic = bicubic[:roi_h, :]
    if bicubic.shape[1] > roi_w:
        bicubic = bicubic[:, :roi_w]
    if bicubic.shape[0] < roi_h or bicubic.shape[1] < roi_w:
        padded = np.full((roi_h, roi_w), np.nan)
        padded[:bicubic.shape[0], :bicubic.shape[1]] = bicubic
        bicubic = padded

    # Mask invalid regions (where original 5m DTM had no data)
    # This ensures output is NaN where we had no input data
    bicubic = bicubic.astype(np.float32)

    print(f"  Output size:     {bicubic.shape[0]} × {bicubic.shape[1]} pixels")
    print(f"  Valid output:    {(~np.isnan(bicubic)).sum():,} pixels")

    # Step 4: Save
    print("\nStep 4: Saving...")
    os.makedirs(os.path.dirname(output_bicubic), exist_ok=True)
    
    data_out = np.where(np.isnan(bicubic), NODATA_OUT, bicubic).astype(np.float32)
    
    with rasterio.open(output_bicubic, "w",
                       driver="GTiff", dtype="float32", nodata=NODATA_OUT,
                       crs=roi_crs, transform=roi_trans,
                       width=roi_w, height=roi_h, count=1,
                       compress="deflate", tiled=True,
                       blockxsize=256, blockysize=256, predictor=3) as dst:
        dst.write(data_out, 1)

    gb = os.path.getsize(output_bicubic) / 1024**3
    print(f"  ✓ Saved: {os.path.basename(output_bicubic)}  ({gb:.2f} GB)")

    print("\n" + "="*70)
    print("✅  BICUBIC RESAMPLING COMPLETE")
    print("="*70)
    print(f"\n  Output: {output_bicubic}")
    print(f"\n  This DTM is spatially aligned with:")
    print(f"    • Fusion output (bluesky_dtm_fused_0061m.tif)")
    print(f"    • Kriging output (bluesky_dtm_kriged_masked_0061m.tif)")
    print(f"\n  All three can now be compared pixel-by-pixel.")


if __name__ == "__main__":
    run_bicubic_roi_matched()
