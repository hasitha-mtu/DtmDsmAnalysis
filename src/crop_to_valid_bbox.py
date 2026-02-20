"""
Crop fusion and kriging outputs to just the valid data bounding box

This creates much smaller, easier-to-view files by removing all the NaN padding.
"""

import rasterio
from rasterio.windows import Window
import numpy as np
import os

# Input files
fusion_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\fused\bluesky_dtm_fused_0061m.tif"
kriging_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\kriged\bluesky_dtm_kriged_masked_0061m.tif"
kriging_var_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\kriged\bluesky_dtm_kriging_variance_masked.tif"

# Output files (cropped versions)
fusion_cropped = fusion_file.replace('.tif', '_cropped.tif')
kriging_cropped = kriging_file.replace('.tif', '_cropped.tif')
kriging_var_cropped = kriging_var_file.replace('.tif', '_cropped.tif')

NODATA = -9999.0

def find_valid_bbox(filepath):
    """Find bounding box of valid (non-NaN) data"""
    print(f"\nAnalyzing: {os.path.basename(filepath)}")
    
    with rasterio.open(filepath) as src:
        # Read full array
        data = src.read(1)
        valid_mask = ~np.isnan(data) & (data != src.nodata)
        
        if not valid_mask.any():
            print("  ✗ No valid data found!")
            return None, None, None
        
        rows, cols = np.where(valid_mask)
        r0, r1 = int(rows.min()), int(rows.max() + 1)
        c0, c1 = int(cols.min()), int(cols.max() + 1)
        
        print(f"  Original size: {src.height} × {src.width}")
        print(f"  Valid bbox:    rows {r0}–{r1}, cols {c0}–{c1}")
        print(f"  Cropped size:  {r1-r0} × {c1-c0}")
        print(f"  Reduction:     {(1 - (r1-r0)*(c1-c0)/(src.height*src.width))*100:.1f}% smaller")
        
        return Window(c0, r0, c1-c0, r1-r0), src.transform, src.crs


def crop_file(input_file, output_file, window, transform, crs):
    """Crop a GeoTIFF to the specified window"""
    
    if not os.path.exists(input_file):
        print(f"  ✗ Input file not found: {input_file}")
        return
    
    with rasterio.open(input_file) as src:
        # Read data within window
        data = src.read(1, window=window)
        
        # Update transform for the window
        window_transform = src.window_transform(window)
        
        # Write cropped file
        with rasterio.open(output_file, 'w',
                          driver='GTiff',
                          height=window.height,
                          width=window.width,
                          count=1,
                          dtype=data.dtype,
                          crs=crs,
                          transform=window_transform,
                          nodata=NODATA,
                          compress='deflate',
                          tiled=True,
                          blockxsize=256,
                          blockysize=256) as dst:
            
            # Convert NaN to nodata value
            data_out = np.where(np.isnan(data), NODATA, data)
            dst.write(data_out, 1)
        
        file_size_mb = os.path.getsize(output_file) / 1024**2
        print(f"  ✓ Saved: {os.path.basename(output_file)} ({file_size_mb:.1f} MB)")


def main():
    print("="*70)
    print("CROP OUTPUTS TO VALID DATA BOUNDING BOX")
    print("="*70)
    
    # Find bbox from fusion (use this for all files to ensure alignment)
    window, transform, crs = find_valid_bbox(fusion_file)
    
    if window is None:
        print("\n✗ Cannot proceed - fusion file has no valid data")
        return
    
    print("\n" + "="*70)
    print("CROPPING FILES")
    print("="*70)
    
    # Crop fusion
    print("\n1. Fusion DTM")
    crop_file(fusion_file, fusion_cropped, window, transform, crs)
    
    # Crop kriging
    if os.path.exists(kriging_file):
        print("\n2. Kriging DTM")
        crop_file(kriging_file, kriging_cropped, window, transform, crs)
    else:
        print(f"\n2. Kriging DTM: ✗ Not found")
    
    # Crop variance
    if os.path.exists(kriging_var_file):
        print("\n3. Kriging Variance")
        crop_file(kriging_var_file, kriging_var_cropped, window, transform, crs)
    else:
        print(f"\n3. Kriging Variance: ✗ Not found")
    
    print("\n" + "="*70)
    print("✅ CROPPING COMPLETE")
    print("="*70)
    print(f"\nCropped files are much smaller and easier to visualize:")
    print(f"  • {os.path.basename(fusion_cropped)}")
    if os.path.exists(kriging_file):
        print(f"  • {os.path.basename(kriging_cropped)}")
    if os.path.exists(kriging_var_file):
        print(f"  • {os.path.basename(kriging_var_cropped)}")
    
    print(f"\nThese files contain the same data but with NaN padding removed.")
    print(f"Open the '_cropped.tif' versions in QGIS for better visualization.")


if __name__ == "__main__":
    main()
