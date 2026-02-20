"""
Quick diagnostic to check kriged DTM files and their data coverage
"""

import rasterio
import numpy as np
import os
import glob

kriged_dir = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\kriged"

print("="*70)
print("KRIGED DTM FILES DIAGNOSTIC")
print("="*70)

# Find all kriged DTM files
pattern = os.path.join(kriged_dir, "*kriged*.tif")
files = glob.glob(pattern)

if not files:
    print(f"\n✗ No kriged DTM files found in:")
    print(f"  {kriged_dir}")
else:
    print(f"\nFound {len(files)} kriged DTM file(s):\n")
    
    for filepath in files:
        filename = os.path.basename(filepath)
        filesize_mb = os.path.getsize(filepath) / 1024**2
        
        print(f"File: {filename}")
        print(f"  Size: {filesize_mb:.1f} MB")
        
        with rasterio.open(filepath) as src:
            print(f"  Dimensions: {src.height} × {src.width}")
            print(f"  Resolution: {abs(src.transform.a):.4f}m")
            print(f"  Nodata: {src.nodata}")
            
            # Sample coverage check
            sample = src.read(1)[::100, ::100]
            
            if src.nodata is not None:
                valid = (sample != src.nodata) & ~np.isnan(sample)
            else:
                valid = ~np.isnan(sample)
            
            coverage_pct = valid.sum() / sample.size * 100
            
            print(f"  Data coverage (sampled): {coverage_pct:.1f}%")
            
            if coverage_pct > 0:
                valid_vals = sample[valid]
                print(f"  Elevation range: {valid_vals.min():.2f} to {valid_vals.max():.2f} m")
            
            # Status
            if coverage_pct < 1:
                print(f"  ❌ UNUSABLE - No valid data")
            elif coverage_pct < 10:
                print(f"  ⚠️  SPARSE - Very limited coverage")
            elif coverage_pct < 80:
                print(f"  ⚠️  PARTIAL - Moderate coverage")
            else:
                print(f"  ✅ GOOD - Ready for use")
        
        print()

print("="*70)
print("RECOMMENDATION")
print("="*70)

print("\nFor batch water depth extraction, use:")
print("  fusion_based_kriged_0061m.tif")
print("\nThis file should have ~90% coverage matching the fusion DTM extent.")
print("\nUpdate batch_bluesky_depth.py line 34 to:")
print('  bluesky_resampled_file = r"...\\kriged\\fusion_based_kriged_0061m.tif"')
