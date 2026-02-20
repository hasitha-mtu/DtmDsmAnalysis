"""
Quick check of kriging output to see why it shows 0 valid pixels
"""

import rasterio
import numpy as np

kriging_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\kriged\fusion_based_kriged_0061m.tif"

print("="*70)
print("KRIGING OUTPUT DIAGNOSTIC")
print("="*70)

with rasterio.open(kriging_file) as src:
    print(f"\nFile: {kriging_file}")
    print(f"Size: {src.height} × {src.width}")
    print(f"Dtype: {src.dtypes[0]}")
    print(f"Nodata: {src.nodata}")
    
    # Read a sample
    data = src.read(1)
    
    print(f"\nData statistics:")
    print(f"  Min value:    {np.nanmin(data)}")
    print(f"  Max value:    {np.nanmax(data)}")
    print(f"  Mean value:   {np.nanmean(data)}")
    print(f"  Unique vals:  {len(np.unique(data)):,}")
    
    # Count different types of pixels
    is_nan = np.isnan(data)
    is_nodata = (data == src.nodata) if src.nodata is not None else np.zeros_like(data, dtype=bool)
    is_negative = (data < 0) & ~is_nan & ~is_nodata
    is_positive = (data > 0) & ~is_nan & ~is_nodata
    is_zero = (data == 0) & ~is_nan & ~is_nodata
    
    total = data.size
    
    print(f"\nPixel breakdown:")
    print(f"  NaN:          {is_nan.sum():,} ({is_nan.sum()/total*100:.1f}%)")
    print(f"  NODATA:       {is_nodata.sum():,} ({is_nodata.sum()/total*100:.1f}%)")
    print(f"  Zero:         {is_zero.sum():,} ({is_zero.sum()/total*100:.1f}%)")
    print(f"  Negative:     {is_negative.sum():,} ({is_negative.sum()/total*100:.1f}%)")
    print(f"  Positive:     {is_positive.sum():,} ({is_positive.sum()/total*100:.1f}%)")
    print(f"  Total:        {total:,}")
    
    # What would be considered "valid" by different checks?
    print(f"\nValidity checks:")
    
    check1 = ~np.isnan(data)
    print(f"  Not NaN:                {check1.sum():,} ({check1.sum()/total*100:.1f}%)")
    
    if src.nodata is not None:
        check2 = data != src.nodata
        print(f"  Not NODATA ({src.nodata}):      {check2.sum():,} ({check2.sum()/total*100:.1f}%)")
        
        check3 = check1 & check2
        print(f"  Not NaN AND Not NODATA: {check3.sum():,} ({check3.sum()/total*100:.1f}%)")
    
    # In elevation range?
    reasonable = (data > 0) & (data < 500)
    print(f"  In range 0-500m:        {reasonable.sum():,} ({reasonable.sum()/total*100:.1f}%)")
    
    # Sample some actual values
    print(f"\nSample values (first 100 non-NaN):")
    non_nan_vals = data[~np.isnan(data)].ravel()
    if len(non_nan_vals) > 0:
        sample = non_nan_vals[:100]
        unique_sample = np.unique(sample)
        print(f"  Count: {len(sample)}")
        print(f"  Unique: {len(unique_sample)}")
        if len(unique_sample) <= 20:
            print(f"  Values: {unique_sample}")
        else:
            print(f"  Range: {unique_sample.min():.3f} to {unique_sample.max():.3f}")
            print(f"  First 10: {unique_sample[:10]}")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

if is_nan.sum() == total:
    print("\n✗ PROBLEM: All pixels are NaN")
    print("   The kriging script produced no valid output.")
    
elif is_nodata.sum() == total:
    print("\n✗ PROBLEM: All pixels are NODATA")
    print("   The kriging script wrote everything as nodata.")
    
elif is_positive.sum() > 0:
    print(f"\n✓ File has {is_positive.sum():,} valid elevation values")
    print("   The issue is in how the comparison script reads the file.")
    print("\n   Likely cause: NODATA value mismatch")
    print(f"   File nodata: {src.nodata}")
    print("   Comparison script might be checking different nodata value.")
    
else:
    print("\n⚠️  File has data but no positive values")
    print("   This is unusual for elevation data.")
