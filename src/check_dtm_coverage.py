"""
Check spatial coverage overlap between fusion DTM and Bluesky 5m DTM

This will reveal if the 5m DTM actually covers the fusion extent.
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fusion_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\fused\bluesky_dtm_fused_0061m.tif"
dtm_5m_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\5m_dtm\bluesky_dtm_utm29n.tif"

print("="*70)
print("COVERAGE OVERLAP DIAGNOSTIC")
print("="*70)

# Read fusion bounds and find valid data extent
print("\n1. Fusion DTM:")
with rasterio.open(fusion_file) as src:
    fusion_bounds = src.bounds
    fusion_crs = src.crs
    fusion_shape = (src.height, src.width)
    
    # Sample to find valid data (downsample to avoid memory issues)
    ds_factor = max(src.height, src.width) // 1000 + 1
    sample = src.read(1)[::ds_factor, ::ds_factor]
    valid = (sample != -9999) & ~np.isnan(sample)
    
    if valid.any():
        # Map back to full resolution
        rows, cols = np.where(valid)
        r0, r1 = rows.min() * ds_factor, (rows.max() + 1) * ds_factor
        c0, c1 = cols.min() * ds_factor, (cols.max() + 1) * ds_factor
        
        fusion_valid_bounds = (
            src.bounds.left + c0 * src.transform.a,
            src.bounds.top + r1 * src.transform.e,  # e is negative
            src.bounds.left + c1 * src.transform.a,
            src.bounds.top + r0 * src.transform.e
        )
    else:
        fusion_valid_bounds = fusion_bounds

print(f"   Full extent:  {fusion_bounds}")
print(f"   Shape:        {fusion_shape[0]} × {fusion_shape[1]} px")
print(f"   CRS:          {fusion_crs}")
print(f"   Valid data:   {fusion_valid_bounds}")

# Read 5m DTM bounds
print("\n2. Bluesky 5m DTM:")
with rasterio.open(dtm_5m_file) as src:
    dtm_bounds = src.bounds
    dtm_crs = src.crs
    dtm_shape = (src.height, src.width)
    
    # Count valid pixels
    dtm_data = src.read(1)
    dtm_nodata = src.nodata if src.nodata is not None else -9999
    valid = (dtm_data != dtm_nodata) & ~np.isnan(dtm_data)
    
print(f"   Extent:       {dtm_bounds}")
print(f"   Shape:        {dtm_shape[0]} × {dtm_shape[1]} px")
print(f"   CRS:          {dtm_crs}")
print(f"   Valid pixels: {valid.sum():,} / {valid.size:,} ({valid.sum()/valid.size*100:.1f}%)")

# Check CRS match
print("\n3. CRS Compatibility:")
if fusion_crs == dtm_crs:
    print(f"   ✓ CRS match: {fusion_crs}")
else:
    print(f"   ✗ CRS MISMATCH!")
    print(f"     Fusion: {fusion_crs}")
    print(f"     DTM:    {dtm_crs}")
    print(f"   This will cause spatial misalignment!")

# Calculate overlap
print("\n4. Spatial Overlap:")

overlap_left = max(fusion_valid_bounds[0], dtm_bounds.left)
overlap_bottom = max(fusion_valid_bounds[1], dtm_bounds.bottom)
overlap_right = min(fusion_valid_bounds[2], dtm_bounds.right)
overlap_top = min(fusion_valid_bounds[3], dtm_bounds.top)

if overlap_left >= overlap_right or overlap_bottom >= overlap_top:
    print(f"   ✗ NO OVERLAP BETWEEN DATASETS!")
    print(f"\n   Fusion valid extent: {fusion_valid_bounds}")
    print(f"   5m DTM extent:       {dtm_bounds}")
    print(f"\n   These datasets do not cover the same area.")
    print(f"   Kriging cannot work with this 5m DTM file.")
else:
    overlap_area = (overlap_right - overlap_left) * (overlap_top - overlap_bottom)
    fusion_area = (fusion_valid_bounds[2] - fusion_valid_bounds[0]) * \
                  (fusion_valid_bounds[3] - fusion_valid_bounds[1])
    
    overlap_pct = overlap_area / fusion_area * 100
    
    print(f"   Overlap extent:  ({overlap_left:.0f}, {overlap_bottom:.0f}, "
          f"{overlap_right:.0f}, {overlap_top:.0f})")
    print(f"   Overlap area:    {overlap_area / 1e6:.2f} km²")
    print(f"   Fusion area:     {fusion_area / 1e6:.2f} km²")
    print(f"   Coverage:        {overlap_pct:.1f}% of fusion extent")
    
    if overlap_pct < 10:
        print(f"\n   ✗ CRITICAL: Only {overlap_pct:.1f}% overlap!")
        print(f"      Kriging will produce very sparse output.")
    elif overlap_pct < 50:
        print(f"\n   ⚠️  WARNING: Only {overlap_pct:.1f}% overlap")
        print(f"      Kriging output will be incomplete.")
    elif overlap_pct < 95:
        print(f"\n   ⚠️  Partial overlap ({overlap_pct:.1f}%)")
        print(f"      Some areas will have kriged data, others won't.")
    else:
        print(f"\n   ✓ Good overlap ({overlap_pct:.1f}%)")

# Visualize
print("\n5. Creating overlap visualization...")

fig, ax = plt.subplots(figsize=(12, 10))

# Convert to km for plotting
fusion_rect = Rectangle(
    (fusion_valid_bounds[0]/1000, fusion_valid_bounds[1]/1000),
    (fusion_valid_bounds[2] - fusion_valid_bounds[0])/1000,
    (fusion_valid_bounds[3] - fusion_valid_bounds[1])/1000,
    linewidth=3, edgecolor='blue', facecolor='blue', alpha=0.3,
    label=f'Fusion DTM extent ({fusion_area/1e6:.2f} km²)'
)

dtm_rect = Rectangle(
    (dtm_bounds.left/1000, dtm_bounds.bottom/1000),
    (dtm_bounds.right - dtm_bounds.left)/1000,
    (dtm_bounds.top - dtm_bounds.bottom)/1000,
    linewidth=3, edgecolor='red', facecolor='red', alpha=0.3,
    label=f'Bluesky 5m DTM extent ({(dtm_bounds.right-dtm_bounds.left)*(dtm_bounds.top-dtm_bounds.bottom)/1e6:.2f} km²)'
)

ax.add_patch(fusion_rect)
ax.add_patch(dtm_rect)

if overlap_left < overlap_right and overlap_bottom < overlap_top:
    overlap_rect = Rectangle(
        (overlap_left/1000, overlap_bottom/1000),
        (overlap_right - overlap_left)/1000,
        (overlap_top - overlap_bottom)/1000,
        linewidth=2, edgecolor='green', facecolor='green', alpha=0.5,
        label=f'Overlap area ({overlap_area/1e6:.2f} km², {overlap_pct:.1f}%)'
    )
    ax.add_patch(overlap_rect)

ax.set_xlabel('Easting (km)', fontweight='bold', fontsize=12)
ax.set_ylabel('Northing (km)', fontweight='bold', fontsize=12)
ax.set_title('Spatial Coverage: Fusion DTM vs Bluesky 5m DTM', fontweight='bold', fontsize=14)
ax.legend(loc='best', fontsize=11)
ax.grid(alpha=0.3)
ax.set_aspect('equal')

# Auto-scale to show all extents
all_x = [fusion_valid_bounds[0], fusion_valid_bounds[2], dtm_bounds.left, dtm_bounds.right]
all_y = [fusion_valid_bounds[1], fusion_valid_bounds[3], dtm_bounds.bottom, dtm_bounds.top]
margin = 0.05 * max(max(all_x) - min(all_x), max(all_y) - min(all_y))
ax.set_xlim(min(all_x)/1000 - margin/1000, max(all_x)/1000 + margin/1000)
ax.set_ylim(min(all_y)/1000 - margin/1000, max(all_y)/1000 + margin/1000)

plt.tight_layout()
output_plot = fusion_file.replace('.tif', '_coverage_diagnostic.png')
plt.savefig(output_plot, dpi=150, bbox_inches='tight')
plt.close()

print(f"   ✓ Saved: {output_plot}")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

if overlap_left >= overlap_right or overlap_bottom >= overlap_top:
    print("\n✗ The 5m DTM and fusion output DO NOT OVERLAP.")
    print("   Kriging cannot use this 5m DTM file.")
    print("\n   Possible causes:")
    print("   1. Wrong 5m DTM file selected")
    print("   2. Different coordinate systems (CRS mismatch)")
    print("   3. Fusion used a different DTM source")
    print("\n   Solutions:")
    print("   1. Check that bluesky_dtm_5m_file path is correct")
    print("   2. Verify both files are in same CRS (UTM 29N)")
    print("   3. Consider using fusion DTM itself for comparison")
elif overlap_pct < 50:
    print(f"\n⚠️  The 5m DTM only covers {overlap_pct:.1f}% of fusion extent.")
    print("   This explains why kriging output is so small (58 MB vs 2.98 GB).")
    print("\n   What's happening:")
    print("   - Kriging can only predict where it has input points")
    print("   - The 5m DTM has limited spatial coverage")
    print("   - Output will be sparse compared to fusion")
    print("\n   This is NOT a bug - it's a data coverage limitation.")
    print("   The fusion method works better because it uses 1m DSM with wider coverage.")
else:
    print(f"\n✓ Good overlap ({overlap_pct:.1f}%).")
    print("   If kriging output is still small, check for other issues.")
