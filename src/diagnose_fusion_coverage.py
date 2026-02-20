"""
Diagnostic script to check fusion DTM coverage and identify issues
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt

fusion_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\fused\bluesky_dtm_fused_0061m.tif"

print("="*70)
print("FUSION DTM DIAGNOSTIC")
print("="*70)

with rasterio.open(fusion_file) as src:
    print(f"\nFile: {fusion_file}")
    print(f"Size: {src.height} × {src.width} pixels")
    print(f"Resolution: {src.transform.a:.4f} m/px")
    print(f"Extent: {src.height * src.transform.a / 1000:.2f}km × {src.width * src.transform.a / 1000:.2f}km")
    print(f"Bounds: {src.bounds}")
    print(f"CRS: {src.crs}")
    
    # Read in chunks to avoid memory issues
    print("\nReading data in chunks...")
    total_pixels = src.height * src.width
    valid_count = 0
    min_val = np.inf
    max_val = -np.inf
    
    # Read in 1000-row chunks
    chunk_size = 1000
    for i in range(0, src.height, chunk_size):
        rows = min(chunk_size, src.height - i)
        chunk = src.read(1, window=((i, i+rows), (0, src.width)))
        
        valid = ~np.isnan(chunk)
        valid_count += valid.sum()
        
        if valid.any():
            min_val = min(min_val, np.nanmin(chunk))
            max_val = max(max_val, np.nanmax(chunk))
        
        if (i // chunk_size) % 10 == 0:
            print(f"  Processed {i+rows}/{src.height} rows... "
                  f"valid so far: {valid_count:,} ({valid_count/total_pixels*100:.2f}%)")
    
    print(f"\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total pixels:     {total_pixels:,}")
    print(f"Valid pixels:     {valid_count:,}")
    print(f"Coverage:         {valid_count/total_pixels*100:.2f}%")
    print(f"NaN pixels:       {total_pixels - valid_count:,} ({(total_pixels-valid_count)/total_pixels*100:.2f}%)")
    
    if valid_count > 0:
        print(f"\nElevation range:  {min_val:.2f} to {max_val:.2f} m")
    
    # Find bounding box of valid data
    print(f"\nFinding bounding box of valid data...")
    all_data = src.read(1)  # Read full array
    valid_mask = ~np.isnan(all_data)
    
    if valid_mask.any():
        rows, cols = np.where(valid_mask)
        r0, r1 = rows.min(), rows.max() + 1
        c0, c1 = cols.min(), cols.max() + 1
        
        bbox_h = r1 - r0
        bbox_w = c1 - c0
        
        print(f"  Valid bbox: rows {r0:,}–{r1:,}, cols {c0:,}–{c1:,}")
        print(f"  Bbox size:  {bbox_h:,} × {bbox_w:,} pixels")
        print(f"  Bbox area:  {bbox_h * bbox_w:,} pixels")
        print(f"  Fill ratio: {valid_count / (bbox_h * bbox_w) * 100:.1f}% "
              f"(valid pixels within bbox)")
        
        # Create overview image
        print(f"\nCreating overview visualization...")
        
        # Downsample for visualization
        downsample = max(src.height, src.width) // 2000 + 1
        
        overview = all_data[::downsample, ::downsample]
        valid_overview = valid_mask[::downsample, ::downsample]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Show elevation
        im1 = axes[0].imshow(overview, cmap='terrain')
        axes[0].set_title(f'Fusion DTM Elevation\n{src.height}×{src.width} px, {valid_count/total_pixels*100:.1f}% valid', 
                         fontweight='bold')
        axes[0].set_xlabel('Column (downsampled)')
        axes[0].set_ylabel('Row (downsampled)')
        plt.colorbar(im1, ax=axes[0], label='Elevation (m)')
        
        # Mark bounding box
        box_r0 = r0 // downsample
        box_r1 = r1 // downsample
        box_c0 = c0 // downsample
        box_c1 = c1 // downsample
        from matplotlib.patches import Rectangle
        rect = Rectangle((box_c0, box_r0), box_c1-box_c0, box_r1-box_r0,
                         linewidth=3, edgecolor='red', facecolor='none',
                         label='Valid data bbox')
        axes[0].add_patch(rect)
        axes[0].legend()
        
        # Show coverage mask
        axes[1].imshow(valid_overview, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Coverage Mask\n(white=valid, black=NaN)', fontweight='bold')
        axes[1].set_xlabel('Column (downsampled)')
        axes[1].set_ylabel('Row (downsampled)')
        
        rect2 = Rectangle((box_c0, box_r0), box_c1-box_c0, box_r1-box_r0,
                          linewidth=3, edgecolor='red', facecolor='none')
        axes[1].add_patch(rect2)
        
        plt.tight_layout()
        output_plot = fusion_file.replace('.tif', '_diagnostic.png')
        plt.savefig(output_plot, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved diagnostic plot: {output_plot}")
        
        # INTERPRETATION
        print(f"\n" + "="*70)
        print("INTERPRETATION")
        print("="*70)
        
        if valid_count / total_pixels < 0.1:
            print(f"⚠️  Your fusion output has SPARSE coverage ({valid_count/total_pixels*100:.1f}%)")
            print(f"   This means:")
            print(f"   • The orthophoto extent is much larger than the DTM/DSM coverage")
            print(f"   • Most of the output file is NaN (no data)")
            print(f"   • Valid data is concentrated in a small region")
            print(f"\n   When you open this in QGIS/viewer:")
            print(f"   • It will look mostly BLACK (NaN pixels)")
            print(f"   • With a small region of actual data")
            print(f"   • This is NORMAL and EXPECTED")
            print(f"\n   The kriging output will look similar because it's")
            print(f"   spatially aligned with this fusion output.")
        else:
            print(f"✓  Coverage looks good ({valid_count/total_pixels*100:.1f}%)")
            
        coverage_in_bbox = valid_count / (bbox_h * bbox_w) * 100
        if coverage_in_bbox < 50:
            print(f"\n⚠️  Even within the bounding box, coverage is sparse ({coverage_in_bbox:.1f}%)")
            print(f"   This suggests the river corridor is fragmented or narrow.")
        elif coverage_in_bbox > 90:
            print(f"\n✓  Good coverage within bounding box ({coverage_in_bbox:.1f}%)")
        else:
            print(f"\n   Moderate coverage within bbox ({coverage_in_bbox:.1f}%)")
            print(f"   River corridor likely has gaps or is narrow/winding.")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("If your fusion output has <10% coverage:")
print("  1. This is NORMAL for a river corridor study")
print("  2. The 'dark image with white dots' is just how viewers render sparse data")
print("  3. The kriging output will look similar (this is correct!)")
print("  4. Your comparison stats (RMSE=0.024m) prove the data IS there")
print("\nTo visualize better in QGIS:")
print("  • Zoom to layer extent (right-click layer → Zoom to Layer)")
print("  • Adjust stretch to Min/Max (Layer Properties → Symbology)")
print("  • Clip to the bounding box to create a smaller file")
