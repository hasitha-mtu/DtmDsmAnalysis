"""
Quick check of fusion DTM values to understand the "dark image" issue
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt

fusion_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\fused\bluesky_dtm_fused_0061m.tif"

print("="*70)
print("FUSION DTM VALUE ANALYSIS")
print("="*70)

with rasterio.open(fusion_file) as src:
    print(f"\nFile info:")
    print(f"  Size: {src.height} × {src.width} pixels")
    print(f"  Nodata value: {src.nodata}")
    
    # Sample the center 1000x1000 region
    ch, cw = src.height // 2, src.width // 2
    sample = src.read(1, window=((ch-500, ch+500), (cw-500, cw+500)))
    
    print(f"\nCenter 1000×1000 sample:")
    print(f"  Min: {np.nanmin(sample):.3f}")
    print(f"  Max: {np.nanmax(sample):.3f}")
    print(f"  Mean: {np.nanmean(sample):.3f}")
    print(f"  Median: {np.nanmedian(sample):.3f}")
    print(f"  NaN count: {np.isnan(sample).sum():,} / {sample.size:,} "
          f"({np.isnan(sample).sum()/sample.size*100:.1f}%)")
    
    # Check if it's all zeros or a constant
    non_nan = sample[~np.isnan(sample)]
    if len(non_nan) > 0:
        unique_vals = np.unique(non_nan)
        print(f"  Unique values: {len(unique_vals):,}")
        
        if len(unique_vals) == 1:
            print(f"  ⚠️  All values are identical: {unique_vals[0]:.3f}")
            print(f"      This would appear as a solid color (probably dark)")
        elif len(unique_vals) < 10:
            print(f"  ⚠️  Very few unique values: {unique_vals}")
        else:
            print(f"  ✓  Good variation in values")
    
    # Sample histogram
    print(f"\nValue distribution (center sample):")
    if len(non_nan) > 0:
        percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
        pct_vals = np.percentile(non_nan, percentiles)
        for p, v in zip(percentiles, pct_vals):
            print(f"  {p:3d}th percentile: {v:.3f}")
    
    # Check full extent (sample every 100th pixel to avoid memory issues)
    print(f"\nFull extent (sampled):")
    full_sample = src.read(1)[::100, ::100]
    print(f"  Min: {np.nanmin(full_sample):.3f}")
    print(f"  Max: {np.nanmax(full_sample):.3f}")
    print(f"  Mean: {np.nanmean(full_sample):.3f}")
    print(f"  NaN %: {np.isnan(full_sample).sum()/full_sample.size*100:.1f}%")
    
    # Visual check
    print(f"\nCreating diagnostic plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Center sample with default colormap
    axes[0, 0].imshow(sample, cmap='terrain')
    axes[0, 0].set_title('Center 1000×1000 (default scale)', fontweight='bold')
    
    # Plot 2: Center sample with stretched colormap
    vmin, vmax = np.nanpercentile(sample, [1, 99])
    im2 = axes[0, 1].imshow(sample, cmap='terrain', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'Center 1000×1000 (stretched)\n{vmin:.1f} to {vmax:.1f}m', 
                        fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 3: Histogram
    axes[1, 0].hist(non_nan.ravel(), bins=100, edgecolor='black')
    axes[1, 0].set_xlabel('Elevation (m)', fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontweight='bold')
    axes[1, 0].set_title('Value Distribution', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Oversampled to see patterns
    downsample = max(src.height, src.width) // 1000 + 1
    overview = src.read(1)[::downsample, ::downsample]
    vmin_o, vmax_o = np.nanpercentile(overview, [1, 99])
    im4 = axes[1, 1].imshow(overview, cmap='terrain', vmin=vmin_o, vmax=vmax_o)
    axes[1, 1].set_title(f'Full extent overview\n(downsampled {downsample}×)', fontweight='bold')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    output_plot = fusion_file.replace('.tif', '_value_diagnostic.png')
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_plot}")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

print("\nPossible reasons for 'dark image with white dots':")
print("  1. Display software using wrong stretch (use histogram stretch)")
print("  2. Most pixels have similar values (low contrast)")
print("  3. File has nodata value that appears black")
print("  4. Elevation range is compressed")
print("\nCheck the diagnostic plot to see the actual data.")
