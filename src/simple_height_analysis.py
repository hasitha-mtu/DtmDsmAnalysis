"""
Quick DSM-DTM height difference calculator and visualizer
Simple version - minimal code for fast analysis
Handles large datasets automatically
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt

# File paths - UPDATE THESE!
dsm_file = "path/to/your/dsm.tif"
dtm_file = "path/to/your/dtm.tif"

# Load files
print("Loading files...")
with rasterio.open(dsm_file) as src:
    dsm = src.read(1).astype(np.float32)
    profile = src.profile

with rasterio.open(dtm_file) as src:
    dtm = src.read(1).astype(np.float32)

# Calculate height difference (nDSM)
ndsm = dsm - dtm
ndsm[ndsm < 0] = 0  # Remove negative values

# Statistics
print(f"\nHeight Statistics:")
print(f"  Min: {np.nanmin(ndsm):.2f} m")
print(f"  Max: {np.nanmax(ndsm):.2f} m")
print(f"  Mean: {np.nanmean(ndsm):.2f} m")
print(f"  Median: {np.nanmedian(ndsm):.2f} m")

# Downsample for visualization if needed
max_dimension = 2000
height, width = dsm.shape

if height > max_dimension or width > max_dimension:
    downsample_factor = max(height // max_dimension, width // max_dimension, 1)
    print(f"\nDataset is large ({height} × {width})")
    print(f"Downsampling by factor of {downsample_factor} for visualization...")
    
    dsm_plot = dsm[::downsample_factor, ::downsample_factor]
    dtm_plot = dtm[::downsample_factor, ::downsample_factor]
    ndsm_plot = ndsm[::downsample_factor, ::downsample_factor]
    
    print(f"Visualization size: {dsm_plot.shape}")
else:
    dsm_plot = dsm
    dtm_plot = dtm
    ndsm_plot = ndsm

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(dsm_plot, cmap='terrain')
axes[0].set_title('DSM (Surface)', fontweight='bold')
axes[0].axis('off')

axes[1].imshow(dtm_plot, cmap='terrain')
axes[1].set_title('DTM (Terrain)', fontweight='bold')
axes[1].axis('off')

im = axes[2].imshow(ndsm_plot, cmap='YlGnBu', vmin=0)
axes[2].set_title('Height Above Ground (m)', fontweight='bold')
axes[2].axis('off')
plt.colorbar(im, ax=axes[2], label='Height (m)')

plt.tight_layout()
plt.savefig('height_analysis.png', dpi=150, bbox_inches='tight')
print("\nSaved: height_analysis.png")
plt.show()

# Save nDSM (full resolution)
profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
with rasterio.open('ndsm.tif', 'w', **profile) as dst:
    dst.write(ndsm.astype(rasterio.float32), 1)

print("Saved: ndsm.tif (full resolution)")
print(f"\nNote: Plots were downsampled for visualization.")
print(f"      Statistics and saved ndsm.tif use full resolution data.")
