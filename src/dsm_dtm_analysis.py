"""
Calculate and visualize height differences between DSM and DTM
(nDSM = DSM - DTM shows object heights above ground)
Handles large datasets automatically
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.axes_grid1 import make_axes_locatable

def downsample_for_plot(array, max_dim=2000):
    """Downsample array for visualization if too large"""
    height, width = array.shape
    if height > max_dim or width > max_dim:
        factor = max(height // max_dim, width // max_dim, 1)
        return array[::factor, ::factor], factor
    return array, 1

def load_raster(filepath):
    """Load a raster file (GeoTIFF) and return data and metadata"""
    with rasterio.open(filepath) as src:
        data = src.read(1).astype(np.float32)  # Convert to float32 to save memory
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    return data, transform, crs, nodata

def calculate_ndsm(dsm, dtm, dsm_nodata=None, dtm_nodata=None):
    """Calculate normalized DSM (height above ground)"""
    # Create a copy to avoid modifying original data
    ndsm = dsm.copy().astype(np.float32)
    
    # Handle nodata values
    if dsm_nodata is not None:
        ndsm[dsm == dsm_nodata] = np.nan
    if dtm_nodata is not None:
        ndsm[dtm == dtm_nodata] = np.nan
    
    # Calculate difference (DSM - DTM)
    valid_mask = ~np.isnan(ndsm)
    ndsm[valid_mask] = dsm[valid_mask] - dtm[valid_mask]
    
    # Set negative values to 0 (shouldn't have objects below ground)
    ndsm[ndsm < 0] = 0
    
    return ndsm

def visualize_height_differences(dsm, dtm, ndsm, figsize=(18, 6)):
    """Create comprehensive visualization of DSM, DTM, and height differences"""
    # Downsample for visualization
    dsm_plot, factor = downsample_for_plot(dsm)
    dtm_plot, _ = downsample_for_plot(dtm)
    ndsm_plot, _ = downsample_for_plot(ndsm)
    
    if factor > 1:
        print(f"  Downsampling by factor of {factor} for visualization...")
        print(f"  Visualization size: {dsm_plot.shape}")
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Common parameters for visualization
    vmin_dem = np.nanmin([np.nanmin(dsm), np.nanmin(dtm)])
    vmax_dem = np.nanmax([np.nanmax(dsm), np.nanmax(dtm)])
    
    # 1. DSM (Digital Surface Model)
    im1 = axes[0].imshow(dsm_plot, cmap='terrain', vmin=vmin_dem, vmax=vmax_dem)
    axes[0].set_title('DSM (Digital Surface Model)\nTop of canopy/buildings', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    divider1 = make_axes_locatable(axes[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar1.set_label('Elevation (m)', rotation=270, labelpad=15)
    
    # 2. DTM (Digital Terrain Model)
    im2 = axes[1].imshow(dtm_plot, cmap='terrain', vmin=vmin_dem, vmax=vmax_dem)
    axes[1].set_title('DTM (Digital Terrain Model)\nBare earth elevation', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    divider2 = make_axes_locatable(axes[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar2.set_label('Elevation (m)', rotation=270, labelpad=15)
    
    # 3. nDSM (Height above ground)
    im3 = axes[2].imshow(ndsm_plot, cmap='YlGnBu', vmin=0, vmax=np.nanmax(ndsm))
    axes[2].set_title('nDSM (Height Above Ground)\nDSM - DTM', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    divider3 = make_axes_locatable(axes[2])
    cax3 = divider3.append_axes("right", size="5%", pad=0.1)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar3.set_label('Height (m)', rotation=270, labelpad=15)
    
    if factor > 1:
        fig.text(0.5, 0.02, f'Note: Plots downsampled {factor}x for visualization. Statistics use full resolution.',
                ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    return fig

def visualize_ndsm_detailed(ndsm, figsize=(16, 12)):
    """Create detailed multi-view visualization of height differences"""
    # Downsample for visualization
    ndsm_plot, factor = downsample_for_plot(ndsm)
    
    if factor > 1:
        print(f"  Downsampling by factor of {factor} for visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Basic height map
    im1 = axes[0, 0].imshow(ndsm_plot, cmap='YlGnBu', vmin=0)
    axes[0, 0].set_title('Height Above Ground (Blue-Green)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    divider1 = make_axes_locatable(axes[0, 0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im1, cax=cax1, label='Height (m)')
    
    # 2. Hot colormap for height
    im2 = axes[0, 1].imshow(ndsm_plot, cmap='hot', vmin=0)
    axes[0, 1].set_title('Height Above Ground (Hot)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    divider2 = make_axes_locatable(axes[0, 1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im2, cax=cax2, label='Height (m)')
    
    # 3. Hillshade effect
    ls = LightSource(azdeg=315, altdeg=45)
    try:
        hillshade = ls.hillshade(ndsm_plot, vert_exag=2, dx=1, dy=1)
        axes[1, 0].imshow(hillshade, cmap='gray')
        axes[1, 0].set_title('Hillshade (Shows topographic relief)', fontsize=12, fontweight='bold')
    except:
        axes[1, 0].imshow(ndsm_plot, cmap='gray')
        axes[1, 0].set_title('Height Map (Grayscale)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 4. Height classes
    height_classes = np.digitize(ndsm_plot, bins=[0, 2, 5, 10, 15, 100])
    im4 = axes[1, 1].imshow(height_classes, cmap='RdYlGn_r', vmin=1, vmax=5)
    axes[1, 1].set_title('Height Classes\n(0-2m, 2-5m, 5-10m, 10-15m, >15m)', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    divider4 = make_axes_locatable(axes[1, 1])
    cax4 = divider4.append_axes("right", size="5%", pad=0.1)
    cbar4 = plt.colorbar(im4, cax=cax4, ticks=[1, 2, 3, 4, 5])
    cbar4.ax.set_yticklabels(['0-2m', '2-5m', '5-10m', '10-15m', '>15m'])
    
    if factor > 1:
        fig.text(0.5, 0.02, f'Note: Plots downsampled {factor}x for visualization. Statistics use full resolution.',
                ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    return fig

def print_statistics(dsm, dtm, ndsm):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("HEIGHT ANALYSIS STATISTICS")
    print("="*60)
    
    print(f"\nDSM (Digital Surface Model):")
    print(f"  Min elevation: {np.nanmin(dsm):.2f} m")
    print(f"  Max elevation: {np.nanmax(dsm):.2f} m")
    print(f"  Mean elevation: {np.nanmean(dsm):.2f} m")
    print(f"  Std deviation: {np.nanstd(dsm):.2f} m")
    
    print(f"\nDTM (Digital Terrain Model):")
    print(f"  Min elevation: {np.nanmin(dtm):.2f} m")
    print(f"  Max elevation: {np.nanmax(dtm):.2f} m")
    print(f"  Mean elevation: {np.nanmean(dtm):.2f} m")
    print(f"  Std deviation: {np.nanstd(dtm):.2f} m")
    
    print(f"\nnDSM (Height Above Ground):")
    print(f"  Min height: {np.nanmin(ndsm):.2f} m")
    print(f"  Max height: {np.nanmax(ndsm):.2f} m")
    print(f"  Mean height: {np.nanmean(ndsm):.2f} m")
    print(f"  Std deviation: {np.nanstd(ndsm):.2f} m")
    
    # Vegetation/object coverage statistics
    total_pixels = np.sum(~np.isnan(ndsm))
    vegetation_pixels = np.sum(ndsm > 0.5)  # Assume >0.5m is vegetation/objects
    
    print(f"\nVegetation/Object Coverage:")
    print(f"  Pixels with objects (>0.5m): {vegetation_pixels:,} ({vegetation_pixels/total_pixels*100:.1f}%)")
    print(f"  Ground pixels (≤0.5m): {total_pixels - vegetation_pixels:,} ({(1-vegetation_pixels/total_pixels)*100:.1f}%)")
    
    # Height distribution
    print(f"\nHeight Distribution:")
    height_ranges = [(0, 2), (2, 5), (5, 10), (10, 15), (15, np.inf)]
    for low, high in height_ranges:
        count = np.sum((ndsm >= low) & (ndsm < high))
        label = f">{low}m" if high == np.inf else f"{low}-{high}m"
        print(f"  {label:8s}: {count:,} pixels ({count/total_pixels*100:.1f}%)")
    
    print("="*60 + "\n")

# Example usage
if __name__ == "__main__":
    # Replace these with your actual file paths
    dsm_file = "path/to/your/dsm.tif"
    dtm_file = "path/to/your/dtm.tif"
    
    print("Loading DSM and DTM files...")
    dsm, dsm_transform, dsm_crs, dsm_nodata = load_raster(dsm_file)
    dtm, dtm_transform, dtm_crs, dtm_nodata = load_raster(dtm_file)
    
    print(f"DSM shape: {dsm.shape}")
    print(f"DTM shape: {dtm.shape}")
    
    # Calculate height differences
    print("\nCalculating height differences (nDSM)...")
    ndsm = calculate_ndsm(dsm, dtm, dsm_nodata, dtm_nodata)
    
    # Print statistics
    print_statistics(dsm, dtm, ndsm)
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Comparison view
    fig1 = visualize_height_differences(dsm, dtm, ndsm)
    plt.savefig('height_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: height_comparison.png")
    
    # Detailed nDSM view
    fig2 = visualize_ndsm_detailed(ndsm)
    plt.savefig('ndsm_detailed.png', dpi=150, bbox_inches='tight')
    print("Saved: ndsm_detailed.png")
    
    plt.show()
    
    # Optional: Save nDSM as GeoTIFF
    print("\nSaving nDSM as GeoTIFF...")
    with rasterio.open(
        'ndsm_output.tif',
        'w',
        driver='GTiff',
        height=ndsm.shape[0],
        width=ndsm.shape[1],
        count=1,
        dtype=ndsm.dtype,
        crs=dsm_crs,
        transform=dsm_transform,
        nodata=np.nan
    ) as dst:
        dst.write(ndsm, 1)
    
    print("Saved: ndsm_output.tif")
    print("\nDone!")
