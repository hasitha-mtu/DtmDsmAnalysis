"""
SIMPLE SOLUTION: Calculate tile position manually and extract water levels

No automatic search needed!
Just provide tile number and extraction parameters.
"""

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
import cv2
import matplotlib.pyplot as plt

# ============================================================================
# CALCULATE TILE POSITION (INSTANT!)
# ============================================================================

def calculate_tile_position(tile_number, tile_size=1024, stride=768, 
                           orthophoto_width=63600):
    """
    Calculate tile position from tile number
    
    Parameters:
    -----------
    tile_number : int
        Tile number (e.g., 359 from patch_0359.jpg)
    tile_size : int
        Size of tiles (default 1024)
    stride : int
        Stride used when extracting tiles
    orthophoto_width : int
        Width of your orthophoto
    
    Returns:
    --------
    tuple: (row_start, col_start)
    """
    
    # Calculate grid dimensions
    n_cols = (orthophoto_width - tile_size) // stride + 1
    
    # Calculate row and column indices
    tile_row_idx = tile_number // n_cols
    tile_col_idx = tile_number % n_cols
    
    # Calculate pixel positions
    row_start = tile_row_idx * stride
    col_start = tile_col_idx * stride
    
    print(f"\nTile {tile_number}:")
    print(f"  Grid position: row {tile_row_idx}, col {tile_col_idx}")
    print(f"  Pixel position: ({row_start}, {col_start})")
    
    return (row_start, col_start)

# ============================================================================
# EXTRACT WATER LEVELS
# ============================================================================

def extract_water_levels_single_tile(mask, tile_position, orthophoto_file,
                                     dsm_file, dtm_file, tile_size=1024):
    """
    Extract water levels from a single tile
    """
    
    print("\n" + "="*70)
    print("WATER LEVEL EXTRACTION")
    print("="*70)
    
    # Load mask
    if isinstance(mask, str):
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    
    mask = (mask > 127).astype(np.uint8)
    
    print(f"\n1. Mask: {mask.shape}")
    print(f"   Water pixels: {np.sum(mask):,}")
    
    # Get georeferencing
    row_start, col_start = tile_position
    
    with rasterio.open(orthophoto_file) as src:
        ortho_transform = src.transform
        ortho_crs = src.crs
        
        # World coordinates of tile corners
        left, top = rasterio.transform.xy(ortho_transform, row_start, col_start, offset='ul')
        right, bottom = rasterio.transform.xy(ortho_transform, 
                                               row_start + tile_size, 
                                               col_start + tile_size, 
                                               offset='ul')
        
        bounds = (left, bottom, right, top)
        tile_transform = from_bounds(left, bottom, right, top, tile_size, tile_size)
        pixel_size = abs(ortho_transform.a)
        
        print(f"\n2. Georeferencing:")
        print(f"   CRS: {ortho_crs}")
        print(f"   Bounds: {bounds}")
        print(f"   Pixel size: {pixel_size:.3f}m")
    
    # Extract water pixels
    water_coords = np.argwhere(mask > 0)
    rows, cols = water_coords[:, 0], water_coords[:, 1]
    
    print(f"\n3. Water pixels: {len(water_coords):,}")
    
    # Sample DSM
    print(f"\n4. Sampling DSM...")
    window = Window(col_start, row_start, tile_size, tile_size)
    
    with rasterio.open(dsm_file) as src:
        dsm_tile = src.read(1, window=window)
        dsm_nodata = src.nodata
        if dsm_nodata is not None:
            dsm_tile = np.where(dsm_tile == dsm_nodata, np.nan, dsm_tile.astype(np.float32))
        
        water_surface = dsm_tile[rows, cols]
    
    valid_dsm = ~np.isnan(water_surface)
    print(f"   Valid: {np.sum(valid_dsm):,}")
    print(f"   Mean: {np.nanmean(water_surface):.2f}m")
    
    # Sample DTM
    print(f"\n5. Sampling DTM...")
    
    with rasterio.open(dtm_file) as src:
        dtm_tile = src.read(1, window=window)
        dtm_nodata = src.nodata
        if dtm_nodata is not None:
            dtm_tile = np.where(dtm_tile == dtm_nodata, np.nan, dtm_tile.astype(np.float32))
        
        riverbed = dtm_tile[rows, cols]
    
    valid_dtm = ~np.isnan(riverbed)
    print(f"   Valid: {np.sum(valid_dtm):,}")
    print(f"   Mean: {np.nanmean(riverbed):.2f}m")
    
    # Calculate depths
    print(f"\n6. Calculating depths...")
    depth = water_surface - riverbed
    valid_mask = valid_dsm & valid_dtm & (depth > 0) & (depth < 10)
    
    print(f"   Valid depths: {np.sum(valid_mask):,}")
    
    if np.sum(valid_mask) > 0:
        print(f"   Mean: {np.mean(depth[valid_mask]):.2f}m")
        print(f"   Median: {np.median(depth[valid_mask]):.2f}m")
        print(f"   Range: {np.min(depth[valid_mask]):.2f} to {np.max(depth[valid_mask]):.2f}m")
    
    return {
        'mask': mask,
        'tile_position': tile_position,
        'bounds': bounds,
        'crs': ortho_crs,
        'pixel_size': pixel_size,
        'water_surface': water_surface,
        'riverbed': riverbed,
        'depth': depth,
        'valid_mask': valid_mask,
        'dsm_tile': dsm_tile,
        'dtm_tile': dtm_tile
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_results(tile_image, mask, results, output_file='analysis.png'):
    """Create visualization"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Load tile image
    if isinstance(tile_image, str):
        tile_img = cv2.imread(tile_image)
        tile_img = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)
    else:
        tile_img = tile_image
    
    # 1. Original image
    axes[0, 0].imshow(tile_img)
    axes[0, 0].set_title('Original Tile', fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. Mask overlay
    overlay = tile_img.copy()
    mask_colored = np.zeros_like(overlay)
    mask_colored[results['mask'] > 0] = [0, 255, 255]
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title('Water Mask', fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. DSM
    im3 = axes[0, 2].imshow(results['dsm_tile'], cmap='terrain')
    axes[0, 2].set_title('DSM (Surface)', fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], label='Elevation (m)', fraction=0.046)
    
    # 4. DTM
    im4 = axes[1, 0].imshow(results['dtm_tile'], cmap='terrain')
    axes[1, 0].set_title('DTM (Riverbed)', fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], label='Elevation (m)', fraction=0.046)
    
    # 5. Depth map
    depth_map = np.full(results['mask'].shape, np.nan)
    water_coords = np.argwhere(results['mask'] > 0)
    rows, cols = water_coords[:, 0], water_coords[:, 1]
    valid = results['valid_mask']
    depth_map[rows[valid], cols[valid]] = results['depth'][valid]
    
    im5 = axes[1, 1].imshow(depth_map, cmap='YlGnBu', vmin=0)
    axes[1, 1].set_title('Water Depth', fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1], label='Depth (m)', fraction=0.046)
    
    # 6. Histogram
    if np.sum(valid) > 0:
        depths = results['depth'][valid]
        axes[1, 2].hist(depths, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 2].axvline(np.mean(depths), color='red', linestyle='--',
                          label=f'Mean: {np.mean(depths):.2f}m')
        axes[1, 2].set_xlabel('Depth (m)', fontweight='bold')
        axes[1, 2].set_ylabel('Count', fontweight='bold')
        axes[1, 2].set_title('Depth Distribution', fontweight='bold')
        axes[1, 2].legend()
        axes[1, 2].grid(alpha=0.3)
    
    plt.suptitle(f'Water Level Analysis - Tile at {results["tile_position"]}',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    # =======================================================================
    # YOUR FILES - UPDATE THESE!
    # =======================================================================
    
    tile_image = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\images\DJI_20250728101825_0553_V_patch_0359.jpg"
    mask_image = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\masks\DJI_20250728101825_0553_V_patch_0359.png"
    orthophoto_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_orthophoto\odm_orthophoto.tif"
    dsm_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_dem\dsm_aligned.tif"
    dtm_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_dem\dtm_aligned.tif"

    with rasterio.open(orthophoto_file) as src:
        print("orthophoto_file CRS:", src.crs)
        print("orthophoto_file Metadata:", src.tags())

    with rasterio.open(dsm_file) as src:
        print("dsm_file CRS:", src.crs)
        print("dsm_file Metadata:", src.tags())

    with rasterio.open(dtm_file) as src:
        print("dtm_file CRS:", src.crs)
        print("dtm_file Metadata:", src.tags())
    
    # =======================================================================
    # TILE PARAMETERS - MUST MATCH YOUR EXTRACTION!
    # =======================================================================
    
    tile_number = 156  # From filename: patch_0359
    tile_size = 1024
    stride = 768
    
    # Get orthophoto dimensions
    with rasterio.open(orthophoto_file) as src:
        orthophoto_width = src.width
        orthophoto_height = src.height
    
    print("="*70)
    print("SIMPLE TILE WATER LEVEL EXTRACTION")
    print("="*70)
    print(f"\nOrthophoto: {orthophoto_height} × {orthophoto_width} pixels")
    print(f"Tile size: {tile_size}")
    print(f"Stride: {stride}")
    
    # =======================================================================
    # CALCULATE POSITION (INSTANT!)
    # =======================================================================
    
    tile_position = calculate_tile_position(
        tile_number=tile_number,
        tile_size=tile_size,
        stride=stride,
        orthophoto_width=orthophoto_width
    )
    
    # =======================================================================
    # EXTRACT WATER LEVELS
    # =======================================================================
    
    results = extract_water_levels_single_tile(
        mask=mask_image,
        tile_position=tile_position,
        orthophoto_file=orthophoto_file,
        dsm_file=dsm_file,
        dtm_file=dtm_file,
        tile_size=tile_size
    )
    
    # =======================================================================
    # VISUALIZE
    # =======================================================================
    
    visualize_results(tile_image, mask_image, results)
    
    # =======================================================================
    # DISPLAY RESULTS
    # =======================================================================
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    valid = results['valid_mask']
    
    if np.sum(valid) > 0:
        depths = results['depth'][valid]
        surfaces = results['water_surface'][valid]
        riverbeds = results['riverbed'][valid]
        
        print(f"\n📊 Water Statistics:")
        print(f"   Water pixels: {np.sum(results['mask']):,}")
        print(f"   Valid depths: {np.sum(valid):,}")
        
        print(f"\n📏 Elevations (WGS84 Ellipsoid):")
        print(f"   Water surface: {surfaces.mean():.2f} ± {surfaces.std():.2f}m")
        print(f"   Riverbed: {riverbeds.mean():.2f} ± {riverbeds.std():.2f}m")
        
        print(f"\n💧 Water Depths:")
        print(f"   Mean: {depths.mean():.2f}m")
        print(f"   Median: {np.median(depths):.2f}m")
        print(f"   Range: {depths.min():.2f} to {depths.max():.2f}m")
        
        print(f"\n🗺️  Location:")
        print(f"   CRS: {results['crs']}")
        print(f"   Bounds: {results['bounds']}")
        
    else:
        print("\n⚠️  No valid water depths found!")
        print("   Possible reasons:")
        print("   - Mask has no water pixels")
        print("   - DSM/DTM don't cover this area")
        print("   - Tile position incorrect (wrong stride?)")
    
    print("\n✅ Analysis complete!")
