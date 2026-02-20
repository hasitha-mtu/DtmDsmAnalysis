"""
Single Tile Water Level Extraction

Extract water levels from ONE 1024×1024 tile + mask
No need to process all tiles or reassemble!

Requirements:
- One image tile (1024×1024)
- Corresponding mask tile (1024×1024)
- Know WHERE this tile is in the orthophoto (row, col position)
- WebODM DSM/DTM files
"""

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# HELPER: CALCULATE TILE POSITION MANUALLY (RECOMMENDED!)
# ============================================================================

def calculate_tile_position(tile_number, tile_size=1024, stride=768, orthophoto_width=63600):
    """
    Calculate tile position from tile number
    
    MUCH FASTER than automatic search!
    Use this if you know your tile extraction parameters.
    
    Parameters:
    -----------
    tile_number : int
        Tile number (e.g., 0, 1, 2, ... from tile_00000.png, tile_00001.png)
    tile_size : int
        Size of tiles (default 1024)
    stride : int
        Stride used when extracting tiles (default 768)
    orthophoto_width : int
        Width of orthophoto in pixels (default 63600 for your data)
    
    Returns:
    --------
    tuple: (row_start, col_start) position
    
    Example:
    --------
    >>> # For tile_00050.png with stride=768
    >>> pos = calculate_tile_position(50, tile_size=1024, stride=768)
    >>> print(pos)  # (768, 38400) or similar
    """
    
    # Calculate grid dimensions
    n_cols = (orthophoto_width - tile_size) // stride + 1
    
    # Calculate position
    tile_row_idx = tile_number // n_cols
    tile_col_idx = tile_number % n_cols
    
    row_start = tile_row_idx * stride
    col_start = tile_col_idx * stride
    
    return (row_start, col_start)

# ============================================================================
# SINGLE TILE GEOREFERENCING
# ============================================================================

def georeference_single_tile(tile_position, orthophoto_file, tile_size=1024):
    """
    Get georeferencing for a single tile
    
    Parameters:
    -----------
    tile_position : tuple
        (row_start, col_start) in orthophoto coordinates
        Example: (0, 0) = top-left tile
                 (0, 768) = second tile in first row (if stride=768)
    orthophoto_file : str
        Path to WebODM orthophoto
    tile_size : int
        Size of tile (default 1024)
    
    Returns:
    --------
    dict with 'transform', 'crs', 'bounds', 'window'
    """
    
    row_start, col_start = tile_position
    
    with rasterio.open(orthophoto_file) as src:
        # Get orthophoto georeferencing
        ortho_transform = src.transform
        ortho_crs = src.crs
        
        # Calculate world coordinates of tile corners
        # Top-left corner
        left, top = rasterio.transform.xy(ortho_transform, row_start, col_start, offset='ul')
        
        # Bottom-right corner
        right, bottom = rasterio.transform.xy(ortho_transform, 
                                               row_start + tile_size, 
                                               col_start + tile_size, 
                                               offset='ul')
        
        bounds = (left, bottom, right, top)
        
        # Create transform for this tile
        tile_transform = from_bounds(left, bottom, right, top, tile_size, tile_size)
        
        # Create window for reading from DSM/DTM
        window = Window(col_start, row_start, tile_size, tile_size)
        
        return {
            'transform': tile_transform,
            'crs': ortho_crs,
            'bounds': bounds,
            'window': window,
            'position': tile_position,
            'pixel_size': abs(ortho_transform.a)  # meters per pixel
        }

def find_tile_position_automatically(tile_image, orthophoto_file, search_region=None,
                                    chunk_size=10000):
    """
    Automatically find where a tile is located in the orthophoto
    Uses chunked template matching to handle large orthophotos
    
    **WARNING:** This can be slow for large orthophotos!
    Better to calculate position manually if you know tile extraction parameters.
    
    Parameters:
    -----------
    tile_image : numpy array or str
        The tile image to find
    orthophoto_file : str
        Path to orthophoto
    search_region : tuple (optional)
        (row_start, row_end, col_start, col_end) to limit search
        HIGHLY RECOMMENDED for large orthophotos!
    chunk_size : int
        Process orthophoto in chunks of this size (default 10000 pixels)
    
    Returns:
    --------
    tuple: (row_start, col_start) position in orthophoto
    """
    
    print("🔍 Searching for tile position in orthophoto...")
    print("   Using memory-efficient chunked search...")
    
    # Load tile
    if isinstance(tile_image, str):
        tile = cv2.imread(tile_image)
        if tile is None:
            raise ValueError(f"Could not load tile image: {tile_image}")
        tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
    else:
        tile = tile_image
    
    tile_h, tile_w = tile.shape[:2]
    
    # Convert to grayscale for matching
    tile_gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
    
    with rasterio.open(orthophoto_file) as src:
        ortho_h, ortho_w = src.height, src.width
        
        print(f"   Orthophoto: {ortho_h} × {ortho_w} pixels")
        print(f"   Tile: {tile_h} × {tile_w} pixels")
        
        # Define search region
        if search_region:
            search_row_start, search_row_end, search_col_start, search_col_end = search_region
        else:
            # Default: search entire orthophoto
            search_row_start, search_row_end = 0, ortho_h
            search_col_start, search_col_end = 0, ortho_w
        
        search_h = search_row_end - search_row_start
        search_w = search_col_end - search_col_start
        
        print(f"   Search region: {search_h} × {search_w} pixels")
        
        # Check if search region is reasonable
        if search_h * search_w > 100_000_000:  # >100M pixels
            print(f"\n   ⚠️  WARNING: Very large search region!")
            print(f"      This will take 5-10 minutes and use lots of memory.")
            print(f"      STRONGLY RECOMMEND: Specify search_region to limit area.")
            print(f"      Example: search_region=(0, 10000, 0, 10000)")
            print(f"\n   Press Ctrl+C to cancel...")
            import time
            time.sleep(3)
        
        best_match_val = -1
        best_match_pos = None
        
        # Search in chunks
        n_row_chunks = max(1, (search_h - tile_h) // chunk_size + 1)
        n_col_chunks = max(1, (search_w - tile_w) // chunk_size + 1)
        total_chunks = n_row_chunks * n_col_chunks
        
        print(f"   Processing in {n_row_chunks} × {n_col_chunks} = {total_chunks} chunks...")
        
        chunk_idx = 0
        for row_chunk in range(n_row_chunks):
            for col_chunk in range(n_col_chunks):
                # Calculate chunk boundaries (with overlap for tile)
                row_start = search_row_start + row_chunk * chunk_size
                col_start = search_col_start + col_chunk * chunk_size
                row_end = min(row_start + chunk_size + tile_h, search_row_end)
                col_end = min(col_start + chunk_size + tile_w, search_col_end)
                
                # Read chunk
                try:
                    window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
                    chunk = src.read([1, 2, 3], window=window)
                    
                    if chunk.size == 0:
                        continue
                    
                    chunk = chunk.transpose(1, 2, 0)
                    
                    # Skip if chunk is all black/white (likely padding)
                    if chunk.std() < 5:
                        continue
                    
                    chunk_gray = cv2.cvtColor(chunk, cv2.COLOR_RGB2GRAY)
                    
                    # Template matching on this chunk
                    if chunk_gray.shape[0] >= tile_h and chunk_gray.shape[1] >= tile_w:
                        result = cv2.matchTemplate(chunk_gray, tile_gray, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)
                        
                        # Update best match
                        if max_val > best_match_val:
                            best_match_val = max_val
                            best_match_pos = (row_start + max_loc[1], col_start + max_loc[0])
                
                except Exception as e:
                    print(f"   Warning: Error processing chunk {chunk_idx}: {str(e)[:50]}")
                    continue
                
                chunk_idx += 1
                if chunk_idx % max(1, total_chunks // 10) == 0:
                    print(f"      Progress: {chunk_idx}/{total_chunks} chunks ({chunk_idx/total_chunks*100:.0f}%) - Best match: {best_match_val:.3f}")
        
        if best_match_pos is None:
            raise ValueError(
                "Could not find tile in orthophoto!\n"
                "Try:\n"
                "  1. Specify search_region to limit search area\n"
                "  2. Use workflow_with_known_position() instead\n"
                "  3. Calculate position manually (see guide)"
            )
        
        tile_row, tile_col = best_match_pos
        
        print(f"\n   ✓ Found tile at position: ({tile_row}, {tile_col})")
        print(f"   Match confidence: {best_match_val:.3f}")
        
        if best_match_val < 0.7:
            print(f"   ⚠️  WARNING: Low confidence!")
            print(f"      Verify this is correct before using results.")
        
        return (tile_row, tile_col)

# ============================================================================
# WATER LEVEL EXTRACTION FOR SINGLE TILE
# ============================================================================

def extract_water_levels_single_tile(mask, tile_position, orthophoto_file,
                                     dsm_file, dtm_file, tile_size=1024):
    """
    Extract water levels from a single tile
    
    Parameters:
    -----------
    mask : numpy array or str
        Binary mask (1024×1024) where 1=water, 0=non-water
    tile_position : tuple
        (row_start, col_start) position in orthophoto
    orthophoto_file : str
        Path to WebODM orthophoto
    dsm_file : str
        Path to WebODM DSM
    dtm_file : str
        Path to WebODM DTM
    tile_size : int
        Tile size (default 1024)
    
    Returns:
    --------
    dict with water level data
    """
    
    print("="*70)
    print("SINGLE TILE WATER LEVEL EXTRACTION")
    print("="*70)
    
    # Load mask
    if isinstance(mask, str):
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    
    # Ensure binary
    mask = (mask > 127).astype(np.uint8)
    
    print(f"\n1. Mask loaded: {mask.shape}")
    print(f"   Water pixels: {np.sum(mask):,} ({np.sum(mask)/mask.size*100:.1f}%)")
    
    # Georeference tile
    print(f"\n2. Georeferencing tile...")
    print(f"   Tile position: row={tile_position[0]}, col={tile_position[1]}")
    
    georef = georeference_single_tile(tile_position, orthophoto_file, tile_size)
    
    print(f"   CRS: {georef['crs']}")
    print(f"   Pixel size: {georef['pixel_size']:.3f}m")
    print(f"   Bounds: {georef['bounds']}")
    
    # Extract water pixel coordinates
    print(f"\n3. Extracting water pixel coordinates...")
    water_coords = np.argwhere(mask > 0)  # [row, col]
    print(f"   Water pixels: {len(water_coords):,}")
    
    # Convert to world coordinates
    print(f"\n4. Converting to world coordinates...")
    rows, cols = water_coords[:, 0], water_coords[:, 1]
    xs, ys = rasterio.transform.xy(georef['transform'], rows, cols)
    world_coords = np.column_stack([xs, ys])
    
    # Sample DSM for this tile's region
    print(f"\n5. Sampling DSM (water surface)...")
    with rasterio.open(dsm_file) as src:
        # Read tile region from DSM
        dsm_tile = src.read(1, window=georef['window'])
        dsm_nodata = src.nodata
        
        # Mask nodata
        if dsm_nodata is not None:
            dsm_tile = np.where(dsm_tile == dsm_nodata, np.nan, dsm_tile.astype(np.float32))
        
        # Sample at water pixels
        water_surface = dsm_tile[rows, cols]
    
    valid_dsm = ~np.isnan(water_surface)
    print(f"   Valid samples: {np.sum(valid_dsm):,} / {len(water_surface):,}")
    print(f"   Water surface range: {np.nanmin(water_surface):.2f} to {np.nanmax(water_surface):.2f}m")
    print(f"   Mean: {np.nanmean(water_surface):.2f}m")
    
    # Sample DTM for this tile's region
    print(f"\n6. Sampling DTM (riverbed)...")
    with rasterio.open(dtm_file) as src:
        # Read tile region from DTM
        dtm_tile = src.read(1, window=georef['window'])
        dtm_nodata = src.nodata
        
        # Mask nodata
        if dtm_nodata is not None:
            dtm_tile = np.where(dtm_tile == dtm_nodata, np.nan, dtm_tile.astype(np.float32))
        
        # Sample at water pixels
        riverbed = dtm_tile[rows, cols]
    
    valid_dtm = ~np.isnan(riverbed)
    print(f"   Valid samples: {np.sum(valid_dtm):,} / {len(riverbed):,}")
    print(f"   Riverbed range: {np.nanmin(riverbed):.2f} to {np.nanmax(riverbed):.2f}m")
    print(f"   Mean: {np.nanmean(riverbed):.2f}m")
    
    # Calculate depths
    print(f"\n7. Calculating water depths...")
    depth = water_surface - riverbed
    valid_mask = valid_dsm & valid_dtm & (depth > 0) & (depth < 10)
    
    print(f"   Valid depths: {np.sum(valid_mask):,}")
    
    if np.sum(valid_mask) > 0:
        print(f"   Depth range: {np.min(depth[valid_mask]):.2f} to {np.max(depth[valid_mask]):.2f}m")
        print(f"   Mean depth: {np.mean(depth[valid_mask]):.2f}m")
        print(f"   Median depth: {np.median(depth[valid_mask]):.2f}m")
    else:
        print(f"   ⚠️  No valid depth measurements!")
    
    return {
        'mask': mask,
        'tile_position': tile_position,
        'georeference': georef,
        'water_coords_pixel': water_coords,
        'water_coords_world': world_coords,
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

def visualize_single_tile_results(tile_image, results, output_file='single_tile_analysis.png'):
    """
    Visualize water level extraction results for single tile
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Load tile image if path provided
    if isinstance(tile_image, str):
        tile_img = cv2.imread(tile_image)
        tile_img = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)
    else:
        tile_img = tile_image
    
    # 1. Original image
    axes[0, 0].imshow(tile_img)
    axes[0, 0].set_title('Original Tile Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. Mask overlay
    overlay = tile_img.copy()
    mask_colored = np.zeros_like(overlay)
    mask_colored[results['mask'] > 0] = [0, 255, 255]  # Cyan for water
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title('Water Mask Overlay', fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. DSM
    im3 = axes[0, 2].imshow(results['dsm_tile'], cmap='terrain')
    axes[0, 2].set_title('DSM (Water Surface)', fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], label='Elevation (m)', fraction=0.046)
    
    # 4. DTM
    im4 = axes[1, 0].imshow(results['dtm_tile'], cmap='terrain')
    axes[1, 0].set_title('DTM (Riverbed)', fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], label='Elevation (m)', fraction=0.046)
    
    # 5. Depth map
    depth_map = np.full(results['mask'].shape, np.nan)
    rows = results['water_coords_pixel'][:, 0]
    cols = results['water_coords_pixel'][:, 1]
    valid = results['valid_mask']
    depth_map[rows[valid], cols[valid]] = results['depth'][valid]
    
    im5 = axes[1, 1].imshow(depth_map, cmap='YlGnBu', vmin=0)
    axes[1, 1].set_title('Water Depth', fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1], label='Depth (m)', fraction=0.046)
    
    # 6. Depth histogram
    if np.sum(valid) > 0:
        valid_depths = results['depth'][valid]
        axes[1, 2].hist(valid_depths, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 2].axvline(np.mean(valid_depths), color='red', linestyle='--',
                          label=f'Mean: {np.mean(valid_depths):.2f}m')
        axes[1, 2].set_xlabel('Depth (m)', fontweight='bold')
        axes[1, 2].set_ylabel('Pixel Count', fontweight='bold')
        axes[1, 2].set_title('Depth Distribution', fontweight='bold')
        axes[1, 2].legend()
        axes[1, 2].grid(alpha=0.3)
    
    # Overall title with info
    georef = results['georeference']
    plt.suptitle(
        f'Single Tile Water Level Analysis\n'
        f'Position: ({georef["position"][0]}, {georef["position"][1]}) | '
        f'Bounds: ({georef["bounds"][0]:.0f}, {georef["bounds"][1]:.0f}) to '
        f'({georef["bounds"][2]:.0f}, {georef["bounds"][3]:.0f}) | '
        f'CRS: {georef["crs"]}',
        fontsize=12, fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_file}")
    plt.close()

# ============================================================================
# MAIN WORKFLOWS
# ============================================================================

def workflow_with_known_position(tile_image, mask, tile_position,
                                orthophoto_file, dsm_file, dtm_file):
    """
    Workflow when you KNOW the tile position
    
    Parameters:
    -----------
    tile_position : tuple
        (row_start, col_start) in orthophoto pixels
        Example: (0, 0), (0, 768), (768, 0), etc.
    """
    
    print("\n📍 WORKFLOW: Known Tile Position")
    
    # Extract water levels
    results = extract_water_levels_single_tile(
        mask=mask,
        tile_position=tile_position,
        orthophoto_file=orthophoto_file,
        dsm_file=dsm_file,
        dtm_file=dtm_file,
        tile_size=1024
    )
    
    # Visualize
    visualize_single_tile_results(tile_image, results)
    
    return results

def workflow_with_automatic_search(tile_image, mask, orthophoto_file,
                                   dsm_file, dtm_file, search_region=None):
    """
    Workflow when you DON'T KNOW tile position
    Automatically finds where tile is located (slower)
    
    Parameters:
    -----------
    search_region : tuple (optional)
        (row_start, row_end, col_start, col_end) to limit search area
        Example: (0, 10000, 0, 10000) searches top-left corner only
    """
    
    print("\n🔍 WORKFLOW: Automatic Tile Position Search")
    
    # Find tile position
    tile_position = find_tile_position_automatically(
        tile_image=tile_image,
        orthophoto_file=orthophoto_file,
        search_region=search_region
    )
    
    # Extract water levels
    results = extract_water_levels_single_tile(
        mask=mask,
        tile_position=tile_position,
        orthophoto_file=orthophoto_file,
        dsm_file=dsm_file,
        dtm_file=dtm_file,
        tile_size=1024
    )
    
    # Visualize
    visualize_single_tile_results(tile_image, results)
    
    return results

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    # File paths
    tile_image = r"C:\\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\images\DJI_20250728101825_0553_V_patch_0359.jpg"
    mask_image = r"C:\\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\masks\DJI_20250728101825_0553_V_patch_0359.png"
    orthophoto_file = r"C:\\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_orthophoto\odm_orthophoto.tif"
    dsm_file = r"C:\\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_dem\dsm_aligned.tif"
    dtm_file = r"C:\\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_dem\dtm_aligned.tif"

    # ========================================================================
    # OPTION 1: You KNOW the tile position
    # ========================================================================
    
    # Example: Second row, third column with stride=768
    # tile_position = (768, 1536)  # row_start=768, col_start=1536
    
    # Example: Top-left tile
    # tile_position = (0, 0)
    #
    # print("\n" + "🌊 "*35)
    # print("OPTION 1: Extract water levels with KNOWN position")
    # print("🌊 "*35)
    #
    # results = workflow_with_known_position(
    #     tile_image=tile_image,
    #     mask=mask_image,
    #     tile_position=tile_position,
    #     orthophoto_file=orthophoto_file,
    #     dsm_file=dsm_file,
    #     dtm_file=dtm_file
    # )
    
    # ========================================================================
    # OPTION 2: You DON'T KNOW the tile position (automatic search)
    # ========================================================================
    

    print("\n" + "🌊 "*35)
    print("OPTION 2: Extract water levels with AUTOMATIC search")
    print("🌊 "*35)

    # Optional: limit search to a region (faster)
    # search_region = (0, 10000, 0, 20000)  # top-left quarter only

    results = workflow_with_automatic_search(
        tile_image=tile_image,
        mask=mask_image,
        orthophoto_file=orthophoto_file,
        dsm_file=dsm_file,
        dtm_file=dtm_file,
        search_region=None  # or specify region
    )

    
    # ========================================================================
    # Display final results
    # ========================================================================
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    valid = results['valid_mask']
    if np.sum(valid) > 0:
        print(f"\n📊 Water Level Statistics:")
        print(f"   Water pixels: {np.sum(results['mask']):,}")
        print(f"   Valid depths: {np.sum(valid):,}")
        print(f"   Mean water surface: {np.mean(results['water_surface'][valid]):.2f}m")
        print(f"   Mean riverbed: {np.mean(results['riverbed'][valid]):.2f}m")
        print(f"   Mean depth: {np.mean(results['depth'][valid]):.2f}m")
        print(f"   Median depth: {np.median(results['depth'][valid]):.2f}m")
        print(f"   Max depth: {np.max(results['depth'][valid]):.2f}m")
        
        # Estimate river width in this tile
        rows_with_water = np.unique(results['water_coords_pixel'][valid, 0])
        if len(rows_with_water) > 10:
            widths = []
            for row in rows_with_water:
                cols_in_row = results['water_coords_pixel'][
                    results['water_coords_pixel'][:, 0] == row, 1
                ]
                if len(cols_in_row) > 0:
                    width_pixels = np.max(cols_in_row) - np.min(cols_in_row) + 1
                    width_meters = width_pixels * results['georeference']['pixel_size']
                    widths.append(width_meters)
            
            if widths:
                print(f"\n📏 River Geometry in This Tile:")
                print(f"   Mean width: {np.mean(widths):.2f}m")
                print(f"   Max width: {np.max(widths):.2f}m")
    else:
        print("\n⚠️  No valid water depth measurements in this tile!")
        print("   Check if mask covers water and DSM/DTM overlap this area")
    
    print("\n✅ Single tile analysis complete!")
