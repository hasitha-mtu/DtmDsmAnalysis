"""
Water Depth Using Bluesky DTM + WebODM DSM

Bluesky DTM (5m) penetrates canopy → true riverbed
WebODM DSM (0.061m) → water surface

Key correction: Bluesky uses Malin Head orthometric heights
                WebODM uses WGS84 ellipsoid heights
                Offset for Ireland: +58.0m (add to Bluesky to match WebODM)

Resampling: 5m → 0.061m is valid because riverbed is a smooth surface
"""

import numpy as np
import rasterio
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.enums import Resampling
from rasterio.warp import reproject
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.interpolate import RectBivariateSpline


# =======================================================================
# YOUR FILES
# =======================================================================

orthophoto_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_orthophoto\odm_orthophoto.tif"
webodm_dsm_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_dem\dsm_aligned.tif"
bluesky_dtm_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\5m_dtm\bluesky_dtm_utm29n.tif"

# Geoid-ellipsoid separation for Ireland (Malin Head → WGS84 ellipsoid)
# This converts Bluesky orthometric heights to WebODM ellipsoid heights
GEOID_OFFSET = 58.0  # metres - ADD to Bluesky to match WebODM datum

# =======================================================================
# RESAMPLING METHODS (choose one)
# =======================================================================

def resample_bluesky_dtm_to_tile(bluesky_dtm_file, bounds, target_shape,
                                  method='bicubic'):
    """
    Resample Bluesky DTM from 5m to match WebODM resolution for a tile
    
    Parameters:
    -----------
    bounds : tuple (left, bottom, right, top) in WebODM CRS (EPSG:32629)
    target_shape : tuple (height, width) - size of WebODM tile (1024, 1024)
    method : 'bilinear', 'bicubic', 'spline'
        - bilinear: Fast, smooth, adequate for gentle terrain
        - bicubic: Smoother curves, better for riverbed (RECOMMENDED)
        - spline: Best quality but slower
    
    Returns:
    --------
    numpy array, shape = target_shape, values in WebODM ellipsoid metres
    """
    
    left, bottom, right, top = bounds
    target_h, target_w = target_shape
    
    with rasterio.open(bluesky_dtm_file) as src:
        
        # Check CRS - reproject if needed
        if src.crs.to_epsg() != 32629:
            print(f"   Bluesky CRS: {src.crs} → reprojecting to EPSG:32629")
            dtm_coarse = _read_and_reproject_bluesky(
                bluesky_dtm_file, bounds, target_shape
            )
        else:
            # Same CRS - just read and resample
            window = window_from_bounds(left, bottom, right, top, 
                                       transform=src.transform)
            
            # Read at native resolution (5m)
            dtm_coarse = src.read(1, window=window)
            nodata = src.nodata
            
            if nodata is not None:
                dtm_coarse = np.where(
                    dtm_coarse == nodata, np.nan, 
                    dtm_coarse.astype(np.float32)
                )
        
        print(f"   Bluesky native shape: {dtm_coarse.shape} ({dtm_coarse.shape[0] * 5:.0f}m × {dtm_coarse.shape[1] * 5:.0f}m)")
        print(f"   Valid pixels: {np.sum(~np.isnan(dtm_coarse)):,} / {dtm_coarse.size:,}")
        
        if np.all(np.isnan(dtm_coarse)):
            print("   ⚠️  No Bluesky data for this area!")
            return None
        
        # Apply datum correction BEFORE resampling
        # (Add 58m to convert Malin Head → WGS84 ellipsoid)
        dtm_coarse_corrected = dtm_coarse + GEOID_OFFSET
        print(f"   After +{GEOID_OFFSET}m datum correction: {np.nanmean(dtm_coarse_corrected):.2f}m mean")
        
        # Resample to target resolution
        dtm_fine = _resample_array(dtm_coarse_corrected, target_shape, method)
        
        print(f"   Resampled shape: {dtm_fine.shape} (0.061m resolution)")
        
        return dtm_fine


def _resample_array(coarse, target_shape, method='bicubic'):
    """
    Resample a coarse array to target shape
    
    Handles NaN values properly during interpolation
    """
    
    coarse_h, coarse_w = coarse.shape
    target_h, target_w = target_shape
    
    if method == 'bilinear':
        # Fast - scipy zoom with linear interpolation
        # Handle NaN by filling with mean before zoom, mask after
        nan_mask = np.isnan(coarse)
        if nan_mask.any():
            fill_value = np.nanmean(coarse)
            coarse_filled = np.where(nan_mask, fill_value, coarse)
        else:
            coarse_filled = coarse
        
        zoom_factors = (target_h / coarse_h, target_w / coarse_w)
        fine = zoom(coarse_filled, zoom_factors, order=1)  # order=1 = bilinear
        
        # Propagate NaN mask (zoom the nan mask and threshold)
        if nan_mask.any():
            nan_zoomed = zoom(nan_mask.astype(float), zoom_factors, order=1)
            fine = np.where(nan_zoomed > 0.5, np.nan, fine)
    
    elif method == 'bicubic':
        # Smoother - better preserves terrain curvature
        nan_mask = np.isnan(coarse)
        if nan_mask.any():
            fill_value = np.nanmean(coarse)
            coarse_filled = np.where(nan_mask, fill_value, coarse)
        else:
            coarse_filled = coarse
        
        zoom_factors = (target_h / coarse_h, target_w / coarse_w)
        fine = zoom(coarse_filled, zoom_factors, order=3)  # order=3 = bicubic
        
        if nan_mask.any():
            nan_zoomed = zoom(nan_mask.astype(float), zoom_factors, order=1)
            fine = np.where(nan_zoomed > 0.5, np.nan, fine)
    
    elif method == 'spline':
        # Best quality - fits smooth spline through Bluesky points
        # Ideal for smooth riverbed terrain
        nan_mask = np.isnan(coarse)
        
        # Create coordinate grids
        y_coarse = np.linspace(0, 1, coarse_h)
        x_coarse = np.linspace(0, 1, coarse_w)
        y_fine = np.linspace(0, 1, target_h)
        x_fine = np.linspace(0, 1, target_w)
        
        if nan_mask.any():
            # Fill NaN with interpolated values for spline fitting
            from scipy.interpolate import griddata
            valid_coords = np.argwhere(~nan_mask)
            valid_values = coarse[~nan_mask]
            all_coords = np.argwhere(np.ones_like(coarse, dtype=bool))
            filled = griddata(valid_coords, valid_values, all_coords, method='linear')
            coarse_filled = filled.reshape(coarse.shape)
        else:
            coarse_filled = coarse
        
        # Fit 2D spline
        spline = RectBivariateSpline(y_coarse, x_coarse, coarse_filled, kx=3, ky=3)
        fine = spline(y_fine, x_fine)
        
        # Propagate NaN
        if nan_mask.any():
            nan_zoomed = zoom(nan_mask.astype(float), 
                            (target_h / coarse_h, target_w / coarse_w), order=1)
            fine = np.where(nan_zoomed > 0.5, np.nan, fine)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'bilinear', 'bicubic', or 'spline'")
    
    return fine.astype(np.float32)


def _read_and_reproject_bluesky(bluesky_dtm_file, bounds, target_shape):
    """
    Read and reproject Bluesky DTM to EPSG:32629 for the tile bounds
    Uses rasterio's warp for proper reprojection
    """
    
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds as transform_from_bounds
    
    left, bottom, right, top = bounds
    target_h, target_w = target_shape
    
    dst_crs = CRS.from_epsg(32629)
    dst_transform = transform_from_bounds(left, bottom, right, top, 
                                          target_w, target_h)
    
    with rasterio.open(bluesky_dtm_file) as src:
        dtm_reprojected = np.zeros((target_h, target_w), dtype=np.float32)
        
        reproject(
            source=rasterio.band(src, 1),
            destination=dtm_reprojected,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )
    
    # Convert nodata to NaN
    dtm_reprojected = np.where(dtm_reprojected == 0, np.nan, dtm_reprojected)
    
    return dtm_reprojected

# =======================================================================
# MAIN EXTRACTION FUNCTION
# =======================================================================

def extract_depth_bluesky_dtm(mask_path, tile_position, orthophoto_file,
                               webodm_dsm_file, bluesky_dtm_file,
                               tile_size=1024, resample_method='bicubic'):
    """
    Extract water depths using:
      - WebODM DSM (0.061m) for water surface
      - Bluesky DTM (5m, resampled) for riverbed
    
    Returns dict with depth statistics
    """
    
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask = (mask > 127).astype(np.uint8)
    
    if np.sum(mask) == 0:
        return None  # No water
    
    row_start, col_start = tile_position
    
    # Get world bounds from orthophoto
    with rasterio.open(orthophoto_file) as src:
        ortho_transform = src.transform
        ortho_crs = src.crs
        pixel_size = abs(ortho_transform.a)
        
        left, top = rasterio.transform.xy(ortho_transform, row_start, col_start, offset='ul')
        right, bottom = rasterio.transform.xy(ortho_transform, 
                                              row_start + tile_size, 
                                              col_start + tile_size, 
                                              offset='ul')
    
    bounds = (left, bottom, right, top)
    
    # Water pixel coordinates
    water_coords = np.argwhere(mask > 0)
    rows, cols = water_coords[:, 0], water_coords[:, 1]
    
    # -----------------------------------------------------------------------
    # 1. WebODM DSM → water surface (0.061m resolution)
    # -----------------------------------------------------------------------
    
    print(f"\n  Sampling WebODM DSM (water surface)...")
    
    with rasterio.open(webodm_dsm_file) as src:
        window = window_from_bounds(bounds[0], bounds[1], bounds[2], bounds[3],
                                   transform=src.transform)
        dsm_tile = src.read(1, window=window)
        dsm_nodata = src.nodata
        
        if dsm_nodata is not None:
            dsm_tile = np.where(dsm_tile == dsm_nodata, np.nan, 
                               dsm_tile.astype(np.float32))
        
        if dsm_tile.shape != (tile_size, tile_size):
            zoom_f = (tile_size / dsm_tile.shape[0], tile_size / dsm_tile.shape[1])
            dsm_tile = zoom(dsm_tile, zoom_f, order=1)
        
        water_surface = dsm_tile[rows, cols]
    
    valid_dsm = ~np.isnan(water_surface)
    print(f"     Valid: {np.sum(valid_dsm):,} / {len(water_surface):,}")
    if np.sum(valid_dsm) > 0:
        print(f"     Mean surface: {np.nanmean(water_surface):.2f}m (WGS84 ellipsoid)")
    
    # -----------------------------------------------------------------------
    # 2. Bluesky DTM → riverbed (5m → 0.061m resampled, +58m corrected)
    # -----------------------------------------------------------------------
    
    print(f"\n  Resampling Bluesky DTM (riverbed)...")
    
    dtm_fine = resample_bluesky_dtm_to_tile(
        bluesky_dtm_file, bounds, (tile_size, tile_size),
        method=resample_method
    )
    
    if dtm_fine is None:
        print(f"  ⚠️  No Bluesky data for this tile!")
        return None
    
    riverbed = dtm_fine[rows, cols]
    valid_dtm = ~np.isnan(riverbed)
    
    print(f"     Valid: {np.sum(valid_dtm):,} / {len(riverbed):,}")
    if np.sum(valid_dtm) > 0:
        print(f"     Mean riverbed: {np.nanmean(riverbed):.2f}m (WGS84 ellipsoid, corrected)")
    
    # -----------------------------------------------------------------------
    # 3. Calculate depth
    # -----------------------------------------------------------------------
    
    depth = water_surface - riverbed
    
    # Filter valid depths (physical bounds)
    valid_mask = valid_dsm & valid_dtm & (depth > 0) & (depth < 5)
    
    if np.sum(valid_mask) == 0:
        return None
    
    depths = depth[valid_mask]
    
    print(f"\n  Depths:")
    print(f"     Valid: {np.sum(valid_mask):,}")
    print(f"     Mean: {np.mean(depths):.3f}m")
    print(f"     Median: {np.median(depths):.3f}m")
    print(f"     Range: {np.min(depths):.3f} to {np.max(depths):.3f}m")
    
    return {
        'mask': mask,
        'water_coords': water_coords,
        'dsm_tile': dsm_tile,
        'dtm_tile': dtm_fine,
        'water_surface': water_surface,
        'riverbed': riverbed,
        'depth': depth,
        'valid_mask': valid_mask,
        'bounds': bounds,
        'pixel_size': pixel_size,
        'geoid_offset_applied': GEOID_OFFSET,
        'resample_method': resample_method
    }

# =======================================================================
# COMPARE: WebODM DTM vs Bluesky DTM
# =======================================================================

def compare_dtm_sources(mask_path, tile_position, orthophoto_file,
                         webodm_dsm_file, webodm_dtm_file, bluesky_dtm_file,
                         tile_size=1024, output_file='dtm_comparison.png'):
    """
    Side-by-side comparison of depths using:
    1. WebODM DTM (cannot penetrate canopy)
    2. Bluesky DTM (penetrates canopy, but 5m resolution)
    
    Shows clearly why Bluesky gives better results
    """
    
    print("="*70)
    print("COMPARING DTM SOURCES")
    print("="*70)
    
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 127).astype(np.uint8)
    water_coords = np.argwhere(mask > 0)
    rows, cols = water_coords[:, 0], water_coords[:, 1]
    
    # Get bounds
    row_start, col_start = tile_position
    with rasterio.open(orthophoto_file) as src:
        left, top = rasterio.transform.xy(src.transform, row_start, col_start, offset='ul')
        right, bottom = rasterio.transform.xy(src.transform, 
                                              row_start + tile_size, 
                                              col_start + tile_size, 
                                              offset='ul')
    bounds = (left, bottom, right, top)
    
    # Read WebODM DSM
    with rasterio.open(webodm_dsm_file) as src:
        window = window_from_bounds(*bounds, transform=src.transform)
        dsm_tile = src.read(1, window=window).astype(np.float32)
        if src.nodata:
            dsm_tile = np.where(dsm_tile == src.nodata, np.nan, dsm_tile)
        if dsm_tile.shape != (tile_size, tile_size):
            dsm_tile = zoom(dsm_tile, 
                          (tile_size/dsm_tile.shape[0], tile_size/dsm_tile.shape[1]), 
                          order=1)
    
    # Read WebODM DTM
    with rasterio.open(webodm_dtm_file) as src:
        window = window_from_bounds(*bounds, transform=src.transform)
        webodm_dtm_tile = src.read(1, window=window).astype(np.float32)
        if src.nodata:
            webodm_dtm_tile = np.where(webodm_dtm_tile == src.nodata, np.nan, webodm_dtm_tile)
        if webodm_dtm_tile.shape != (tile_size, tile_size):
            webodm_dtm_tile = zoom(webodm_dtm_tile,
                                  (tile_size/webodm_dtm_tile.shape[0], 
                                   tile_size/webodm_dtm_tile.shape[1]),
                                  order=1)
    
    # Read Bluesky DTM (resampled + corrected)
    bluesky_dtm_tile = resample_bluesky_dtm_to_tile(
        bluesky_dtm_file, bounds, (tile_size, tile_size), method='bicubic'
    )
    
    # Calculate depths
    depth_webodm = dsm_tile - webodm_dtm_tile
    depth_bluesky = dsm_tile - bluesky_dtm_tile
    
    # Filter valid
    valid_w = ~np.isnan(depth_webodm) & (depth_webodm > 0) & (depth_webodm < 10)
    valid_b = ~np.isnan(depth_bluesky) & (depth_bluesky > 0) & (depth_bluesky < 5)
    water_valid_w = valid_w[rows, cols]
    water_valid_b = valid_b[rows, cols]
    
    # -----------------------------------------------------------------------
    # PLOT COMPARISON
    # -----------------------------------------------------------------------
    
    fig, axes = plt.subplots(2, 4, figsize=(22, 12))
    
    # Row 1: DTM comparison
    im1 = axes[0, 0].imshow(webodm_dtm_tile, cmap='terrain')
    axes[0, 0].set_title('WebODM DTM\n(Cannot penetrate canopy)', fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], label='m (WGS84)', fraction=0.046)
    
    im2 = axes[0, 1].imshow(bluesky_dtm_tile, cmap='terrain')
    axes[0, 1].set_title(f'Bluesky DTM (5m→0.061m)\n(Penetrates canopy, +{GEOID_OFFSET}m corrected)', fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], label='m (WGS84)', fraction=0.046)
    
    # DTM difference
    dtm_diff = bluesky_dtm_tile - webodm_dtm_tile
    im3 = axes[0, 2].imshow(dtm_diff, cmap='RdBu_r', 
                            vmin=-np.nanstd(dtm_diff)*2, 
                            vmax=np.nanstd(dtm_diff)*2)
    axes[0, 2].set_title('DTM Difference\n(Bluesky - WebODM)', fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], label='m', fraction=0.046)
    
    # Elevation profile text
    axes[0, 3].axis('off')
    txt = (
        f"ELEVATION STATISTICS\n"
        f"{'─'*32}\n\n"
        f"WebODM DTM (riverbed):\n"
        f"  Mean: {np.nanmean(webodm_dtm_tile):.2f}m\n"
        f"  Std:  {np.nanstd(webodm_dtm_tile):.2f}m\n\n"
        f"Bluesky DTM (riverbed):\n"
        f"  Mean: {np.nanmean(bluesky_dtm_tile):.2f}m\n"
        f"  Std:  {np.nanstd(bluesky_dtm_tile):.2f}m\n\n"
        f"DTM Difference:\n"
        f"  Mean: {np.nanmean(dtm_diff):.2f}m\n"
        f"  Std:  {np.nanstd(dtm_diff):.2f}m\n\n"
        f"Geoid offset applied:\n"
        f"  +{GEOID_OFFSET:.1f}m to Bluesky"
    )
    axes[0, 3].text(0.05, 0.95, txt, transform=axes[0, 3].transAxes,
                   fontfamily='monospace', fontsize=10, va='top')
    
    # Row 2: Depth comparison
    depth_map_w = np.full(mask.shape, np.nan)
    depth_map_b = np.full(mask.shape, np.nan)
    
    depth_w_vals = depth_webodm[rows, cols]
    depth_b_vals = depth_bluesky[rows, cols]
    depth_map_w[rows[water_valid_w], cols[water_valid_w]] = depth_w_vals[water_valid_w]
    depth_map_b[rows[water_valid_b], cols[water_valid_b]] = depth_b_vals[water_valid_b]
    
    vmax = max(np.nanmax(depth_map_w) if np.sum(water_valid_w) > 0 else 3,
               np.nanmax(depth_map_b) if np.sum(water_valid_b) > 0 else 3)
    
    im4 = axes[1, 0].imshow(depth_map_w, cmap='YlGnBu', vmin=0, vmax=vmax)
    axes[1, 0].set_title('Depth: WebODM DSM - WebODM DTM\n(OVERESTIMATES - no canopy penetration)', fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], label='Depth (m)', fraction=0.046)
    
    im5 = axes[1, 1].imshow(depth_map_b, cmap='YlGnBu', vmin=0, vmax=vmax)
    axes[1, 1].set_title('Depth: WebODM DSM - Bluesky DTM\n(BETTER - canopy-penetrating riverbed)', fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1], label='Depth (m)', fraction=0.046)
    
    # Depth difference
    depth_diff_map = depth_map_w - depth_map_b
    im6 = axes[1, 2].imshow(depth_diff_map, cmap='RdYlGn_r')
    axes[1, 2].set_title('Depth Overestimation\n(WebODM - Bluesky)', fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2], label='Overestimation (m)', fraction=0.046)
    
    # Histogram comparison
    if np.sum(water_valid_w) > 0 and np.sum(water_valid_b) > 0:
        axes[1, 3].hist(depth_w_vals[water_valid_w], bins=40, alpha=0.6, 
                       color='red', label='WebODM DTM (overestimates)', 
                       density=True)
        axes[1, 3].hist(depth_b_vals[water_valid_b], bins=40, alpha=0.6,
                       color='blue', label='Bluesky DTM (better)',
                       density=True)
        axes[1, 3].axvline(1.0, color='black', linestyle='--', 
                          label='~Expected max depth')
        
        axes[1, 3].set_xlabel('Water Depth (m)', fontweight='bold')
        axes[1, 3].set_ylabel('Density', fontweight='bold')
        axes[1, 3].set_title('Depth Distribution Comparison', fontweight='bold')
        axes[1, 3].legend(fontsize=9)
        axes[1, 3].grid(alpha=0.3)
        
        mean_w = np.mean(depth_w_vals[water_valid_w])
        mean_b = np.mean(depth_b_vals[water_valid_b])
        axes[1, 3].text(0.98, 0.95, 
                       f'WebODM mean: {mean_w:.2f}m\nBluesky mean: {mean_b:.2f}m\nImprovement: {mean_w-mean_b:.2f}m',
                       transform=axes[1, 3].transAxes,
                       ha='right', va='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.suptitle(f'DTM Source Comparison for Water Depth\n'
                f'Tile position: {tile_position}  |  Bluesky geoid correction: +{GEOID_OFFSET}m',
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison saved: {output_file}")
    plt.close()

# =======================================================================
# MAIN
# =======================================================================

if __name__ == "__main__":
    
    from rasterio.transform import from_bounds
    
    # Your files
    mask_image = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\masks\DJI_20250728101825_0553_V_patch_0359.png"
    webodm_dtm_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_dem\dtm_aligned.tif"
    
    # Tile parameters
    tile_number = 359
    tile_size = 1024
    stride = 768
    
    with rasterio.open(orthophoto_file) as src:
        ortho_w = src.width
    
    n_cols = (ortho_w - tile_size) // stride + 1
    row_start = (tile_number // n_cols) * stride
    col_start = (tile_number % n_cols) * stride
    tile_position = (row_start, col_start)
    
    print("="*70)
    print("WATER DEPTH: WebODM DSM + Bluesky DTM")
    print("="*70)
    print(f"\nGeoid offset: +{GEOID_OFFSET}m (Malin Head → WGS84 ellipsoid)")
    print(f"Resample: 5m → 0.061m (bicubic spline)")
    
    # Option 1: Extract depths only
    print("\n--- EXTRACTING DEPTHS ---")
    results = extract_depth_bluesky_dtm(
        mask_path=mask_image,
        tile_position=tile_position,
        orthophoto_file=orthophoto_file,
        webodm_dsm_file=webodm_dsm_file,
        bluesky_dtm_file=bluesky_dtm_file,
        tile_size=tile_size,
        resample_method='bicubic'  # or 'bilinear', 'spline'
    )
    
    if results and np.sum(results['valid_mask']) > 0:
        depths = results['depth'][results['valid_mask']]
        print(f"\n{'='*70}")
        print(f"RESULTS (WebODM DSM + Bluesky DTM, bicubic resampled)")
        print(f"{'='*70}")
        print(f"  Mean depth:   {np.mean(depths):.3f}m")
        print(f"  Median depth: {np.median(depths):.3f}m")
        print(f"  Std:          {np.std(depths):.3f}m")
        print(f"  Range:        {np.min(depths):.3f} to {np.max(depths):.3f}m")
    
    # Option 2: Side-by-side comparison with WebODM DTM
    print("\n--- GENERATING COMPARISON ---")
    compare_dtm_sources(
        mask_path=mask_image,
        tile_position=tile_position,
        orthophoto_file=orthophoto_file,
        webodm_dsm_file=webodm_dsm_file,
        webodm_dtm_file=webodm_dtm_file,
        bluesky_dtm_file=bluesky_dtm_file,
        output_file='dtm_comparison.png'
    )
    
    print("\n✅ Done! Check dtm_comparison.png")
