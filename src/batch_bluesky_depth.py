"""
STEP 2: Batch Water Depth Extraction using Bluesky DTM

Processes ALL tiles in working_tiles.txt using:
  - WebODM DSM (water surface)
  - Pre-resampled Bluesky DTM (riverbed, canopy-penetrating)

Run save_bluesky_resampled.py FIRST to create the resampled DTM file.
Run find_valid_from_files.py FIRST to create working_tiles.txt.
"""

import numpy as np
import rasterio
from rasterio.windows import from_bounds as window_from_bounds
import cv2
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# =======================================================================
# YOUR FILES
# =======================================================================

images_dir = r"C:\Users\AdikariAdikari\PycharmProjects\river_segmentation\dataset\processed_1024\train\images"
masks_dir = r"C:\Users\AdikariAdikari\PycharmProjects\river_segmentation\dataset\processed_1024\train\masks"
orthophoto_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_orthophoto\odm_orthophoto.tif"
webodm_dsm_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_dem\dsm_aligned.tif"

# Pre-resampled Bluesky DTM (created by save_bluesky_resampled.py)
# Already at 0.061m resolution + geoid corrected (+58m applied)
bluesky_resampled_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\kriged\fusion_based_kriged_0061m.tif"

working_tiles_file = "working_tiles.txt"
output_dir = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\results_bluesky_kriged"

tile_size = 1024
stride    = 768

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)

# =======================================================================
# LOAD WORKING TILES
# =======================================================================

def load_working_tiles(filepath):
    """Parse working_tiles.txt - reads exact filenames"""

    tiles = []
    current = {}

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Tile ') and ':' in line:
                if current.get('image_file'):
                    tiles.append(current)
                tile_num = int(line.split()[1].rstrip(':'))
                current = {'tile_num': tile_num, 'image_file': None, 'mask_file': None}
            elif line.startswith('Image:'):
                current['image_file'] = line.split('Image:', 1)[1].strip()
            elif line.startswith('Mask:'):
                current['mask_file'] = line.split('Mask:', 1)[1].strip()

    if current.get('image_file'):
        tiles.append(current)

    return tiles

# =======================================================================
# TILE POSITION
# =======================================================================

def tile_position_from_number(tile_number, stride, ortho_width):
    n_cols = (ortho_width - tile_size) // stride + 1
    row_idx = tile_number // n_cols
    col_idx = tile_number % n_cols
    return (row_idx * stride, col_idx * stride), (row_idx, col_idx)

# =======================================================================
# SINGLE TILE EXTRACTION
# =======================================================================

def extract_depth_single_tile(mask_path, tile_position, ortho_transform):
    """
    Extract water depths for one tile using pre-resampled Bluesky DTM.

    Both DSM and Bluesky DTM are now the same resolution (0.061m),
    so we just read the matching window from each - no per-tile resampling!
    """

    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None, "mask_not_found"

    mask = (mask > 127).astype(np.uint8)

    if np.sum(mask) == 0:
        return None, "no_water"

    row_start, col_start = tile_position

    # World bounds of this tile
    left, top   = rasterio.transform.xy(ortho_transform, row_start, col_start, offset='ul')
    right, bottom = rasterio.transform.xy(ortho_transform,
                                          row_start + tile_size,
                                          col_start + tile_size, offset='ul')
    bounds = (left, bottom, right, top)

    water_coords = np.argwhere(mask > 0)
    rows, cols   = water_coords[:, 0], water_coords[:, 1]

    # ------------------------------------------------------------------
    # Read WebODM DSM  (water surface)
    # ------------------------------------------------------------------
    with rasterio.open(webodm_dsm_file) as src:
        window  = window_from_bounds(*bounds, transform=src.transform)
        dsm_tile = src.read(1, window=window).astype(np.float32)
        if src.nodata is not None:
            dsm_tile = np.where(dsm_tile == src.nodata, np.nan, dsm_tile)

        # Resize to tile_size if DSM resolution differs slightly
        if dsm_tile.shape != (tile_size, tile_size):
            from scipy.ndimage import zoom as scipy_zoom
            dsm_tile = scipy_zoom(dsm_tile,
                                  (tile_size / dsm_tile.shape[0],
                                   tile_size / dsm_tile.shape[1]), order=1)

    # ------------------------------------------------------------------
    # Read pre-resampled Bluesky DTM  (riverbed, already 0.061m + geoid corrected)
    # Same resolution as orthophoto → just read exact window, no resampling!
    # ------------------------------------------------------------------
    with rasterio.open(bluesky_resampled_file) as src:
        window   = window_from_bounds(*bounds, transform=src.transform)
        dtm_tile = src.read(1, window=window).astype(np.float32)
        if src.nodata is not None:
            dtm_tile = np.where(dtm_tile == src.nodata, np.nan, dtm_tile)

        if dtm_tile.shape != (tile_size, tile_size):
            from scipy.ndimage import zoom as scipy_zoom
            dtm_tile = scipy_zoom(dtm_tile,
                                  (tile_size / dtm_tile.shape[0],
                                   tile_size / dtm_tile.shape[1]), order=1)

    # ------------------------------------------------------------------
    # Calculate depth
    # ------------------------------------------------------------------
    water_surface = dsm_tile[rows, cols]
    riverbed      = dtm_tile[rows, cols]

    depth = water_surface - riverbed

    valid_mask = (
        ~np.isnan(water_surface) &
        ~np.isnan(riverbed) &
        (depth > 0.0) &
        (depth < 5.0)    # Physical upper bound for this river
    )

    if np.sum(valid_mask) == 0:
        return None, "no_valid_depths"

    return {
        'mask':          mask,
        'water_coords':  water_coords,
        'dsm_tile':      dsm_tile,
        'dtm_tile':      dtm_tile,
        'water_surface': water_surface,
        'riverbed':      riverbed,
        'depth':         depth,
        'valid_mask':    valid_mask,
        'bounds':        bounds,
    }, "ok"

# =======================================================================
# VISUALIZATION
# =======================================================================

def visualize_tile(tile_num, image_path, results, output_path):

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None \
          else np.zeros((tile_size, tile_size, 3), dtype=np.uint8)

    # 1. Image + mask overlay
    overlay = img.copy()
    colour  = np.zeros_like(overlay)
    colour[results['mask'] > 0] = [0, 255, 255]
    overlay = cv2.addWeighted(overlay, 0.7, colour, 0.3, 0)
    axes[0, 0].imshow(overlay);  axes[0, 0].set_title('Image + Water Mask', fontweight='bold'); axes[0, 0].axis('off')

    # 2. DSM
    im2 = axes[0, 1].imshow(results['dsm_tile'], cmap='terrain')
    axes[0, 1].set_title('WebODM DSM (Water Surface)', fontweight='bold'); axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], label='m (WGS84)', fraction=0.046)

    # 3. Bluesky DTM
    im3 = axes[0, 2].imshow(results['dtm_tile'], cmap='terrain')
    axes[0, 2].set_title('Bluesky DTM (Riverbed, +58m corrected)', fontweight='bold'); axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], label='m (WGS84)', fraction=0.046)

    # 4. Depth map
    depth_map = np.full(results['mask'].shape, np.nan)
    wc = results['water_coords']
    r, c = wc[:, 0], wc[:, 1]
    v = results['valid_mask']
    depth_map[r[v], c[v]] = results['depth'][v]
    im4 = axes[1, 0].imshow(depth_map, cmap='YlGnBu', vmin=0, vmax=2)
    axes[1, 0].set_title('Water Depth (m)', fontweight='bold'); axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], label='Depth (m)', fraction=0.046)

    # 5. Histogram
    depths = results['depth'][v]
    axes[1, 1].hist(depths, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1, 1].axvline(np.mean(depths), color='red',  linestyle='--', label=f'Mean {np.mean(depths):.2f}m')
    axes[1, 1].axvline(np.median(depths), color='orange', linestyle='--', label=f'Median {np.median(depths):.2f}m')
    axes[1, 1].set_xlabel('Depth (m)', fontweight='bold')
    axes[1, 1].set_ylabel('Pixel Count', fontweight='bold')
    axes[1, 1].set_title('Depth Distribution', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    # 6. Stats text
    axes[1, 2].axis('off')
    txt = (
        f"TILE {tile_num} STATISTICS\n"
        f"{'─'*30}\n\n"
        f"Water pixels:  {np.sum(results['mask']):,}\n"
        f"Valid depths:  {np.sum(v):,}\n"
        f"Coverage:      {np.sum(v)/np.sum(results['mask'])*100:.1f}%\n\n"
        f"Depth (m):\n"
        f"  Mean:    {np.mean(depths):.3f}\n"
        f"  Median:  {np.median(depths):.3f}\n"
        f"  Std:     {np.std(depths):.3f}\n"
        f"  Min:     {np.min(depths):.3f}\n"
        f"  Max:     {np.max(depths):.3f}\n\n"
        f"Water surface: {np.nanmean(results['water_surface'][v]):.2f}m\n"
        f"Riverbed:      {np.nanmean(results['riverbed'][v]):.2f}m\n\n"
        f"DTM source: Bluesky 5m (bicubic)\n"
        f"Geoid corr: +58.0m applied"
    )
    axes[1, 2].text(0.05, 0.95, txt, transform=axes[1, 2].transAxes,
                   fontfamily='monospace', fontsize=10, va='top')

    plt.suptitle(f'Water Depth Analysis — Tile {tile_num} (Bluesky DTM)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

# =======================================================================
# SUMMARY PLOTS
# =======================================================================

def create_summary_plots(df, output_dir):

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].hist(df['depth_mean'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].set_xlabel('Mean Depth (m)', fontweight='bold')
    axes[0, 0].set_ylabel('Tile Count', fontweight='bold')
    axes[0, 0].set_title('Distribution of Mean Depths', fontweight='bold')
    axes[0, 0].axvline(df['depth_mean'].mean(), color='red', linestyle='--',
                      label=f"Overall mean: {df['depth_mean'].mean():.2f}m")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].hist(df['coverage_pct'], bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Valid Depth Coverage (%)', fontweight='bold')
    axes[0, 1].set_ylabel('Tile Count', fontweight='bold')
    axes[0, 1].set_title('Data Quality: Coverage per Tile', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    sc = axes[0, 2].scatter(df['grid_col'], df['grid_row'],
                            c=df['depth_mean'], cmap='YlGnBu', s=60, vmin=0)
    axes[0, 2].set_xlabel('Grid Column', fontweight='bold')
    axes[0, 2].set_ylabel('Grid Row', fontweight='bold')
    axes[0, 2].set_title('Spatial Mean Depth Distribution', fontweight='bold')
    axes[0, 2].invert_yaxis()
    plt.colorbar(sc, ax=axes[0, 2], label='Mean Depth (m)')

    axes[1, 0].hist(df['surface_mean'], bins=30, edgecolor='black', alpha=0.7, color='navy')
    axes[1, 0].set_xlabel('Water Surface Elevation (m)', fontweight='bold')
    axes[1, 0].set_ylabel('Tile Count', fontweight='bold')
    axes[1, 0].set_title('Water Surface Elevation (WGS84)', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].hist(df['riverbed_mean'], bins=30, edgecolor='black', alpha=0.7, color='saddlebrown')
    axes[1, 1].set_xlabel('Riverbed Elevation (m)', fontweight='bold')
    axes[1, 1].set_ylabel('Tile Count', fontweight='bold')
    axes[1, 1].set_title('Riverbed Elevation - Bluesky DTM (WGS84)', fontweight='bold')
    axes[1, 1].grid(alpha=0.3)

    axes[1, 2].axis('off')
    summary_txt = (
        f"OVERALL SUMMARY\n"
        f"{'─'*35}\n\n"
        f"Tiles processed:   {len(df)}\n\n"
        f"Water Depth (m):\n"
        f"  Mean:    {df['depth_mean'].mean():.3f} ± {df['depth_mean'].std():.3f}\n"
        f"  Median:  {df['depth_median'].median():.3f}\n"
        f"  Range:   {df['depth_min'].min():.3f} to {df['depth_max'].max():.3f}\n\n"
        f"Elevations (WGS84 ellipsoid):\n"
        f"  Surface: {df['surface_mean'].mean():.2f} ± {df['surface_mean'].std():.2f}m\n"
        f"  Riverbed:{df['riverbed_mean'].mean():.2f} ± {df['riverbed_mean'].std():.2f}m\n\n"
        f"Coverage:\n"
        f"  Mean:    {df['coverage_pct'].mean():.1f}%\n"
        f"  Water px:{df['water_pixels'].sum():,}\n"
        f"  Valid dp:{df['valid_depths'].sum():,}\n\n"
        f"DTM source: Bluesky 5m (bicubic)\n"
        f"Geoid corr: +58.0m (Malin Head→WGS84)"
    )
    axes[1, 2].text(0.05, 0.95, summary_txt, transform=axes[1, 2].transAxes,
                   fontfamily='monospace', fontsize=10, va='top')

    plt.suptitle('Batch Water Depth Analysis — Bluesky DTM',
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'summary_plots.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"✓ Summary plots: {path}")
    plt.close()

# =======================================================================
# MAIN BATCH
# =======================================================================

def batch_extract():

    print("="*70)
    print("BATCH WATER DEPTH EXTRACTION — BLUESKY DTM")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Verify pre-resampled Bluesky DTM exists
    if not os.path.exists(bluesky_resampled_file):
        print(f"\n✗ Pre-resampled Bluesky DTM not found:")
        print(f"  {bluesky_resampled_file}")
        print(f"\n  Run save_bluesky_resampled.py first!")
        return

    # Load orthophoto metadata
    with rasterio.open(orthophoto_file) as src:
        ortho_w, ortho_h = src.width, src.height
        ortho_transform  = src.transform

    # Confirm Bluesky file info
    with rasterio.open(bluesky_resampled_file) as src:
        print(f"\nBluesky DTM (pre-resampled):")
        print(f"  Resolution: {abs(src.transform.a):.4f}m")
        print(f"  Size:       {src.height} × {src.width}")
        print(f"  CRS:        {src.crs}")
        print(f"  (Geoid correction +58m already applied)")

    # Load tile list
    if not os.path.exists(working_tiles_file):
        print(f"\n✗ {working_tiles_file} not found!")
        print(f"  Run find_valid_from_files.py first.")
        return

    working_tiles = load_working_tiles(working_tiles_file)
    print(f"\n✓ Loaded {len(working_tiles)} tiles from {working_tiles_file}")

    # Verify first few file paths
    for tile in working_tiles[:3]:
        img_ok  = os.path.exists(os.path.join(images_dir, tile['image_file']))
        mask_ok = os.path.exists(os.path.join(masks_dir,  tile['mask_file']))
        status  = "✓" if img_ok and mask_ok else "✗"
        print(f"  {status} Tile {tile['tile_num']}: {tile['image_file']}")

    print(f"\nOutput: {output_dir}")

    # ------------------------------------------------------------------
    # Process tiles
    # ------------------------------------------------------------------

    all_results = []
    successful = failed = no_water = no_data = 0
    error_log = []

    print(f"\n{'='*70}")
    print(f"PROCESSING {len(working_tiles)} TILES")
    print(f"{'='*70}\n")

    for tile_info in tqdm(working_tiles, desc="Extracting depths"):

        tile_num   = tile_info['tile_num']
        image_path = os.path.join(images_dir, tile_info['image_file'])
        mask_path  = os.path.join(masks_dir,  tile_info['mask_file'])

        if not os.path.exists(image_path):
            error_log.append(f"Tile {tile_num}: image not found")
            failed += 1;  continue

        if not os.path.exists(mask_path):
            error_log.append(f"Tile {tile_num}: mask not found")
            failed += 1;  continue

        tile_position, grid_pos = tile_position_from_number(tile_num, stride, ortho_w)

        try:
            results, status = extract_depth_single_tile(mask_path, tile_position, ortho_transform)

            if status == "no_water":
                no_water += 1;  continue
            if status in ("no_valid_depths", "mask_not_found"):
                no_data += 1;   continue

            v      = results['valid_mask']
            depths = results['depth'][v]

            row = {
                'tile_number':  int(tile_num),
                'grid_row':     int(grid_pos[0]),
                'grid_col':     int(grid_pos[1]),
                'pixel_row':    int(tile_position[0]),
                'pixel_col':    int(tile_position[1]),
                'bounds_left':  float(results['bounds'][0]),
                'bounds_bottom':float(results['bounds'][1]),
                'bounds_right': float(results['bounds'][2]),
                'bounds_top':   float(results['bounds'][3]),
                'water_pixels': int(np.sum(results['mask'])),
                'valid_depths': int(np.sum(v)),
                'coverage_pct': float(np.sum(v) / np.sum(results['mask']) * 100),
                'depth_mean':   float(np.mean(depths)),
                'depth_median': float(np.median(depths)),
                'depth_std':    float(np.std(depths)),
                'depth_min':    float(np.min(depths)),
                'depth_max':    float(np.max(depths)),
                'surface_mean': float(np.nanmean(results['water_surface'][v])),
                'surface_std':  float(np.nanstd(results['water_surface'][v])),
                'riverbed_mean':float(np.nanmean(results['riverbed'][v])),
                'riverbed_std': float(np.nanstd(results['riverbed'][v])),
                'image_file':   tile_info['image_file'],
                'mask_file':    tile_info['mask_file'],
            }
            all_results.append(row)
            successful += 1

            # Visualize every 10th successful tile
            if successful % 10 == 1 or successful <= 3:
                viz_path = os.path.join(output_dir, 'visualizations', f'tile_{tile_num:04d}.png')
                visualize_tile(tile_num, image_path, results, viz_path)

        except Exception as e:
            error_log.append(f"Tile {tile_num}: {type(e).__name__}: {str(e)}")
            failed += 1

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")

    if all_results:
        df = pd.DataFrame(all_results)

        csv_path = os.path.join(output_dir, 'water_depths_bluesky.csv')
        df.to_csv(csv_path, index=False)
        print(f"✓ CSV:  {csv_path}")

        json_path = os.path.join(output_dir, 'water_depths_bluesky.json')
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"✓ JSON: {json_path}")

        summary = {
            'processing_date':    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dtm_source':         'Bluesky 5m (bicubic resampled to 0.061m)',
            'geoid_offset_m':     58.0,
            'total_tiles':        len(working_tiles),
            'successful':         successful,
            'failed':             failed,
            'no_water':           no_water,
            'no_data':            no_data,
            'total_water_pixels': int(df['water_pixels'].sum()),
            'total_valid_depths': int(df['valid_depths'].sum()),
            'depth_mean':         float(df['depth_mean'].mean()),
            'depth_median':       float(df['depth_median'].median()),
            'depth_std':          float(df['depth_mean'].std()),
            'depth_min':          float(df['depth_min'].min()),
            'depth_max':          float(df['depth_max'].max()),
            'surface_mean':       float(df['surface_mean'].mean()),
            'riverbed_mean':      float(df['riverbed_mean'].mean()),
        }

        summary_path = os.path.join(output_dir, 'summary_statistics.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary: {summary_path}")

        create_summary_plots(df, output_dir)

    if error_log:
        print(f"\nFirst 10 errors:")
        for e in error_log[:10]:
            print(f"  {e}")

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"Total tiles:    {len(working_tiles)}")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed:     {failed}")
    print(f"  ⊘ No water:   {no_water}")
    print(f"  ⊘ No data:    {no_data}")

    if all_results:
        print(f"\nDepth statistics (Bluesky DTM):")
        print(f"  Mean:   {summary['depth_mean']:.3f}m")
        print(f"  Median: {summary['depth_median']:.3f}m")
        print(f"  Range:  {summary['depth_min']:.3f} to {summary['depth_max']:.3f}m")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results:  {output_dir}")


if __name__ == "__main__":
    batch_extract()
