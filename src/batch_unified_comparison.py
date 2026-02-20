"""
Unified Batch Water Depth Extraction
Processes BOTH WebODM DTM and Bluesky DTM on IDENTICAL tile sets

This guarantees the summary plots are directly comparable because:
  - Same tiles used for both
  - Same depth validity bounds for both
  - Tile included if DSM has data (regardless of which DTM has data)
  - NaN stored when a DTM has no data for a tile (not skipped)

Outputs:
  - water_depths_webodm.csv
  - water_depths_bluesky.csv
  - comparison_plots.png   ← side-by-side on same tiles
  - per_tile_comparison/   ← individual tiles showing both DTMs
"""

import numpy as np
import rasterio
from rasterio.windows import from_bounds as window_from_bounds
from scipy.ndimage import zoom as scipy_zoom
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# =======================================================================
# FILES
# =======================================================================

images_dir      = r"C:\Users\AdikariAdikari\PycharmProjects\river_segmentation\dataset\processed_1024\train\images"
masks_dir       = r"C:\Users\AdikariAdikari\PycharmProjects\river_segmentation\dataset\processed_1024\train\masks"
orthophoto_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_orthophoto\odm_orthophoto.tif"
webodm_dsm_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_dem\dsm_aligned.tif"
webodm_dtm_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_dem\dtm_aligned.tif"

# Pre-resampled Bluesky DTM (from save_bluesky_resampled.py)
# Already at 0.061m + geoid corrected (+58m applied)
bluesky_resampled_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\5m_dtm\bluesky_dtm_resampled_0061m.tif"

working_tiles_file = "working_tiles.txt"
output_dir = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\results_comparison"

tile_size = 1024
stride    = 768

# Depth validity bounds — SAME for both DTM sources
DEPTH_MIN = 0.0   # metres
DEPTH_MAX = 5.0   # metres (physical upper bound for this river)

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'per_tile_comparison'), exist_ok=True)

# =======================================================================
# HELPERS
# =======================================================================

def load_working_tiles(filepath):
    """Parse working_tiles.txt — reads exact filenames"""
    tiles = []
    current = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('Tile ') and ':' in line:
                if current.get('image_file'):
                    tiles.append(current)
                current = {
                    'tile_num': int(line.split()[1].rstrip(':')),
                    'image_file': None, 'mask_file': None
                }
            elif line.startswith('Image:'):
                current['image_file'] = line.split('Image:', 1)[1].strip()
            elif line.startswith('Mask:'):
                current['mask_file'] = line.split('Mask:', 1)[1].strip()
    if current.get('image_file'):
        tiles.append(current)
    return tiles


def tile_position(tile_num, stride, ortho_width):
    n_cols  = (ortho_width - tile_size) // stride + 1
    row_idx = tile_num // n_cols
    col_idx = tile_num % n_cols
    return (row_idx * stride, col_idx * stride), (row_idx, col_idx)


def read_raster_window(raster_path, bounds, fallback_shape):
    """
    Read a raster within world bounds.
    Returns (array, had_data: bool)
    Array is always fallback_shape, NaN where nodata.
    """
    with rasterio.open(raster_path) as src:
        window = window_from_bounds(*bounds, transform=src.transform)
        data   = src.read(1, window=window).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)

    had_data = not np.all(np.isnan(data))

    # Resize to tile_size if resolution differs
    if data.shape != fallback_shape:
        data = scipy_zoom(data,
                          (fallback_shape[0] / data.shape[0],
                           fallback_shape[1] / data.shape[1]),
                          order=3)   # bicubic

    return data, had_data


def compute_depth_stats(depth, valid_mask):
    """Return dict of stats for valid depth pixels. All NaN if no valid."""
    if np.sum(valid_mask) == 0:
        return {k: np.nan for k in
                ['n', 'mean', 'median', 'std', 'min', 'max']}
    d = depth[valid_mask]
    return {
        'n':      int(np.sum(valid_mask)),
        'mean':   float(np.mean(d)),
        'median': float(np.median(d)),
        'std':    float(np.std(d)),
        'min':    float(np.min(d)),
        'max':    float(np.max(d)),
    }

# =======================================================================
# SINGLE TILE — BOTH DTMs
# =======================================================================

def process_tile(mask_path, image_path, tile_pos, ortho_transform):
    """
    Extract depths for one tile using both WebODM DTM and Bluesky DTM.

    Returns dict with results for both sources, or None if no water / no DSM.
    """

    # --- mask ---
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None, "mask_not_found"
    mask = (mask > 127).astype(np.uint8)
    if np.sum(mask) == 0:
        return None, "no_water"

    row_start, col_start = tile_pos
    left, top   = rasterio.transform.xy(ortho_transform, row_start, col_start, offset='ul')
    right, bottom = rasterio.transform.xy(ortho_transform,
                                          row_start + tile_size,
                                          col_start + tile_size, offset='ul')
    bounds = (left, bottom, right, top)
    shape  = (tile_size, tile_size)

    wc   = np.argwhere(mask > 0)
    rows, cols = wc[:, 0], wc[:, 1]

    # --- DSM (shared for both) ---
    dsm_tile, dsm_ok = read_raster_window(webodm_dsm_file, bounds, shape)
    if not dsm_ok:
        return None, "no_dsm"

    water_surface = dsm_tile[rows, cols]

    # --- WebODM DTM ---
    webodm_dtm_tile, _ = read_raster_window(webodm_dtm_file, bounds, shape)
    riverbed_w          = webodm_dtm_tile[rows, cols]
    depth_w             = water_surface - riverbed_w
    valid_w             = (~np.isnan(water_surface) & ~np.isnan(riverbed_w) &
                           (depth_w > DEPTH_MIN) & (depth_w < DEPTH_MAX))

    # --- Bluesky DTM ---
    bluesky_dtm_tile, _ = read_raster_window(bluesky_resampled_file, bounds, shape)
    riverbed_b           = bluesky_dtm_tile[rows, cols]
    depth_b              = water_surface - riverbed_b
    valid_b              = (~np.isnan(water_surface) & ~np.isnan(riverbed_b) &
                            (depth_b > DEPTH_MIN) & (depth_b < DEPTH_MAX))

    return {
        'mask':            mask,
        'water_coords':    wc,
        'bounds':          bounds,
        'dsm_tile':        dsm_tile,
        'webodm_dtm_tile': webodm_dtm_tile,
        'bluesky_dtm_tile':bluesky_dtm_tile,
        'water_surface':   water_surface,
        'riverbed_w':      riverbed_w,
        'riverbed_b':      riverbed_b,
        'depth_w':         depth_w,
        'depth_b':         depth_b,
        'valid_w':         valid_w,
        'valid_b':         valid_b,
    }, "ok"

# =======================================================================
# PER-TILE COMPARISON PLOT
# =======================================================================

def visualize_tile_comparison(tile_num, image_path, res, output_path):

    fig, axes = plt.subplots(2, 4, figsize=(22, 12))

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None \
          else np.zeros((tile_size, tile_size, 3), dtype=np.uint8)

    wc = res['water_coords']
    r, c = wc[:, 0], wc[:, 1]
    vw, vb = res['valid_w'], res['valid_b']

    # Depth maps
    dm_w = np.full(res['mask'].shape, np.nan)
    dm_b = np.full(res['mask'].shape, np.nan)
    dm_w[r[vw], c[vw]] = res['depth_w'][vw]
    dm_b[r[vb], c[vb]] = res['depth_b'][vb]

    vmax = np.nanpercentile(
        np.concatenate([res['depth_w'][vw] if vw.any() else [2],
                        res['depth_b'][vb] if vb.any() else [2]]), 95)
    vmax = max(vmax, 0.5)

    # Row 0
    overlay = img.copy()
    col_mask = np.zeros_like(overlay)
    col_mask[res['mask'] > 0] = [0, 255, 255]
    overlay = cv2.addWeighted(overlay, 0.7, col_mask, 0.3, 0)
    axes[0, 0].imshow(overlay);  axes[0, 0].set_title('Image + Water Mask'); axes[0, 0].axis('off')

    im1 = axes[0, 1].imshow(res['webodm_dtm_tile'], cmap='terrain')
    axes[0, 1].set_title('WebODM DTM (no canopy penetration)'); axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], label='m', fraction=0.046)

    im2 = axes[0, 2].imshow(res['bluesky_dtm_tile'], cmap='terrain')
    axes[0, 2].set_title('Bluesky DTM (canopy penetrating, +58m)'); axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], label='m', fraction=0.046)

    diff = res['bluesky_dtm_tile'] - res['webodm_dtm_tile']
    im3  = axes[0, 3].imshow(diff, cmap='RdBu_r',
                              vmin=-np.nanstd(diff)*2, vmax=np.nanstd(diff)*2)
    axes[0, 3].set_title('DTM Difference (Bluesky − WebODM)'); axes[0, 3].axis('off')
    plt.colorbar(im3, ax=axes[0, 3], label='m', fraction=0.046)

    # Row 1
    im4 = axes[1, 0].imshow(dm_w, cmap='YlGnBu', vmin=0, vmax=vmax)
    axes[1, 0].set_title(f'Depth: WebODM DTM  (n={vw.sum():,})'); axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], label='Depth (m)', fraction=0.046)

    im5 = axes[1, 1].imshow(dm_b, cmap='YlGnBu', vmin=0, vmax=vmax)
    axes[1, 1].set_title(f'Depth: Bluesky DTM  (n={vb.sum():,})'); axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1], label='Depth (m)', fraction=0.046)

    depth_diff = dm_w - dm_b
    im6 = axes[1, 2].imshow(depth_diff, cmap='RdYlGn_r')
    axes[1, 2].set_title('Depth Overestimation (WebODM − Bluesky)'); axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2], label='m', fraction=0.046)

    # Histogram
    ax = axes[1, 3]
    if vw.any():
        ax.hist(res['depth_w'][vw], bins=35, alpha=0.6, color='red',
                label=f"WebODM  μ={res['depth_w'][vw].mean():.2f}m", density=True)
    if vb.any():
        ax.hist(res['depth_b'][vb], bins=35, alpha=0.6, color='steelblue',
                label=f"Bluesky μ={res['depth_b'][vb].mean():.2f}m", density=True)
    ax.axvline(1.0, color='black', linestyle='--', label='~Field max')
    ax.set_xlabel('Depth (m)'); ax.set_ylabel('Density')
    ax.set_title('Depth Distribution'); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle(f'Tile {tile_num} — DTM Source Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

# =======================================================================
# SUMMARY COMPARISON PLOTS  (same tiles, side by side)
# =======================================================================

def create_comparison_plots(df, output_dir):
    """
    6 panels comparing WebODM vs Bluesky on IDENTICAL tile set.
    Only tiles where BOTH have valid depths are included in direct comparisons.
    """

    # Tiles with valid data for each source
    w_ok = df['webodm_depth_mean'].notna()
    b_ok = df['bluesky_depth_mean'].notna()
    both = w_ok & b_ok

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))

    # 1. Histogram comparison
    ax = axes[0, 0]
    if w_ok.any():
        ax.hist(df.loc[w_ok, 'webodm_depth_mean'], bins=30, alpha=0.6,
                color='red', label=f'WebODM  n={w_ok.sum()}  μ={df.loc[w_ok,"webodm_depth_mean"].mean():.2f}m')
    if b_ok.any():
        ax.hist(df.loc[b_ok, 'bluesky_depth_mean'], bins=30, alpha=0.6,
                color='steelblue', label=f'Bluesky  n={b_ok.sum()}  μ={df.loc[b_ok,"bluesky_depth_mean"].mean():.2f}m')
    ax.axvline(1.0, color='black', linestyle='--', label='~Field max depth')
    ax.set_xlabel('Mean Depth per Tile (m)', fontweight='bold')
    ax.set_ylabel('Tile Count', fontweight='bold')
    ax.set_title('Depth Distribution — All Tiles\n(each tile appears in BOTH if data available)',
                fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    # 2. Scatter: WebODM vs Bluesky depth (tiles where both valid)
    ax = axes[0, 1]
    if both.any():
        ax.scatter(df.loc[both, 'webodm_depth_mean'],
                   df.loc[both, 'bluesky_depth_mean'],
                   alpha=0.5, s=20, color='purple')
        lim = max(df.loc[both, 'webodm_depth_mean'].max(),
                  df.loc[both, 'bluesky_depth_mean'].max()) * 1.05
        ax.plot([0, lim], [0, lim], 'k--', label='1:1 line')
        ax.set_xlabel('WebODM Depth (m)', fontweight='bold')
        ax.set_ylabel('Bluesky Depth (m)', fontweight='bold')
        ax.set_title(f'WebODM vs Bluesky Depth\n(n={both.sum()} tiles with both valid)',
                    fontweight='bold')
        ax.legend(); ax.grid(alpha=0.3)
        corr = df.loc[both, ['webodm_depth_mean', 'bluesky_depth_mean']].corr().iloc[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                fontsize=11, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat'))
    else:
        ax.text(0.5, 0.5, 'No tiles with both\nWebODM and Bluesky valid',
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

    # 3. Spatial map — which tiles have which data
    ax = axes[0, 2]
    colors = []
    for _, row in df.iterrows():
        has_w = not np.isnan(row['webodm_depth_mean'])
        has_b = not np.isnan(row['bluesky_depth_mean'])
        if has_w and has_b:
            colors.append('purple')
        elif has_w:
            colors.append('red')
        elif has_b:
            colors.append('steelblue')
        else:
            colors.append('lightgrey')

    ax.scatter(df['grid_col'], df['grid_row'], c=colors, s=30)
    ax.invert_yaxis()
    ax.set_xlabel('Grid Column', fontweight='bold')
    ax.set_ylabel('Grid Row', fontweight='bold')
    ax.set_title('Tile Coverage by DTM Source', fontweight='bold')

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor='purple',    label=f'Both  ({(np.array(colors)=="purple").sum()})'),
        Patch(facecolor='red',       label=f'WebODM only  ({(np.array(colors)=="red").sum()})'),
        Patch(facecolor='steelblue', label=f'Bluesky only  ({(np.array(colors)=="steelblue").sum()})'),
        Patch(facecolor='lightgrey', label=f'Neither  ({(np.array(colors)=="lightgrey").sum()})'),
    ]
    ax.legend(handles=legend_elems, fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)

    # 4. Depth overestimation per tile (WebODM - Bluesky)
    ax = axes[1, 0]
    if both.any():
        overestimate = df.loc[both, 'webodm_depth_mean'] - df.loc[both, 'bluesky_depth_mean']
        ax.hist(overestimate, bins=30, edgecolor='black', alpha=0.7, color='darkorange')
        ax.axvline(overestimate.mean(), color='red', linestyle='--',
                  label=f'Mean overestimate: {overestimate.mean():.2f}m')
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Depth Overestimation (WebODM − Bluesky, m)', fontweight='bold')
        ax.set_ylabel('Tile Count', fontweight='bold')
        ax.set_title('WebODM Systematic Overestimation\n(positive = WebODM deeper than Bluesky)',
                    fontweight='bold')
        ax.legend(); ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No overlapping tiles', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

    # 5. Coverage % comparison
    ax = axes[1, 1]
    if w_ok.any():
        ax.scatter(df.loc[w_ok, 'tile_number'],
                   df.loc[w_ok, 'webodm_coverage_pct'],
                   alpha=0.4, s=15, color='red', label='WebODM')
    if b_ok.any():
        ax.scatter(df.loc[b_ok, 'tile_number'],
                   df.loc[b_ok, 'bluesky_coverage_pct'],
                   alpha=0.4, s=15, color='steelblue', label='Bluesky')
    ax.set_xlabel('Tile Number', fontweight='bold')
    ax.set_ylabel('Valid Depth Coverage (%)', fontweight='bold')
    ax.set_title('Data Quality: Coverage per Tile', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    # 6. Summary statistics text
    ax = axes[1, 2]
    ax.axis('off')

    def fmt(series):
        if series.notna().any():
            return f"{series.mean():.3f} ± {series.std():.3f}m"
        return "N/A"

    txt = (
        f"COMPARISON SUMMARY\n"
        f"{'─'*38}\n\n"
        f"Total tiles in working_tiles.txt: {len(df)}\n\n"
        f"Tiles with valid depths:\n"
        f"  WebODM: {w_ok.sum()}\n"
        f"  Bluesky: {b_ok.sum()}\n"
        f"  Both: {both.sum()}\n\n"
        f"Mean depth per tile:\n"
        f"  WebODM: {fmt(df.loc[w_ok,'webodm_depth_mean'])}\n"
        f"  Bluesky: {fmt(df.loc[b_ok,'bluesky_depth_mean'])}\n\n"
        f"WebODM overestimation:\n"
    )
    if both.any():
        oe = (df.loc[both,'webodm_depth_mean'] - df.loc[both,'bluesky_depth_mean'])
        txt += f"  Mean: {oe.mean():.3f}m\n  Std:  {oe.std():.3f}m\n\n"
    else:
        txt += "  N/A (no overlapping tiles)\n\n"

    txt += (
        f"Depth bounds applied: {DEPTH_MIN}–{DEPTH_MAX}m\n"
        f"Bluesky geoid corr: +58.0m\n"
        f"Bluesky resample: bicubic (5m→0.061m)\n"
    )
    ax.text(0.05, 0.95, txt, transform=ax.transAxes,
           fontfamily='monospace', fontsize=10, va='top')

    plt.suptitle('WebODM DTM vs Bluesky DTM — Water Depth Comparison\n'
                f'Identical tile set  |  Depth bounds {DEPTH_MIN}–{DEPTH_MAX}m  |  '
                f'{len(df)} tiles total',
                fontsize=13, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir, 'comparison_plots.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison plots: {path}")
    plt.close()

# =======================================================================
# MAIN
# =======================================================================

def batch_extract_both():

    print("="*70)
    print("UNIFIED BATCH DEPTH EXTRACTION — WebODM + Bluesky DTM")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Depth bounds: {DEPTH_MIN}–{DEPTH_MAX}m (same for both sources)")

    # Check required files
    for label, path in [
        ("Orthophoto",      orthophoto_file),
        ("WebODM DSM",      webodm_dsm_file),
        ("WebODM DTM",      webodm_dtm_file),
        ("Bluesky DTM",     bluesky_resampled_file),
        ("Working tiles",   working_tiles_file),
    ]:
        status = "✓" if os.path.exists(path) else "✗ MISSING"
        print(f"  {status}  {label}: {os.path.basename(path)}")

    if not os.path.exists(bluesky_resampled_file):
        print("\n  Run save_bluesky_resampled.py first!")
        return
    if not os.path.exists(working_tiles_file):
        print("\n  Run find_valid_from_files.py first!")
        return

    with rasterio.open(orthophoto_file) as src:
        ortho_w, ortho_h = src.width, src.height
        ortho_transform  = src.transform

    working_tiles = load_working_tiles(working_tiles_file)
    print(f"\n✓ Loaded {len(working_tiles)} tiles")

    # ------------------------------------------------------------------
    # Process every tile — store results for BOTH DTMs
    # ------------------------------------------------------------------

    all_rows  = []
    n_ok      = n_no_water = n_no_dsm = n_failed = 0

    print(f"\n{'='*70}")
    print(f"PROCESSING {len(working_tiles)} TILES (both DTM sources per tile)")
    print(f"{'='*70}\n")

    for tile_info in tqdm(working_tiles, desc="Processing"):

        tile_num   = tile_info['tile_num']
        image_path = os.path.join(images_dir, tile_info['image_file'])
        mask_path  = os.path.join(masks_dir,  tile_info['mask_file'])

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            n_failed += 1;  continue

        tpos, gpos = tile_position(tile_num, stride, ortho_w)

        try:
            res, status = process_tile(mask_path, image_path, tpos, ortho_transform)
        except Exception as e:
            n_failed += 1;  continue

        if status == "no_water":   n_no_water += 1; continue
        if status == "no_dsm":     n_no_dsm   += 1; continue
        if status == "mask_not_found": n_failed += 1; continue

        n_ok += 1

        # Compute stats for WebODM
        ws = compute_depth_stats(res['depth_w'], res['valid_w'])
        # Compute stats for Bluesky
        bs = compute_depth_stats(res['depth_b'], res['valid_b'])

        water_px = int(np.sum(res['mask']))

        row = {
            'tile_number':          tile_num,
            'grid_row':             gpos[0],
            'grid_col':             gpos[1],

            # WebODM DTM columns
            'webodm_valid_depths':  ws['n'] if not np.isnan(ws['n']) else None,
            'webodm_coverage_pct':  ws['n'] / water_px * 100 if not np.isnan(ws['n']) else np.nan,
            'webodm_depth_mean':    ws['mean'],
            'webodm_depth_median':  ws['median'],
            'webodm_depth_std':     ws['std'],
            'webodm_depth_min':     ws['min'],
            'webodm_depth_max':     ws['max'],
            'webodm_surface_mean':  float(np.nanmean(res['water_surface'][res['valid_w']])) if res['valid_w'].any() else np.nan,
            'webodm_riverbed_mean': float(np.nanmean(res['riverbed_w'][res['valid_w']])) if res['valid_w'].any() else np.nan,

            # Bluesky DTM columns
            'bluesky_valid_depths': bs['n'] if not np.isnan(bs['n']) else None,
            'bluesky_coverage_pct': bs['n'] / water_px * 100 if not np.isnan(bs['n']) else np.nan,
            'bluesky_depth_mean':   bs['mean'],
            'bluesky_depth_median': bs['median'],
            'bluesky_depth_std':    bs['std'],
            'bluesky_depth_min':    bs['min'],
            'bluesky_depth_max':    bs['max'],
            'bluesky_surface_mean': float(np.nanmean(res['water_surface'][res['valid_b']])) if res['valid_b'].any() else np.nan,
            'bluesky_riverbed_mean':float(np.nanmean(res['riverbed_b'][res['valid_b']])) if res['valid_b'].any() else np.nan,

            'water_pixels':         water_px,
            'image_file':           tile_info['image_file'],
            'mask_file':            tile_info['mask_file'],
        }
        all_rows.append(row)

        # Per-tile comparison plot every 10th successful tile
        if n_ok % 10 == 1 or n_ok <= 3:
            viz_path = os.path.join(output_dir, 'per_tile_comparison',
                                   f'tile_{tile_num:04d}.png')
            try:
                visualize_tile_comparison(tile_num, image_path, res, viz_path)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")

    if not all_rows:
        print("✗ No results to save!")
        return

    df = pd.DataFrame(all_rows)

    # Single combined CSV (most useful)
    csv_path = os.path.join(output_dir, 'water_depths_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Combined CSV: {csv_path}")

    # Separate CSVs per source (for scripts expecting single-source format)
    webodm_cols  = ['tile_number','grid_row','grid_col','water_pixels','image_file','mask_file'] + \
                   [c for c in df.columns if c.startswith('webodm')]
    bluesky_cols = ['tile_number','grid_row','grid_col','water_pixels','image_file','mask_file'] + \
                   [c for c in df.columns if c.startswith('bluesky')]

    df[webodm_cols].rename(columns=lambda x: x.replace('webodm_', '')) \
                   .to_csv(os.path.join(output_dir, 'water_depths_webodm.csv'), index=False)
    df[bluesky_cols].rename(columns=lambda x: x.replace('bluesky_', '')) \
                    .to_csv(os.path.join(output_dir, 'water_depths_bluesky.csv'), index=False)
    print(f"✓ WebODM CSV:  water_depths_webodm.csv")
    print(f"✓ Bluesky CSV: water_depths_bluesky.csv")

    # Summary plots
    create_comparison_plots(df, output_dir)

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------

    w_ok   = df['webodm_depth_mean'].notna()
    b_ok   = df['bluesky_depth_mean'].notna()
    both   = w_ok & b_ok

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"Tiles attempted:  {len(working_tiles)}")
    print(f"  DSM valid:      {n_ok}  (both sources run on these)")
    print(f"  No water:       {n_no_water}")
    print(f"  No DSM data:    {n_no_dsm}")
    print(f"  Errors:         {n_failed}")
    print(f"\nValid depths found:")
    print(f"  WebODM DTM:  {w_ok.sum()} tiles")
    print(f"  Bluesky DTM: {b_ok.sum()} tiles")
    print(f"  Both:        {both.sum()} tiles")

    if w_ok.any():
        print(f"\nWebODM  mean depth: {df.loc[w_ok,'webodm_depth_mean'].mean():.3f}m")
    if b_ok.any():
        print(f"Bluesky mean depth: {df.loc[b_ok,'bluesky_depth_mean'].mean():.3f}m")
    if both.any():
        oe = (df.loc[both,'webodm_depth_mean'] - df.loc[both,'bluesky_depth_mean'])
        print(f"WebODM overestimation: {oe.mean():.3f} ± {oe.std():.3f}m")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results:  {output_dir}")


if __name__ == "__main__":
    batch_extract_both()
