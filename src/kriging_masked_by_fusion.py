"""
Masked Kriging-based DTM Upsampling
Uses the DSM-guided fusion output as a spatial mask to define where kriging is needed.

Strategy:
  1. Read the fusion DTM output (bluesky_dtm_fused_0061m.tif)
  2. Find bounding box of all valid (non-NaN) pixels
  3. Clip the 5m Bluesky DTM to that bounding box
  4. Run kriging ONLY within that clipped region
  5. Save output aligned with the fusion grid

Benefits:
  - Reduces kriging extent by ~90% (only river corridor, not entire catchment)
  - Makes kriging/fusion/bicubic directly comparable on identical pixels
  - Feasible RAM and runtime even for large study areas
  
Runtime estimate: 
  For typical river corridor (10k × 15k px), expect ~15-45 minutes
"""

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds as window_from_bounds
from rasterio.transform import from_bounds
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

# =======================================================================
# FILES — UPDATE THESE
# =======================================================================

# Fusion output — used to define the region of interest
fusion_dtm_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\fused\bluesky_dtm_fused_0061m.tif"

# 5m Bluesky DTM (will be clipped to ROI)
bluesky_dtm_5m_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\5m_dtm\bluesky_dtm_utm29n.tif"

# Outputs
output_kriged    = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\kriged\bluesky_dtm_kriged_masked_0061m.tif"
output_variance  = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\kriged\bluesky_dtm_kriging_variance_masked.tif"
output_variogram = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\kriged\variogram_fit_masked.png"
output_comparison = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\kriged\fusion_vs_kriging_comparison.png"

GEOID_OFFSET = 58.0   # metres
NODATA_OUT   = -9999.0

# =======================================================================
# KRIGING PARAMETERS
# =======================================================================

VARIOGRAM_MODEL = 'spherical'
BLOCK_SIZE_PX = 512          # Smaller blocks for faster processing
SEARCH_RADIUS_M = 25.0
MAX_NEIGHBOURS = 36
N_VARIOGRAM_SAMPLES = 3000
BLOCK_OVERLAP_PX = 32

# =======================================================================
# HELPERS
# =======================================================================

def save_raster(array, path, transform, crs, nodata=NODATA_OUT, desc=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.where(np.isnan(array), nodata, array).astype(np.float32)
    with rasterio.open(path, "w",
                       driver="GTiff", dtype="float32", nodata=nodata,
                       crs=crs, transform=transform,
                       width=array.shape[1], height=array.shape[0], count=1,
                       compress="deflate", tiled=True,
                       blockxsize=256, blockysize=256, predictor=3) as dst:
        dst.write(data, 1)
    gb = os.path.getsize(path) / 1024**3
    print(f"  Saved {desc}: {os.path.basename(path)}  ({gb:.2f} GB)")


# =======================================================================
# STEP 1: FIND ROI FROM FUSION OUTPUT
# =======================================================================

def find_roi_from_fusion():
    """
    Read fusion DTM and find bounding box of valid data.
    Returns (window, transform, crs, width, height, pixel_size).
    """
    print("\n" + "="*70)
    print("STEP 1: Finding region of interest from fusion output")
    print("="*70)

    with rasterio.open(fusion_dtm_file) as src:
        fusion_data = src.read(1)
        fusion_trans = src.transform
        fusion_crs = src.crs
        pixel_size = abs(src.transform.a)

        # Find valid pixels
        valid_mask = ~np.isnan(fusion_data)
        
        if not valid_mask.any():
            raise ValueError("Fusion DTM has no valid data!")

        # Bounding box of valid data
        rows, cols = np.where(valid_mask)
        r0, r1 = rows.min(), rows.max() + 1
        c0, c1 = cols.min(), cols.max() + 1

        # Add small buffer (100 pixels ~6m each side)
        buffer_px = 100
        r0 = max(0, r0 - buffer_px)
        r1 = min(src.height, r1 + buffer_px)
        c0 = max(0, c0 - buffer_px)
        c1 = min(src.width, c1 + buffer_px)

        roi_h = r1 - r0
        roi_w = c1 - c0

        # Transform for the ROI window
        roi_trans = fusion_trans * fusion_trans.translation(c0, r0)

        # World bounds
        left = fusion_trans.c + c0 * fusion_trans.a
        top = fusion_trans.f + r0 * fusion_trans.e
        right = fusion_trans.c + c1 * fusion_trans.a
        bottom = fusion_trans.f + r1 * fusion_trans.e
        roi_bounds = (left, bottom, right, top)

    print(f"  Fusion extent:   {src.height} × {src.width} pixels")
    print(f"  Valid coverage:  {valid_mask.sum():,} pixels "
          f"({valid_mask.sum()/valid_mask.size*100:.1f}%)")
    print(f"\n  Valid data bbox: rows {rows.min()}–{rows.max()}, "
          f"cols {cols.min()}–{cols.max()}")
    print(f"  + Buffer ({buffer_px}px): rows {r0}–{r1}, cols {c0}–{c1}")
    print(f"  ROI size:        {roi_h} × {roi_w} pixels  "
          f"({roi_h * pixel_size / 1000:.2f}km × {roi_w * pixel_size / 1000:.2f}km)")
    print(f"  ROI world bounds: {left:.0f}, {bottom:.0f}, {right:.0f}, {top:.0f}")
    print(f"\n  ⚠️  If ROI size ≈ fusion extent, then fusion has sparse valid data!")
    print(f"      This means kriging will only predict where fusion has data.")

    # Memory estimate
    mem_gb = (roi_h * roi_w * 4 * 3) / 1024**3  # 3 float32 arrays
    print(f"\n  Memory required: {mem_gb:.1f} GB")
    
    if mem_gb > 16:
        print(f"  ⚠️  Still quite large. Consider increasing BLOCK_SIZE_PX to 1024")
        print(f"     or further restricting the ROI to just the main channel.")

    return roi_bounds, roi_trans, fusion_crs, roi_w, roi_h, pixel_size


# =======================================================================
# STEP 2: LOAD 5m DTM POINTS WITHIN ROI
# =======================================================================

def load_dtm_points_in_roi(roi_bounds):
    """
    Load 5m DTM, clip to ROI bounds, return as point arrays.
    """
    print("\n" + "="*70)
    print("STEP 2: Loading 5m DTM points within ROI")
    print("="*70)

    with rasterio.open(bluesky_dtm_5m_file) as src:
        dtm_bounds = src.bounds
        print(f"  5m DTM full extent: {dtm_bounds}")
        print(f"  ROI target extent:  left={roi_bounds[0]:.0f}, bottom={roi_bounds[1]:.0f}, "
              f"right={roi_bounds[2]:.0f}, top={roi_bounds[3]:.0f}")
        
        # Check overlap
        overlap_left = max(dtm_bounds.left, roi_bounds[0])
        overlap_bottom = max(dtm_bounds.bottom, roi_bounds[1])
        overlap_right = min(dtm_bounds.right, roi_bounds[2])
        overlap_top = min(dtm_bounds.top, roi_bounds[3])
        
        if overlap_left >= overlap_right or overlap_bottom >= overlap_top:
            print(f"\n  ✗ ERROR: NO OVERLAP between 5m DTM and ROI!")
            print(f"     5m DTM extent: {dtm_bounds}")
            print(f"     Fusion ROI:    {roi_bounds}")
            print(f"\n     This means fusion is using different data than Bluesky 5m DTM.")
            print(f"     Possible causes:")
            print(f"       1. Wrong 5m DTM file path")
            print(f"       2. Fusion used a different DTM source")
            print(f"       3. Coordinate system mismatch")
            return None, None, None, None, None
        
        overlap_coverage = ((overlap_right - overlap_left) * (overlap_top - overlap_bottom)) / \
                          ((roi_bounds[2] - roi_bounds[0]) * (roi_bounds[3] - roi_bounds[1]))
        
        print(f"  Overlap coverage: {overlap_coverage*100:.1f}% of ROI")
        
        if overlap_coverage < 0.5:
            print(f"  ⚠️  WARNING: 5m DTM only covers {overlap_coverage*100:.1f}% of ROI!")
            print(f"     Kriging output will be sparse.")
        
        # Read window that overlaps ROI
        window = window_from_bounds(*roi_bounds, transform=src.transform)
        dtm_data = src.read(1, window=window).astype(np.float64)
        dtm_trans = src.window_transform(window)
        dtm_nodata = src.nodata if src.nodata is not None else NODATA_OUT

    # Mask nodata
    valid = (dtm_data != dtm_nodata) & ~np.isnan(dtm_data)
    dtm_data[~valid] = np.nan

    # Apply geoid correction
    dtm_data[valid] += GEOID_OFFSET

    # Extract valid points
    rows, cols = np.where(valid)
    
    if len(rows) == 0:
        print(f"\n  ✗ ERROR: No valid 5m DTM points found in ROI window!")
        print(f"     Window read: {dtm_data.shape}")
        print(f"     This means the 5m DTM doesn't cover this area.")
        return None, None, None, None, None
    
    x_pts = dtm_trans.c + (cols + 0.5) * dtm_trans.a
    y_pts = dtm_trans.f + (rows + 0.5) * dtm_trans.e
    z_pts = dtm_data[rows, cols]
    
    # Calculate actual spatial extent of valid points
    pts_bounds = (x_pts.min(), y_pts.min(), x_pts.max(), y_pts.max())
    pts_coverage = ((pts_bounds[2] - pts_bounds[0]) * (pts_bounds[3] - pts_bounds[1])) / \
                   ((roi_bounds[2] - roi_bounds[0]) * (roi_bounds[3] - roi_bounds[1]))

    print(f"\n  DTM window read: {dtm_data.shape[0]} × {dtm_data.shape[1]} pixels")
    print(f"  Valid pts:       {len(z_pts):,}")
    print(f"  Valid pts extent: left={x_pts.min():.0f}, bottom={y_pts.min():.0f}, "
          f"right={x_pts.max():.0f}, top={y_pts.max():.0f}")
    print(f"  Valid pts coverage: {pts_coverage*100:.1f}% of ROI")
    print(f"  Elevation:       {z_pts.min():.2f} to {z_pts.max():.2f} m  "
          f"(after +{GEOID_OFFSET}m geoid)")
    
    if pts_coverage < 0.5:
        print(f"\n  ⚠️  CRITICAL: 5m DTM points only cover {pts_coverage*100:.1f}% of ROI!")
        print(f"     Expected coverage: >90%")
        print(f"     Kriging will only work where DTM points exist.")
        print(f"\n  Likely causes:")
        print(f"     1. Bluesky 5m DTM has limited spatial extent")
        print(f"     2. Fusion used different/additional data sources")
        print(f"     3. ROI is larger than Bluesky DTM coverage")
        print(f"\n  Solution: Use fusion DTM as input for kriging instead of 5m DTM")

    return x_pts, y_pts, z_pts, dtm_data, dtm_trans


# =======================================================================
# STEP 3: FIT VARIOGRAM
# =======================================================================

def fit_variogram(x_pts, y_pts, z_pts):
    print("\n" + "="*70)
    print(f"STEP 3: Fitting {VARIOGRAM_MODEL} variogram")
    print("="*70)

    n_pts = len(x_pts)
    n_sample = min(N_VARIOGRAM_SAMPLES, n_pts)
    rng = np.random.default_rng(seed=42)
    idx = rng.choice(n_pts, size=n_sample, replace=False)

    x_s, y_s, z_s = x_pts[idx], y_pts[idx], z_pts[idx]
    print(f"  Fitting from {n_sample:,} sample points...")

    t0 = time.time()
    ok = OrdinaryKriging(
        x_s, y_s, z_s,
        variogram_model=VARIOGRAM_MODEL,
        verbose=False,
        enable_plotting=False,
        nlags=20,
        weight=True,
        coordinates_type='euclidean',
    )
    elapsed = time.time() - t0

    params = ok.variogram_model_parameters
    print(f"  Variogram fit ({elapsed:.1f}s):")
    if VARIOGRAM_MODEL in ('spherical', 'exponential', 'gaussian'):
        print(f"    Nugget: {params[2]:.4f} m²")
        print(f"    Sill:   {params[1] + params[2]:.4f} m²")
        print(f"    Range:  {params[0]:.2f} m")

    # Save plot
    fig, ax = plt.subplots(figsize=(8, 5))
    lags = ok.lags
    semivariance = ok.semivariance
    fitted = ok.variogram_function(params, lags)

    ax.scatter(lags, semivariance, s=40, color='steelblue', zorder=3,
               label='Empirical variogram')
    ax.plot(lags, fitted, 'r-', linewidth=2,
            label=f'{VARIOGRAM_MODEL.capitalize()} model')

    if VARIOGRAM_MODEL in ('spherical', 'exponential', 'gaussian'):
        ax.axhline(params[1] + params[2], color='grey', linestyle='--',
                   alpha=0.6, label=f'Sill = {params[1]+params[2]:.3f} m²')
        ax.axvline(params[0], color='orange', linestyle='--',
                   alpha=0.6, label=f'Range = {params[0]:.1f} m')

    ax.set_xlabel('Lag distance (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Semivariance (m²)', fontsize=12, fontweight='bold')
    ax.set_title(f'Variogram — {VARIOGRAM_MODEL.capitalize()} model (ROI only)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(output_variogram), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_variogram, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Variogram: {output_variogram}")

    return ok


# =======================================================================
# STEP 4: BLOCK KRIGING (same as before, but on ROI only)
# =======================================================================

def krige_block(ok_model, x_pts, y_pts, z_pts, kdtree,
                block_x_grid, block_y_grid):
    out_h, out_w = block_x_grid.shape
    block_centre_x = np.mean(block_x_grid)
    block_centre_y = np.mean(block_y_grid)

    block_half_diag = np.sqrt((block_x_grid.max() - block_x_grid.min())**2 +
                               (block_y_grid.max() - block_y_grid.min())**2) / 2
    search_r = SEARCH_RADIUS_M + block_half_diag

    idx_nearby = kdtree.query_ball_point([block_centre_x, block_centre_y], r=search_r)

    if len(idx_nearby) < 4:
        nan_block = np.full((out_h, out_w), np.nan, dtype=np.float32)
        return nan_block, nan_block

    if len(idx_nearby) > MAX_NEIGHBOURS:
        nearby_coords = np.column_stack([x_pts[idx_nearby], y_pts[idx_nearby]])
        dists = np.linalg.norm(nearby_coords - [block_centre_x, block_centre_y], axis=1)
        top_n = np.argsort(dists)[:MAX_NEIGHBOURS]
        idx_nearby = [idx_nearby[i] for i in top_n]

    x_local = x_pts[idx_nearby]
    y_local = y_pts[idx_nearby]
    z_local = z_pts[idx_nearby]

    if np.std(z_local) < 1e-6:
        pred = np.full((out_h, out_w), np.mean(z_local), dtype=np.float32)
        var  = np.zeros((out_h, out_w), dtype=np.float32)
        return pred, var

    try:
        ok_local = OrdinaryKriging(
            x_local, y_local, z_local,
            variogram_model=ok_model.variogram_model,
            variogram_parameters=ok_model.variogram_model_parameters,
            verbose=False,
            enable_plotting=False,
            coordinates_type='euclidean',
        )

        z_pred, z_var = ok_local.execute('points',
                                          block_x_grid.ravel(),
                                          block_y_grid.ravel())

        pred = z_pred.data.reshape(out_h, out_w).astype(np.float32)
        var  = z_var.data.reshape(out_h, out_w).astype(np.float32)
        pred[pred == 0.0] = np.nan
        var[var < 0] = 0.0

        return pred, var

    except Exception:
        nan_block = np.full((out_h, out_w), np.nan, dtype=np.float32)
        return nan_block, nan_block


def block_krige_roi(ok_model, x_pts, y_pts, z_pts, roi_trans, roi_h, roi_w, pixel_size):
    print("\n" + "="*70)
    print("STEP 4: Block Kriging (ROI only)")
    print("="*70)

    print(f"  ROI size:       {roi_h} × {roi_w} pixels")
    print(f"  Block size:     {BLOCK_SIZE_PX} × {BLOCK_SIZE_PX} px")
    print(f"  Search radius:  {SEARCH_RADIUS_M}m")
    print(f"  Max neighbours: {MAX_NEIGHBOURS}")

    n_blocks_r = int(np.ceil(roi_h / BLOCK_SIZE_PX))
    n_blocks_c = int(np.ceil(roi_w / BLOCK_SIZE_PX))
    n_blocks   = n_blocks_r * n_blocks_c
    print(f"  Blocks:         {n_blocks_r} × {n_blocks_c} = {n_blocks} total")
    print(f"  Est. time:      ~{n_blocks * 3 / 60:.0f} min\n")

    from scipy.spatial import cKDTree
    coords = np.column_stack([x_pts, y_pts])
    kdtree = cKDTree(coords)

    pred_acc = np.zeros((roi_h, roi_w), dtype=np.float32)
    var_acc  = np.zeros((roi_h, roi_w), dtype=np.float32)
    weight_acc = np.zeros((roi_h, roi_w), dtype=np.float32)

    t_start = time.time()

    for ri in tqdm(range(n_blocks_r), desc="Row blocks"):
        for ci in range(n_blocks_c):
            r0 = max(0, ri * BLOCK_SIZE_PX - BLOCK_OVERLAP_PX)
            r1 = min(roi_h, (ri + 1) * BLOCK_SIZE_PX + BLOCK_OVERLAP_PX)
            c0 = max(0, ci * BLOCK_SIZE_PX - BLOCK_OVERLAP_PX)
            c1 = min(roi_w, (ci + 1) * BLOCK_SIZE_PX + BLOCK_OVERLAP_PX)

            cols_idx = np.arange(c0, c1)
            rows_idx = np.arange(r0, r1)
            x_out = roi_trans.c + (cols_idx + 0.5) * roi_trans.a
            y_out = roi_trans.f + (rows_idx + 0.5) * roi_trans.e

            x_grid, y_grid = np.meshgrid(x_out, y_out)

            pred, var = krige_block(ok_model, x_pts, y_pts, z_pts, kdtree,
                                    x_grid, y_grid)

            # Taper
            taper = np.ones_like(pred)
            taper_px = BLOCK_OVERLAP_PX
            if taper_px > 0:
                for i in range(taper_px):
                    w = (i + 1) / (taper_px + 1)
                    if i < pred.shape[0]:  taper[i, :] = np.minimum(taper[i, :], w)
                    if pred.shape[0] - 1 - i >= 0:
                        ii = pred.shape[0] - 1 - i
                        if ii >= 0: taper[ii, :] = np.minimum(taper[ii, :], w)
                    if i < pred.shape[1]:  taper[:, i] = np.minimum(taper[:, i], w)
                    if pred.shape[1] - 1 - i >= 0:
                        jj = pred.shape[1] - 1 - i
                        if jj >= 0: taper[:, jj] = np.minimum(taper[:, jj], w)

            valid = ~np.isnan(pred)
            pred_acc[r0:r1, c0:c1][valid] += pred[valid] * taper[valid]
            var_acc [r0:r1, c0:c1][valid] += var [valid] * taper[valid]
            weight_acc[r0:r1, c0:c1][valid] += taper[valid]

    elapsed = time.time() - t_start
    print(f"\n  Kriging complete in {elapsed/60:.1f} min")

    with np.errstate(invalid='ignore'):
        pred_final = np.where(weight_acc > 0, pred_acc / weight_acc, np.nan)
        var_final  = np.where(weight_acc > 0, var_acc  / weight_acc, np.nan)

    del pred_acc, var_acc, weight_acc

    valid = ~np.isnan(pred_final)
    print(f"  Output coverage: {valid.sum():,} / {roi_h*roi_w:,} pixels "
          f"({valid.sum()/(roi_h*roi_w)*100:.1f}%)")
    print(f"  Elevation range: {np.nanmin(pred_final):.2f} to {np.nanmax(pred_final):.2f} m")
    print(f"  Mean uncertainty: {np.nanmean(np.sqrt(var_final)):.4f} m")

    return pred_final.astype(np.float32), var_final.astype(np.float32)


# =======================================================================
# STEP 5: COMPARE WITH FUSION OUTPUT
# =======================================================================

def compare_fusion_vs_kriging(pred, roi_trans, roi_h, roi_w):
    print("\n" + "="*70)
    print("STEP 5: Comparing with fusion output")
    print("="*70)

    # Read fusion DTM for same ROI window
    with rasterio.open(fusion_dtm_file) as src:
        # Find window in fusion that matches roi_trans
        # (they should already be aligned, but let's be careful)
        window = window_from_bounds(
            roi_trans.c,
            roi_trans.f + roi_h * roi_trans.e,
            roi_trans.c + roi_w * roi_trans.a,
            roi_trans.f,
            transform=src.transform
        )
        fusion_crop = src.read(1, window=window)

    # Ensure shapes match
    if fusion_crop.shape != pred.shape:
        min_h = min(fusion_crop.shape[0], pred.shape[0])
        min_w = min(fusion_crop.shape[1], pred.shape[1])
        fusion_crop = fusion_crop[:min_h, :min_w]
        pred = pred[:min_h, :min_w]

    # Compare on pixels where both are valid
    both_valid = ~np.isnan(fusion_crop) & ~np.isnan(pred)
    
    if not both_valid.any():
        print("  ⚠️  No overlapping valid pixels for comparison")
        return

    diff = pred[both_valid] - fusion_crop[both_valid]

    print(f"  Compared pixels: {both_valid.sum():,}")
    print(f"  Kriging mean:    {pred[both_valid].mean():.3f} m")
    print(f"  Fusion mean:     {fusion_crop[both_valid].mean():.3f} m")
    print(f"  Mean diff:       {diff.mean():.4f} m")
    print(f"  Std diff:        {diff.std():.4f} m")
    print(f"  RMSE:            {np.sqrt((diff**2).mean()):.4f} m")
    print(f"  Max abs diff:    {np.abs(diff).max():.4f} m")

    # Downsample for plotting to avoid 74GB matplotlib allocation
    # Target: ~2000x3000 px max for visualization
    MAX_PLOT_DIM = 2000
    
    if max(pred.shape) > MAX_PLOT_DIM:
        downsample_factor = max(pred.shape) // MAX_PLOT_DIM + 1
        print(f"\n  Downsampling by {downsample_factor}× for visualization "
              f"({pred.shape[0]}×{pred.shape[1]} → "
              f"{pred.shape[0]//downsample_factor}×{pred.shape[1]//downsample_factor})")
        
        fusion_plot = fusion_crop[::downsample_factor, ::downsample_factor]
        pred_plot = pred[::downsample_factor, ::downsample_factor]
        both_valid_plot = both_valid[::downsample_factor, ::downsample_factor]
    else:
        fusion_plot = fusion_crop
        pred_plot = pred
        both_valid_plot = both_valid

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    extent = [0, pred.shape[1] * 0.061 / 1000, 0, pred.shape[0] * 0.061 / 1000]
    
    im1 = axes[0].imshow(fusion_plot, cmap='terrain', extent=extent)
    axes[0].set_title('DSM-Guided Fusion', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('km'); axes[0].set_ylabel('km')
    plt.colorbar(im1, ax=axes[0], label='Elevation (m)')

    im2 = axes[1].imshow(pred_plot, cmap='terrain', extent=extent)
    axes[1].set_title('Kriging', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('km'); axes[1].set_ylabel('km')
    plt.colorbar(im2, ax=axes[1], label='Elevation (m)')

    diff_map = np.full_like(pred_plot, np.nan)
    diff_map[both_valid_plot] = pred_plot[both_valid_plot] - fusion_plot[both_valid_plot]
    im3 = axes[2].imshow(diff_map, cmap='RdBu_r', vmin=-1, vmax=1, extent=extent)
    axes[2].set_title('Difference: Kriging − Fusion', fontweight='bold', fontsize=12)
    axes[2].set_xlabel('km'); axes[2].set_ylabel('km')
    plt.colorbar(im3, ax=axes[2], label='m')

    plt.suptitle(f'DTM Upsampling Method Comparison\n'
                 f'RMSE = {np.sqrt((diff**2).mean()):.4f}m  |  '
                 f'Mean diff = {diff.mean():.4f}m',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_comparison), exist_ok=True)
    plt.savefig(output_comparison, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Comparison: {output_comparison}")


# =======================================================================
# MAIN
# =======================================================================

def run_masked_kriging():
    print("="*70)
    print("MASKED KRIGING — Using fusion output as spatial mask")
    print("="*70)

    for label, path in [
        ("Fusion DTM",    fusion_dtm_file),
        ("Bluesky 5m DTM", bluesky_dtm_5m_file),
    ]:
        status = "✓" if os.path.exists(path) else "✗ MISSING"
        print(f"  {status}  {label}: {os.path.basename(path)}")

    if not os.path.exists(fusion_dtm_file):
        print("\n✗ Fusion DTM missing. Run dsm_guided_dtm_fusion.py first.")
        return

    # Step 1
    roi_bounds, roi_trans, roi_crs, roi_w, roi_h, pixel_size = find_roi_from_fusion()

    # Step 2
    result = load_dtm_points_in_roi(roi_bounds)
    if result[0] is None:
        print("\n" + "="*70)
        print("✗ KRIGING CANNOT PROCEED")
        print("="*70)
        print("The 5m Bluesky DTM does not cover the fusion output extent.")
        print("\nTwo possible solutions:")
        print("\n1. Use fusion DTM itself for kriging comparison:")
        print("   - The fusion already incorporates DTM data")
        print("   - You can resample fusion at different resolutions to compare")
        print("\n2. Check if you have the correct 5m DTM file:")
        print("   - Verify the file path in the script")
        print("   - Check that it covers your study area")
        print("   - Compare CRS with fusion output")
        return
    
    x_pts, y_pts, z_pts, dtm_data, dtm_trans = result

    # Step 3
    ok_model = fit_variogram(x_pts, y_pts, z_pts)

    # Step 4
    pred, var = block_krige_roi(ok_model, x_pts, y_pts, z_pts,
                                roi_trans, roi_h, roi_w, pixel_size)

    # Save
    print("\n" + "="*70)
    print("Saving outputs")
    print("="*70)
    save_raster(pred, output_kriged,   roi_trans, roi_crs,
                desc="Kriged DTM (masked, 0.061m)")
    save_raster(var,  output_variance, roi_trans, roi_crs,
                desc="Kriging variance (m²)")
    
    # Print diagnostic info
    valid_mask = ~np.isnan(pred)
    print("\n" + "="*70)
    print("OUTPUT DIAGNOSTIC")
    print("="*70)
    print(f"  Output dimensions: {roi_h} × {roi_w} pixels")
    print(f"  Valid kriged pixels: {valid_mask.sum():,} ({valid_mask.sum()/(roi_h*roi_w)*100:.1f}%)")
    print(f"  NaN pixels: {(~valid_mask).sum():,} ({(~valid_mask).sum()/(roi_h*roi_w)*100:.1f}%)")
    print(f"\n  If most pixels are NaN:")
    print(f"    → This is NORMAL if your fusion output also has sparse coverage")
    print(f"    → The output will look 'mostly dark' in viewers")
    print(f"    → But the valid data IS there (proven by comparison stats)")
    print(f"\n  To visualize better:")
    print(f"    → Run crop_to_valid_bbox.py to remove NaN padding")
    print(f"    → Or zoom to layer extent in QGIS")
    print(f"    → Or adjust symbology stretch to Min/Max")

    # Step 5
    compare_fusion_vs_kriging(pred, roi_trans, roi_h, roi_w)

    print("\n" + "="*70)
    print("✅  MASKED KRIGING COMPLETE")
    print("="*70)
    print(f"\n  Kriged DTM:  {output_kriged}")
    print(f"  Variance:    {output_variance}")
    print(f"  Variogram:   {output_variogram}")
    print(f"  Comparison:  {output_comparison}")
    print(f"\n  This output is spatially aligned with the fusion output.")
    print(f"  Use it in batch_unified_comparison.py for direct comparison.")


if __name__ == "__main__":
    run_masked_kriging()
