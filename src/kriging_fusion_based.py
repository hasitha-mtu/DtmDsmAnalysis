"""
Kriging-based DTM Upsampling Using Fusion DTM as Input

Strategy:
  1. Read fusion DTM (0.061m resolution)
  2. Downsample to 5m (baseline for comparison)
  3. Krige 5m → 0.061m
  4. Compare with original fusion DTM

This tests whether kriging adds value over the fusion method's implicit 
interpolation, using the same data as input.

Comparison question: Does geostatistical kriging improve upon the fusion 
method's bicubic-like interpolation when starting from the same data?
"""

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import zoom as scipy_zoom
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

# =======================================================================
# FILES — UPDATE THESE
# =======================================================================

# Fusion output — will be downsampled to 5m then kriged back to 0.061m
fusion_dtm_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\fused\bluesky_dtm_fused_0061m.tif"

# Outputs
output_kriged    = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\kriged\fusion_based_kriged_0061m.tif"
output_variance  = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\kriged\fusion_based_kriging_variance.tif"
output_fusion_5m = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\kriged\fusion_downsampled_5m.tif"
output_variogram = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\kriged\variogram_fit_fusion.png"
output_comparison = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\kriged\kriging_vs_fusion_comparison.png"

NODATA_OUT = -9999.0

# =======================================================================
# KRIGING PARAMETERS
# =======================================================================

VARIOGRAM_MODEL = 'spherical'
BLOCK_SIZE_PX = 512
# CRITICAL: Search radius must be large enough to find points from 5m grid
# At 5m spacing, need ~100-200m radius to ensure coverage
SEARCH_RADIUS_M = 150.0  # Increased from 25m
MAX_NEIGHBOURS = 36
N_VARIOGRAM_SAMPLES = 3000
BLOCK_OVERLAP_PX = 32

# =======================================================================
# STEP 1: DOWNSAMPLE FUSION DTM TO 5m
# =======================================================================

def downsample_fusion_to_5m():
    print("\n" + "="*70)
    print("STEP 1: Downsampling fusion DTM to 5m baseline")
    print("="*70)
    
    with rasterio.open(fusion_dtm_file) as src:
        fusion_data = src.read(1)
        fusion_trans = src.transform
        fusion_crs = src.crs
        fusion_nodata = src.nodata
        pixel_size_fine = abs(fusion_trans.a)
        
        # Replace nodata with NaN for consistent handling
        if fusion_nodata is not None:
            fusion_data = np.where(fusion_data == fusion_nodata, np.nan, fusion_data)
        
        print(f"  Input (fusion):  {src.height} × {src.width} at {pixel_size_fine:.4f}m")
        print(f"  Input nodata:    {fusion_nodata}")
        print(f"  Input valid:     {(~np.isnan(fusion_data)).sum():,} "
              f"({(~np.isnan(fusion_data)).sum()/fusion_data.size*100:.1f}%)")
        
        # Calculate downsample factor to get ~5m resolution
        target_res = 5.0
        downsample_factor = int(np.round(target_res / pixel_size_fine))
        actual_res = pixel_size_fine * downsample_factor
        
        print(f"  Target resolution: {target_res}m")
        print(f"  Downsample factor: {downsample_factor}×")
        print(f"  Actual resolution: {actual_res:.4f}m")
        
        # Downsample using area averaging (better than nearest neighbor)
        # Take every Nth pixel
        fusion_5m = fusion_data[::downsample_factor, ::downsample_factor].copy()
        
        # Update transform
        fusion_5m_trans = fusion_trans * fusion_trans.scale(downsample_factor, downsample_factor)
        
        # Count valid pixels
        valid_fine = ~np.isnan(fusion_data)
        valid_5m = ~np.isnan(fusion_5m)
        
        print(f"\n  Output (5m):     {fusion_5m.shape[0]} × {fusion_5m.shape[1]} pixels")
        print(f"  Valid pixels:    {valid_5m.sum():,} ({valid_5m.sum()/fusion_5m.size*100:.1f}%)")
        print(f"  Elevation range: {np.nanmin(fusion_5m):.2f} to {np.nanmax(fusion_5m):.2f} m")
        
        # Save 5m version
        os.makedirs(os.path.dirname(output_fusion_5m), exist_ok=True)
        
        profile = {
            'driver': 'GTiff',
            'height': fusion_5m.shape[0],
            'width': fusion_5m.shape[1],
            'count': 1,
            'dtype': 'float32',
            'crs': fusion_crs,
            'transform': fusion_5m_trans,
            'nodata': NODATA_OUT,
            'compress': 'deflate',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256
        }
        
        data_out = np.where(np.isnan(fusion_5m), NODATA_OUT, fusion_5m).astype(np.float32)
        
        with rasterio.open(output_fusion_5m, 'w', **profile) as dst:
            dst.write(data_out, 1)
        
        file_size = os.path.getsize(output_fusion_5m) / 1024**2
        print(f"  ✓ Saved 5m baseline: {os.path.basename(output_fusion_5m)} ({file_size:.1f} MB)")
        
        return fusion_5m, fusion_5m_trans, fusion_crs, fusion_trans, fusion_data.shape


# =======================================================================
# STEP 2: EXTRACT POINT ARRAY FROM 5m DTM
# =======================================================================

def extract_points_from_5m(fusion_5m, fusion_5m_trans):
    print("\n" + "="*70)
    print("STEP 2: Extracting point array from 5m DTM")
    print("="*70)
    
    # Find valid pixels (not NaN - NODATA was already converted to NaN in downsampling)
    valid = ~np.isnan(fusion_5m)
    
    print(f"  Total pixels:    {fusion_5m.size:,}")
    print(f"  Valid pixels:    {valid.sum():,} ({valid.sum()/fusion_5m.size*100:.1f}%)")
    print(f"  NaN pixels:      {np.isnan(fusion_5m).sum():,}")
    
    if not valid.any():
        print("  ✗ No valid data in 5m DTM!")
        print("     Possible causes:")
        print("       - Fusion downsampling produced all NaN")
        print("       - NODATA value mismatch")
        return None, None, None
    
    rows, cols = np.where(valid)
    
    # Calculate world coordinates (pixel center)
    x_pts = fusion_5m_trans.c + (cols + 0.5) * fusion_5m_trans.a
    y_pts = fusion_5m_trans.f + (rows + 0.5) * fusion_5m_trans.e
    z_pts = fusion_5m[rows, cols]
    
    print(f"\n  Extracted points: {len(z_pts):,}")
    print(f"  X range:         {x_pts.min():.0f} to {x_pts.max():.0f} m")
    print(f"  Y range:         {y_pts.min():.0f} to {y_pts.max():.0f} m")
    print(f"  Z range:         {z_pts.min():.2f} to {z_pts.max():.2f} m")
    
    # Calculate point density
    x_extent = x_pts.max() - x_pts.min()
    y_extent = y_pts.max() - y_pts.min()
    area_km2 = (x_extent * y_extent) / 1e6
    density_per_km2 = len(z_pts) / area_km2
    
    # Average nearest neighbor distance (approximate)
    avg_spacing = np.sqrt(1e6 / density_per_km2)  # in meters
    
    print(f"\n  Point density analysis:")
    print(f"    Extent:          {x_extent/1000:.2f}km × {y_extent/1000:.2f}km")
    print(f"    Area:            {area_km2:.2f} km²")
    print(f"    Density:         {density_per_km2:.1f} points/km²")
    print(f"    Avg spacing:     ~{avg_spacing:.1f}m")
    print(f"    Initial search:  {SEARCH_RADIUS_M}m")
    
    if SEARCH_RADIUS_M < avg_spacing * 3:
        print(f"\n    ⚠️  WARNING: Search radius ({SEARCH_RADIUS_M}m) may be too small!")
        recommended_radius = avg_spacing * 5
        print(f"       Recommended: >{avg_spacing * 3:.0f}m (3× avg spacing)")
        print(f"       For reliable coverage, use {recommended_radius:.0f}m (5× avg spacing)")
        print(f"\n    Auto-adjusting search radius to {recommended_radius:.0f}m...")
        return x_pts, y_pts, z_pts, recommended_radius
    
    return x_pts, y_pts, z_pts, None


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
    
    # Save variogram plot
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
    ax.set_title(f'Variogram — {VARIOGRAM_MODEL.capitalize()} model (Fusion-derived)',
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
# STEP 4: BLOCK KRIGING
# =======================================================================

def krige_block(ok_model, x_pts, y_pts, z_pts, kdtree,
                block_x_grid, block_y_grid, search_radius_m, block_id=None, debug=False):
    """Krige a single block"""
    out_h, out_w = block_x_grid.shape
    block_centre_x = np.mean(block_x_grid)
    block_centre_y = np.mean(block_y_grid)
    
    block_half_diag = np.sqrt((block_x_grid.max() - block_x_grid.min())**2 +
                               (block_y_grid.max() - block_y_grid.min())**2) / 2
    search_r = search_radius_m + block_half_diag
    
    idx_nearby = kdtree.query_ball_point([block_centre_x, block_centre_y], r=search_r)
    
    if debug and len(idx_nearby) < 4:
        print(f"\n  Debug block {block_id}:")
        print(f"    Centre: ({block_centre_x:.0f}, {block_centre_y:.0f})")
        print(f"    Search radius: {search_r:.1f}m")
        print(f"    Points found: {len(idx_nearby)}")
        if len(idx_nearby) > 0:
            nearby_coords = np.column_stack([x_pts[idx_nearby], y_pts[idx_nearby]])
            dists = np.linalg.norm(nearby_coords - [block_centre_x, block_centre_y], axis=1)
            print(f"    Nearest point distance: {dists.min():.1f}m")
    
    if len(idx_nearby) < 4:
        # Not enough points - return NaN
        # This is expected at edges or where input data is sparse
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
        if debug or (block_id is not None and block_id < 3):
            print(f"\n  Block {block_id}: Constant elevation ({np.mean(z_local):.2f}m)")
        pred = np.full((out_h, out_w), np.mean(z_local), dtype=np.float32)
        var  = np.zeros((out_h, out_w), dtype=np.float32)
        return pred, var
    
    try:
        # Convert parameters to list (pykrige requirement when reusing variogram)
        params_list = ok_model.variogram_model_parameters.tolist() if hasattr(ok_model.variogram_model_parameters, 'tolist') else list(ok_model.variogram_model_parameters)
        
        ok_local = OrdinaryKriging(
            x_local, y_local, z_local,
            variogram_model=ok_model.variogram_model,
            variogram_parameters=params_list,  # Must be list, not numpy array
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
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)[:200]}"
        # Always report first few exceptions
        if block_id is not None and block_id < 3:
            print(f"\n  ⚠️  Block {block_id} kriging exception:")
            print(f"      {error_msg}")
            print(f"      Points used: {len(x_local)}")
            print(f"      Z range: {z_local.min():.2f} to {z_local.max():.2f}")
        nan_block = np.full((out_h, out_w), np.nan, dtype=np.float32)
        return nan_block, nan_block


def block_krige_scene(ok_model, x_pts, y_pts, z_pts, 
                      dst_trans, dst_h, dst_w, pixel_size, search_radius_m):
    print("\n" + "="*70)
    print("STEP 4: Block Kriging (5m → 0.061m)")
    print("="*70)
    
    print(f"  Output size:    {dst_h} × {dst_w} pixels")
    print(f"  Block size:     {BLOCK_SIZE_PX} × {BLOCK_SIZE_PX} px")
    print(f"  Search radius:  {search_radius_m}m")
    print(f"  Max neighbours: {MAX_NEIGHBOURS}")
    
    n_blocks_r = int(np.ceil(dst_h / BLOCK_SIZE_PX))
    n_blocks_c = int(np.ceil(dst_w / BLOCK_SIZE_PX))
    n_blocks   = n_blocks_r * n_blocks_c
    
    print(f"  Blocks:         {n_blocks_r} × {n_blocks_c} = {n_blocks} total")
    print(f"  Est. time:      ~{n_blocks * 3 / 60:.0f} min\n")
    
    from scipy.spatial import cKDTree
    coords = np.column_stack([x_pts, y_pts])
    kdtree = cKDTree(coords)
    
    # Test query to verify kdtree is working
    test_x = (dst_trans.c + dst_w/2 * dst_trans.a)
    test_y = (dst_trans.f + dst_h/2 * dst_trans.e)
    test_pts = kdtree.query_ball_point([test_x, test_y], r=search_radius_m)
    
    print(f"\n  KDTree test query:")
    print(f"    Test point:      ({test_x:.0f}, {test_y:.0f})")
    print(f"    Search radius:   {search_radius_m}m")
    print(f"    Points found:    {len(test_pts)}")
    
    if len(test_pts) == 0:
        print(f"\n  ✗ CRITICAL: Test query found 0 points!")
        print(f"     This suggests coordinate system mismatch.")
        print(f"\n  Input point extent:")
        print(f"    X: {x_pts.min():.0f} to {x_pts.max():.0f}")
        print(f"    Y: {y_pts.min():.0f} to {y_pts.max():.0f}")
        print(f"\n  Output grid extent:")
        print(f"    X: {dst_trans.c:.0f} to {dst_trans.c + dst_w * dst_trans.a:.0f}")
        print(f"    Y: {dst_trans.f + dst_h * dst_trans.e:.0f} to {dst_trans.f:.0f}")
        print(f"\n  These should overlap!")
        return None, None
    else:
        print(f"    ✓ KDTree working - found {len(test_pts)} points")
    
    pred_acc = np.zeros((dst_h, dst_w), dtype=np.float32)
    var_acc  = np.zeros((dst_h, dst_w), dtype=np.float32)
    weight_acc = np.zeros((dst_h, dst_w), dtype=np.float32)
    
    # Diagnostic counters
    blocks_processed = 0
    blocks_with_data = 0
    total_valid_predictions = 0
    blocks_no_points = 0  # Track blocks that found no input points
    blocks_exception = 0  # Track blocks that raised exceptions
    first_exception_msg = None
    first_failure_reported = False
    
    t_start = time.time()
    
    for ri in tqdm(range(n_blocks_r), desc="Row blocks"):
        for ci in range(n_blocks_c):
            blocks_processed += 1
            
            r0 = max(0, ri * BLOCK_SIZE_PX - BLOCK_OVERLAP_PX)
            r1 = min(dst_h, (ri + 1) * BLOCK_SIZE_PX + BLOCK_OVERLAP_PX)
            c0 = max(0, ci * BLOCK_SIZE_PX - BLOCK_OVERLAP_PX)
            c1 = min(dst_w, (ci + 1) * BLOCK_SIZE_PX + BLOCK_OVERLAP_PX)
            
            cols_idx = np.arange(c0, c1)
            rows_idx = np.arange(r0, r1)
            x_out = dst_trans.c + (cols_idx + 0.5) * dst_trans.a
            y_out = dst_trans.f + (rows_idx + 0.5) * dst_trans.e
            
            x_grid, y_grid = np.meshgrid(x_out, y_out)
            
            pred, var = krige_block(ok_model, x_pts, y_pts, z_pts, kdtree,
                                    x_grid, y_grid, search_radius_m,
                                    block_id=blocks_processed,
                                    debug=(blocks_processed < 3))
            
            # Track if this block has valid data
            block_valid = ~np.isnan(pred)
            if block_valid.any():
                blocks_with_data += 1
                total_valid_predictions += block_valid.sum()
            else:
                blocks_no_points += 1
            
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
    print(f"\n  Block diagnostics:")
    print(f"    Total blocks:       {blocks_processed}")
    print(f"    Blocks with data:   {blocks_with_data} ({blocks_with_data/blocks_processed*100:.1f}%)")
    print(f"    Blocks no points:   {blocks_no_points} ({blocks_no_points/blocks_processed*100:.1f}%)")
    print(f"    Valid predictions:  {total_valid_predictions:,}")
    
    if blocks_with_data == 0:
        print(f"\n  ✗ CRITICAL: No blocks produced valid predictions!")
        print(f"     Possible causes:")
        print(f"       - Search radius ({search_radius_m}m) too small for 5m point spacing")
        print(f"       - Input points too sparse")
        print(f"     Try increasing SEARCH_RADIUS_M to 200-500m")
    
    with np.errstate(invalid='ignore'):
        pred_final = np.where(weight_acc > 0, pred_acc / weight_acc, np.nan)
        var_final  = np.where(weight_acc > 0, var_acc  / weight_acc, np.nan)
    
    del pred_acc, var_acc, weight_acc
    
    valid = ~np.isnan(pred_final)
    print(f"  Output coverage: {valid.sum():,} / {dst_h*dst_w:,} pixels "
          f"({valid.sum()/(dst_h*dst_w)*100:.1f}%)")
    print(f"  Elevation range: {np.nanmin(pred_final):.2f} to {np.nanmax(pred_final):.2f} m")
    print(f"  Mean uncertainty: {np.nanmean(np.sqrt(var_final)):.4f} m")
    
    return pred_final.astype(np.float32), var_final.astype(np.float32)


# =======================================================================
# STEP 5: COMPARE WITH ORIGINAL FUSION
# =======================================================================

def compare_with_fusion(pred, var, fusion_trans, fusion_shape, fusion_crs):
    print("\n" + "="*70)
    print("STEP 5: Comparing kriging vs original fusion")
    print("="*70)
    
    # Read original fusion at same extent
    with rasterio.open(fusion_dtm_file) as src:
        fusion_orig = src.read(1)
        fusion_nodata = src.nodata
    
    # Replace nodata with NaN
    if fusion_nodata is not None:
        fusion_orig = np.where(fusion_orig == fusion_nodata, np.nan, fusion_orig)
    
    # Both should have same shape
    if fusion_orig.shape != pred.shape:
        print(f"  ⚠️  Shape mismatch: fusion {fusion_orig.shape} vs kriging {pred.shape}")
        # Trim to common size
        min_h = min(fusion_orig.shape[0], pred.shape[0])
        min_w = min(fusion_orig.shape[1], pred.shape[1])
        fusion_orig = fusion_orig[:min_h, :min_w]
        pred = pred[:min_h, :min_w]
        var = var[:min_h, :min_w]
    
    # Compare on valid pixels in both
    fusion_valid = ~np.isnan(fusion_orig)
    krig_valid = ~np.isnan(pred)
    both_valid = fusion_valid & krig_valid
    
    if not both_valid.any():
        print("  ✗ No overlapping valid pixels!")
        return
    
    diff = pred[both_valid] - fusion_orig[both_valid]
    
    print(f"  Compared pixels: {both_valid.sum():,}")
    print(f"  Fusion mean:     {fusion_orig[both_valid].mean():.3f} m")
    print(f"  Kriging mean:    {pred[both_valid].mean():.3f} m")
    print(f"  Mean diff:       {diff.mean():.4f} m")
    print(f"  Std diff:        {diff.std():.4f} m")
    print(f"  RMSE:            {np.sqrt((diff**2).mean()):.4f} m")
    print(f"  Max abs diff:    {np.abs(diff).max():.4f} m")
    
    # Downsample for plotting
    MAX_PLOT_DIM = 2000
    if max(pred.shape) > MAX_PLOT_DIM:
        ds = max(pred.shape) // MAX_PLOT_DIM + 1
        fusion_plot = fusion_orig[::ds, ::ds]
        pred_plot = pred[::ds, ::ds]
        both_plot = both_valid[::ds, ::ds]
    else:
        fusion_plot = fusion_orig
        pred_plot = pred
        both_plot = both_valid
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    extent = [0, pred.shape[1] * 0.061 / 1000, 0, pred.shape[0] * 0.061 / 1000]
    
    im1 = axes[0].imshow(fusion_plot, cmap='terrain', extent=extent)
    axes[0].set_title('Original Fusion DTM', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('km'); axes[0].set_ylabel('km')
    plt.colorbar(im1, ax=axes[0], label='Elevation (m)')
    
    im2 = axes[1].imshow(pred_plot, cmap='terrain', extent=extent)
    axes[1].set_title('Kriged DTM (from 5m fusion)', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('km'); axes[1].set_ylabel('km')
    plt.colorbar(im2, ax=axes[1], label='Elevation (m)')
    
    diff_map = np.full_like(pred_plot, np.nan)
    diff_map[both_plot] = pred_plot[both_plot] - fusion_plot[both_plot]
    im3 = axes[2].imshow(diff_map, cmap='RdBu_r', vmin=-0.5, vmax=0.5, extent=extent)
    axes[2].set_title('Difference: Kriging − Fusion', fontweight='bold', fontsize=12)
    axes[2].set_xlabel('km'); axes[2].set_ylabel('km')
    plt.colorbar(im3, ax=axes[2], label='m')
    
    plt.suptitle(f'Kriging vs Fusion Comparison\n'
                 f'RMSE = {np.sqrt((diff**2).mean()):.4f}m  |  '
                 f'Mean diff = {diff.mean():.4f}m',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_comparison), exist_ok=True)
    plt.savefig(output_comparison, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Comparison plot: {output_comparison}")


# =======================================================================
# MAIN
# =======================================================================

def run_fusion_based_kriging():
    print("="*70)
    print("FUSION-BASED KRIGING")
    print("Test: Does kriging improve upon fusion's implicit interpolation?")
    print("="*70)
    
    if not os.path.exists(fusion_dtm_file):
        print("\n✗ Fusion DTM not found. Run dsm_guided_dtm_fusion.py first.")
        return
    
    # Step 1: Downsample fusion to 5m
    fusion_5m, fusion_5m_trans, fusion_crs, fusion_trans, fusion_shape = downsample_fusion_to_5m()
    
    # Step 2: Extract points
    result = extract_points_from_5m(fusion_5m, fusion_5m_trans)
    if result[0] is None:
        print("\n✗ Cannot proceed - no valid points extracted from 5m DTM")
        return
    
    x_pts, y_pts, z_pts, recommended_radius = result
    
    # Use recommended radius if auto-calculated
    search_radius_to_use = recommended_radius if recommended_radius else SEARCH_RADIUS_M
    if recommended_radius:
        print(f"\n  Using auto-adjusted search radius: {search_radius_to_use:.0f}m")
    
    # Step 3: Fit variogram
    ok_model = fit_variogram(x_pts, y_pts, z_pts)
    
    # Step 4: Krige back to 0.061m
    result = block_krige_scene(ok_model, x_pts, y_pts, z_pts,
                               fusion_trans, fusion_shape[0], fusion_shape[1],
                               abs(fusion_trans.a), search_radius_to_use)
    
    if result[0] is None:
        print("\n✗ Kriging failed - see diagnostics above")
        return
    
    pred, var = result
    
    # Save outputs
    print("\n" + "="*70)
    print("Saving outputs")
    print("="*70)
    
    # Diagnostic check before saving
    pred_valid = ~np.isnan(pred)
    var_valid = ~np.isnan(var)
    
    print(f"\nPre-save diagnostic:")
    print(f"  Pred shape:       {pred.shape}")
    print(f"  Pred valid:       {pred_valid.sum():,} / {pred.size:,} ({pred_valid.sum()/pred.size*100:.1f}%)")
    print(f"  Pred range:       {np.nanmin(pred):.3f} to {np.nanmax(pred):.3f} m")
    print(f"  Var valid:        {var_valid.sum():,}")
    
    if pred_valid.sum() == 0:
        print(f"\n  ✗ ERROR: Kriging output is all NaN!")
        print(f"     Kriging failed to produce valid predictions.")
        print(f"     Check:")
        print(f"       - Input point density")
        print(f"       - Search radius")
        print(f"       - Kriging block size")
        return
    
    os.makedirs(os.path.dirname(output_kriged), exist_ok=True)
    
    profile = {
        'driver': 'GTiff',
        'height': pred.shape[0],
        'width': pred.shape[1],
        'count': 1,
        'dtype': 'float32',
        'crs': fusion_crs,
        'transform': fusion_trans,
        'nodata': NODATA_OUT,
        'compress': 'deflate',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'BIGTIFF': 'YES'
    }
    
    data_out = np.where(np.isnan(pred), NODATA_OUT, pred).astype(np.float32)
    with rasterio.open(output_kriged, 'w', **profile) as dst:
        dst.write(data_out, 1)
    
    print(f"  ✓ Kriged DTM: {os.path.basename(output_kriged)} "
          f"({os.path.getsize(output_kriged)/1024**2:.0f} MB)")
    
    data_out = np.where(np.isnan(var), NODATA_OUT, var).astype(np.float32)
    with rasterio.open(output_variance, 'w', **profile) as dst:
        dst.write(data_out, 1)
    
    print(f"  ✓ Variance: {os.path.basename(output_variance)} "
          f"({os.path.getsize(output_variance)/1024**2:.0f} MB)")
    
    # Step 5: Compare
    compare_with_fusion(pred, var, fusion_trans, fusion_shape, fusion_crs)
    
    print("\n" + "="*70)
    print("✅ FUSION-BASED KRIGING COMPLETE")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  • 5m baseline:   {output_fusion_5m}")
    print(f"  • Kriged DTM:    {output_kriged}")
    print(f"  • Variance:      {output_variance}")
    print(f"  • Variogram:     {output_variogram}")
    print(f"  • Comparison:    {output_comparison}")
    print(f"\nInterpretation:")
    print(f"  This comparison tests whether geostatistical kriging adds")
    print(f"  value over the fusion method's implicit bicubic interpolation")
    print(f"  when both start from the same 5m baseline data.")


if __name__ == "__main__":
    run_fusion_based_kriging()
