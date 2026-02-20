"""
Kriging-based DTM Upsampling
Alternative to: dsm_guided_dtm_fusion.py and save_bluesky_resampled.py

⚠️  MEMORY WARNING: This script processes the entire orthophoto extent.
    If your orthophoto covers a large area (>5km × 5km), you will run out of RAM.
    
    SOLUTION: Clip orthophoto to just your study area BEFORE running this script:
    
    Using gdal (command line):
      gdalwarp -te xmin ymin xmax ymax -te_srs EPSG:32629 \\
               input_orthophoto.tif clipped_orthophoto.tif
    
    Or using QGIS:
      Raster → Extraction → Clip Raster by Extent
    
    Then update `orthophoto_file` path below to point to the clipped version.

Method: Ordinary Kriging (pykrige) — geostatistically optimal interpolation
  - Fits a variogram from the 5m DTM point observations
  - Predicts terrain elevation at 0.061m grid via kriging equations
  - Provides prediction variance (uncertainty) at every output pixel

Why Kriging is geostatistically superior to bicubic:
  - Bicubic: deterministic smooth function, ignores spatial structure of data
  - Kriging: minimises mean squared prediction error given the empirical
    spatial autocorrelation structure (variogram) of the actual DTM data
  - Kriging also quantifies prediction uncertainty — directly useful for
    reporting depth confidence intervals in the paper

Practical limitation:
  - Full-scene kriging at 0.061m resolution is computationally infeasible
    (would require predicting at ~2.5 billion output points)
  - Solution: block-kriging with a fixed variogram fitted once from a
    representative sample of the DTM data
  - Each output block is predicted independently using nearby input points
  - Block size and search radius are tunable

Variogram models available (set VARIOGRAM_MODEL below):
  - 'spherical'   : most common for terrain, range = decorrelation distance
  - 'exponential' : slower decay, appropriate for fractal terrain
  - 'gaussian'    : smooth, parabolic near origin — less common for terrain
  - 'linear'      : unbounded, for non-stationary terrain (use with caution)

Reference:
  Webster, R. & Oliver, M.A. (2007). Geostatistics for Environmental Scientists.
  2nd ed. Wiley. Chapter 4 (variogram modelling) and Chapter 5 (kriging).

  Cressie, N. (1993). Statistics for Spatial Data. Wiley-Interscience.

  Goovaerts, P. (1997). Geostatistics for Natural Resources Evaluation.
  Oxford University Press.

Output:
  bluesky_dtm_kriged_0061m.tif     — kriged elevation surface (m, WGS84 ellipsoid)
  bluesky_dtm_kriging_variance.tif — kriging variance (m²) — uncertainty map
  variogram_fit.png                — empirical variogram + fitted model (for paper)
  kriging_diagnostics.png          — comparison of interpolation outputs
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import rasterio.windows
from scipy.ndimage import gaussian_filter
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time
from tqdm import tqdm

# =======================================================================
# FILES — UPDATE THESE
# =======================================================================

bluesky_dtm_5m_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\5m_dtm\bluesky_dtm_utm29n.tif"
orthophoto_file     = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_orthophoto\odm_orthophoto.tif"

output_kriged    = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\kriged\bluesky_dtm_kriged_0061m.tif"
output_variance  = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\kriged\bluesky_dtm_kriging_variance.tif"
output_variogram = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\kriged\variogram_fit.png"
output_diag      = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\kriged\kriging_diagnostics.png"

GEOID_OFFSET = 58.0   # metres  (Malin Head → WGS84 ellipsoid)
NODATA_OUT   = -9999.0

# =======================================================================
# KRIGING PARAMETERS — tune these
# =======================================================================

# Variogram model
VARIOGRAM_MODEL = 'spherical'   # 'spherical' | 'exponential' | 'gaussian' | 'linear'

# Processing block size in OUTPUT pixels (0.061m grid)
# 1024 px ≈ 62m × 62m — good balance of speed vs. edge effects
# Increase to 2048 for smoother seams, decrease to 512 for faster testing
BLOCK_SIZE_PX = 1024

# Search radius for input DTM points (metres)
# Should be ≥ 2 × DTM grid spacing (5m) to guarantee at least 4 neighbours
# Increase to 30–50m in sparse areas
SEARCH_RADIUS_M = 25.0

# Number of 5m DTM points to use per block prediction (max neighbours)
# More points = better but slower. 16–64 is typical for terrain kriging
MAX_NEIGHBOURS = 36

# Number of DTM sample points used to fit the variogram
# 2000–5000 gives stable fit without long computation
N_VARIOGRAM_SAMPLES = 3000

# Overlap between blocks to avoid edge artefacts (pixels)
BLOCK_OVERLAP_PX = 64


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
# STEP 1: LOAD AND PREPARE 5m DTM
# =======================================================================

def load_dtm_as_points():
    """
    Load the 5m DTM, apply geoid correction, and return as arrays of
    (x_coords, y_coords, z_values) suitable for kriging.
    Also returns rasterio metadata for the target output grid.
    """
    print("\n" + "="*70)
    print("STEP 1: Loading 5m DTM and target grid")
    print("="*70)

    with rasterio.open(bluesky_dtm_5m_file) as src:
        dtm_data  = src.read(1).astype(np.float64)
        dtm_trans = src.transform
        dtm_crs   = src.crs
        dtm_nodata = src.nodata if src.nodata is not None else NODATA_OUT
        dtm_h, dtm_w = dtm_data.shape

    # Mask nodata
    valid = dtm_data != dtm_nodata
    dtm_data[~valid] = np.nan

    # Apply geoid correction
    dtm_data[valid] += GEOID_OFFSET

    # Build coordinate arrays for valid DTM pixels (cell centres)
    rows, cols = np.where(valid)
    x_pts = dtm_trans.c + (cols + 0.5) * dtm_trans.a   # easting
    y_pts = dtm_trans.f + (rows + 0.5) * dtm_trans.e   # northing
    z_pts = dtm_data[rows, cols]

    print(f"  DTM size:   {dtm_h} × {dtm_w} at 5m resolution")
    print(f"  Valid pts:  {len(z_pts):,}")
    print(f"  Elevation:  {z_pts.min():.2f} to {z_pts.max():.2f} m  "
          f"(after +{GEOID_OFFSET}m geoid)")

    # Load target grid parameters from orthophoto
    with rasterio.open(orthophoto_file) as src:
        dst_crs  = src.crs
        dst_trans = src.transform
        dst_h, dst_w = src.height, src.width
        pixel_size = abs(src.transform.a)

    print(f"\n  Target grid: {dst_h} × {dst_w} at {pixel_size:.4f}m resolution")

    return (x_pts, y_pts, z_pts,
            dtm_data, dtm_trans, dtm_crs,
            dst_crs, dst_trans, dst_h, dst_w, pixel_size)


# =======================================================================
# STEP 2: FIT VARIOGRAM FROM SAMPLE POINTS
# =======================================================================

def fit_variogram(x_pts, y_pts, z_pts):
    """
    Fit an empirical variogram using a random sample of the DTM points.

    The variogram γ(h) describes how elevation dissimilarity grows with
    lag distance h. Three parameters are estimated:
      - nugget (c₀): measurement error / micro-scale variation at h=0
      - sill   (c₀+c): total variance in the data
      - range  (a): distance beyond which points are uncorrelated

    For terrain, the range typically equals the river valley width or
    the characteristic length scale of the landform.
    """
    print("\n" + "="*70)
    print(f"STEP 2: Fitting {VARIOGRAM_MODEL} variogram")
    print("="*70)

    # Random subsample for variogram fitting
    n_pts = len(x_pts)
    n_sample = min(N_VARIOGRAM_SAMPLES, n_pts)
    rng = np.random.default_rng(seed=42)
    idx = rng.choice(n_pts, size=n_sample, replace=False)

    x_s, y_s, z_s = x_pts[idx], y_pts[idx], z_pts[idx]
    print(f"  Fitting variogram from {n_sample:,} sample points "
          f"(of {n_pts:,} total)...")

    t0 = time.time()
    ok = OrdinaryKriging(
        x_s, y_s, z_s,
        variogram_model=VARIOGRAM_MODEL,
        verbose=False,
        enable_plotting=False,
        nlags=20,               # number of lag bins
        weight=True,            # weight by number of pairs per lag bin
        coordinates_type='euclidean',
    )
    elapsed = time.time() - t0

    params = ok.variogram_model_parameters
    print(f"  Variogram fit ({elapsed:.1f}s):")
    if VARIOGRAM_MODEL in ('spherical', 'exponential', 'gaussian'):
        print(f"    Nugget: {params[2]:.4f} m²")
        print(f"    Sill:   {params[1] + params[2]:.4f} m²   "
              f"(partial sill: {params[1]:.4f})")
        print(f"    Range:  {params[0]:.2f} m")

    return ok, x_s, y_s, z_s


def plot_variogram(ok, x_s, y_s, z_s):
    """Save variogram plot for paper."""
    print(f"  Saving variogram plot: {output_variogram}")

    lags = ok.lags
    semivariance = ok.semivariance
    fitted = ok.variogram_function(ok.variogram_model_parameters, lags)

    params = ok.variogram_model_parameters

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(lags, semivariance, s=40, color='steelblue',
               zorder=3, label='Empirical variogram')
    ax.plot(lags, fitted, 'r-', linewidth=2,
            label=f'{VARIOGRAM_MODEL.capitalize()} model')

    if VARIOGRAM_MODEL in ('spherical', 'exponential', 'gaussian'):
        ax.axhline(params[1] + params[2], color='grey', linestyle='--',
                   alpha=0.6, label=f'Sill = {params[1]+params[2]:.3f} m²')
        ax.axvline(params[0], color='orange', linestyle='--',
                   alpha=0.6, label=f'Range = {params[0]:.1f} m')
        ax.axhline(params[2], color='purple', linestyle=':',
                   alpha=0.6, label=f'Nugget = {params[2]:.4f} m²')

    ax.set_xlabel('Lag distance (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Semivariance (m²)', fontsize=12, fontweight='bold')
    ax.set_title(f'Bluesky DTM Variogram — {VARIOGRAM_MODEL.capitalize()} model\n'
                 f'Crookstown catchment, Cork  |  n = {len(x_s):,} sample points',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(output_variogram), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_variogram, dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Variogram saved (include in paper as Figure X)")


# =======================================================================
# STEP 3: BLOCK-KRIGING PREDICTION
# =======================================================================

def build_spatial_index(x_pts, y_pts):
    """
    Build a simple spatial index (grid) to quickly retrieve
    nearby 5m DTM points for each output block.
    """
    from scipy.spatial import cKDTree
    coords = np.column_stack([x_pts, y_pts])
    tree = cKDTree(coords)
    return tree


def krige_block(ok_model, x_pts, y_pts, z_pts, kdtree,
                block_x_grid, block_y_grid):
    """
    Predict elevation at all output pixels in one block using
    ordinary kriging with a local neighbourhood search.

    Returns (predicted, variance) arrays of shape matching block_x_grid.
    """
    out_h, out_w = block_x_grid.shape

    # Query all output pixel centres at once for their nearest input neighbours
    out_xy = np.column_stack([block_x_grid.ravel(), block_y_grid.ravel()])
    block_centre_x = np.mean(block_x_grid)
    block_centre_y = np.mean(block_y_grid)

    # Find all input points within search radius of block centre
    # Use a slightly larger radius for block to include edge pixels
    block_half_diag = np.sqrt((block_x_grid.max() - block_x_grid.min())**2 +
                               (block_y_grid.max() - block_y_grid.min())**2) / 2
    search_r = SEARCH_RADIUS_M + block_half_diag

    idx_nearby = kdtree.query_ball_point(
        [block_centre_x, block_centre_y], r=search_r
    )

    if len(idx_nearby) < 4:
        # Fallback: return NaN if not enough neighbours
        nan_block = np.full((out_h, out_w), np.nan, dtype=np.float32)
        return nan_block, nan_block

    # Limit to MAX_NEIGHBOURS closest points
    if len(idx_nearby) > MAX_NEIGHBOURS:
        nearby_coords = np.column_stack([x_pts[idx_nearby], y_pts[idx_nearby]])
        dists = np.linalg.norm(nearby_coords - [block_centre_x, block_centre_y],
                               axis=1)
        top_n = np.argsort(dists)[:MAX_NEIGHBOURS]
        idx_nearby = [idx_nearby[i] for i in top_n]

    x_local = x_pts[idx_nearby]
    y_local = y_pts[idx_nearby]
    z_local = z_pts[idx_nearby]

    # Check we have enough unique elevations for kriging
    if np.std(z_local) < 1e-6:
        # Flat area — return constant (kriging undefined for zero variance)
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

        z_pred, z_var = ok_local.execute(
            'points',
            block_x_grid.ravel(),
            block_y_grid.ravel(),
        )

        pred = z_pred.data.reshape(out_h, out_w).astype(np.float32)
        var  = z_var.data.reshape(out_h, out_w).astype(np.float32)

        # Mask invalid predictions
        pred[pred == 0.0] = np.nan   # pykrige masked values
        var[var < 0] = 0.0           # negative variance = numerical noise

        return pred, var

    except Exception as e:
        nan_block = np.full((out_h, out_w), np.nan, dtype=np.float32)
        return nan_block, nan_block


def block_krige_scene(ok_model, x_pts, y_pts, z_pts,
                       dst_trans, dst_h, dst_w):
    """
    Tile the output scene into blocks and krige each independently.
    Overlapping regions are averaged to reduce edge artefacts.
    """
    print("\n" + "="*70)
    print("STEP 3: Block Kriging")
    print("="*70)

    pixel_size = abs(dst_trans.a)
    search_px  = int(np.ceil(SEARCH_RADIUS_M / pixel_size))

    print(f"  Output size:    {dst_h} × {dst_w} pixels")
    print(f"  Block size:     {BLOCK_SIZE_PX} × {BLOCK_SIZE_PX} px  "
          f"({BLOCK_SIZE_PX * pixel_size:.0f}m × {BLOCK_SIZE_PX * pixel_size:.0f}m)")
    print(f"  Search radius:  {SEARCH_RADIUS_M}m  ({search_px} px)")
    print(f"  Max neighbours: {MAX_NEIGHBOURS}")
    print(f"  Block overlap:  {BLOCK_OVERLAP_PX} px")

    # Memory check
    accumulator_gb = (dst_h * dst_w * 4 * 3) / 1024**3  # 3 float32 arrays
    print(f"\n  Memory required for accumulators: {accumulator_gb:.1f} GB")
    
    if accumulator_gb > 32:
        print(f"\n  ⚠️  ERROR: Scene is too large for available RAM.")
        print(f"     Your output grid is {dst_h} × {dst_w} pixels = "
              f"{dst_h * pixel_size / 1000:.1f}km × {dst_w * pixel_size / 1000:.1f}km")
        print(f"\n  Solutions:")
        print(f"     1. Clip orthophoto_file to just your study area (recommended)")
        print(f"     2. Process in spatial chunks (requires modifying this script)")
        print(f"     3. Use bicubic or DSM-fusion instead (much faster, less RAM)")
        print(f"\n  Kriging is best suited for small-to-medium study areas (<5km × 5km).")
        print(f"  For your {dst_h * pixel_size / 1000:.1f}km × {dst_w * pixel_size / 1000:.1f}km extent, "
              f"bicubic or DSM-fusion are more appropriate.")
        return None, None

    n_blocks_r = int(np.ceil(dst_h / BLOCK_SIZE_PX))
    n_blocks_c = int(np.ceil(dst_w / BLOCK_SIZE_PX))
    n_blocks   = n_blocks_r * n_blocks_c
    print(f"  Blocks:         {n_blocks_r} × {n_blocks_c} = {n_blocks} total")

    # Estimate time
    est_sec = n_blocks * 3.0  # rough 3s per block
    print(f"  Est. time:      ~{est_sec/60:.0f} min  "
          f"(varies with data density and hardware)\n")

    # Build spatial index
    kdtree = build_spatial_index(x_pts, y_pts)

    # Accumulators for overlap averaging (use float32 to save memory)
    pred_acc = np.zeros((dst_h, dst_w), dtype=np.float32)
    var_acc  = np.zeros((dst_h, dst_w), dtype=np.float32)
    weight_acc = np.zeros((dst_h, dst_w), dtype=np.float32)

    t_start = time.time()

    for ri in tqdm(range(n_blocks_r), desc="Row blocks"):
        for ci in range(n_blocks_c):

            # Block extent in output pixel coords (with overlap)
            r0 = max(0, ri * BLOCK_SIZE_PX - BLOCK_OVERLAP_PX)
            r1 = min(dst_h, (ri + 1) * BLOCK_SIZE_PX + BLOCK_OVERLAP_PX)
            c0 = max(0, ci * BLOCK_SIZE_PX - BLOCK_OVERLAP_PX)
            c1 = min(dst_w, (ci + 1) * BLOCK_SIZE_PX + BLOCK_OVERLAP_PX)

            # World coordinates of output pixel centres in this block
            cols_idx = np.arange(c0, c1)
            rows_idx = np.arange(r0, r1)
            x_out = dst_trans.c + (cols_idx + 0.5) * dst_trans.a
            y_out = dst_trans.f + (rows_idx + 0.5) * dst_trans.e

            x_grid, y_grid = np.meshgrid(x_out, y_out)

            pred, var = krige_block(
                ok_model, x_pts, y_pts, z_pts, kdtree, x_grid, y_grid
            )

            # Taper weights at block edges to reduce seam artefacts
            taper = np.ones_like(pred)
            taper_px = BLOCK_OVERLAP_PX
            if taper_px > 0:
                for i in range(taper_px):
                    w = (i + 1) / (taper_px + 1)
                    if r0 + i < r1:  taper[i, :] = min(taper[i, :].min(), w) if i < pred.shape[0] else taper[i, :]
                    if r1 - 1 - i >= r0:
                        ii = pred.shape[0] - 1 - i
                        if ii >= 0: taper[ii, :] = np.minimum(taper[ii, :], w)
                    if c0 + i < c1:  taper[:, i] = np.minimum(taper[:, i], w)
                    if c1 - 1 - i >= c0:
                        jj = pred.shape[1] - 1 - i
                        if jj >= 0: taper[:, jj] = np.minimum(taper[:, jj], w)

            valid = ~np.isnan(pred)
            pred_acc[r0:r1, c0:c1][valid] += pred[valid] * taper[valid]
            var_acc [r0:r1, c0:c1][valid] += var [valid] * taper[valid]
            weight_acc[r0:r1, c0:c1][valid] += taper[valid]

    elapsed = time.time() - t_start
    print(f"\n  Kriging complete in {elapsed/60:.1f} min")

    # Normalise
    with np.errstate(invalid='ignore'):
        pred_final = np.where(weight_acc > 0, pred_acc / weight_acc, np.nan)
        var_final  = np.where(weight_acc > 0, var_acc  / weight_acc, np.nan)

    # Free accumulator memory immediately
    del pred_acc, var_acc, weight_acc

    valid = ~np.isnan(pred_final)
    print(f"  Output coverage: {valid.sum():,} / {dst_h*dst_w:,} pixels "
          f"({valid.sum()/(dst_h*dst_w)*100:.1f}%)")
    print(f"  Elevation range: {np.nanmin(pred_final):.2f} to "
          f"{np.nanmax(pred_final):.2f} m")
    print(f"  Variance range:  {np.nanmin(var_final):.4f} to "
          f"{np.nanmax(var_final):.4f} m²")
    print(f"  Kriging std dev: {np.nanmean(np.sqrt(var_final)):.4f} m  "
          f"(mean prediction uncertainty)")

    return pred_final.astype(np.float32), var_final.astype(np.float32)


# =======================================================================
# STEP 4: DIAGNOSTICS PLOT
# =======================================================================

def save_diagnostics(pred, var, dtm_data, dtm_trans, dst_trans, pixel_size):
    """
    Compare kriging output against bicubic resampled DTM over a 1km region.
    """
    print("\n" + "="*70)
    print("STEP 4: Diagnostics")
    print("="*70)

    # 1km crop from centre of output
    km_px = int(1000 / pixel_size)
    ch, cw = pred.shape
    r0, c0 = ch//2 - km_px//2, cw//2 - km_px//2
    r1, c1 = r0 + km_px, c0 + km_px

    def crop(a): return a[r0:r1, c0:c1]

    p_crop  = crop(pred)
    v_crop  = crop(var)
    sd_crop = np.sqrt(v_crop)

    # Compute bicubic ONLY for the crop region to avoid 275GB allocation
    # Map output crop coords back to 5m DTM pixels
    # Output crop centre in world coords
    crop_centre_x = dst_trans.c + (c0 + km_px//2 + 0.5) * dst_trans.a
    crop_centre_y = dst_trans.f + (r0 + km_px//2 + 0.5) * dst_trans.e
    
    # Half-width in world units
    crop_half_m = (km_px * pixel_size) / 2.0 + 100  # +100m buffer for resampling
    
    # Corresponding 5m DTM rows/cols
    dtm_col_centre = int((crop_centre_x - dtm_trans.c) / dtm_trans.a)
    dtm_row_centre = int((crop_centre_y - dtm_trans.f) / dtm_trans.e)
    dtm_half_px = int(crop_half_m / abs(dtm_trans.a)) + 10
    
    dtm_r0 = max(0, dtm_row_centre - dtm_half_px)
    dtm_r1 = min(dtm_data.shape[0], dtm_row_centre + dtm_half_px)
    dtm_c0 = max(0, dtm_col_centre - dtm_half_px)
    dtm_c1 = min(dtm_data.shape[1], dtm_col_centre + dtm_half_px)
    
    dtm_crop_5m = dtm_data[dtm_r0:dtm_r1, dtm_c0:dtm_c1].copy()
    dtm_crop_5m = np.where(np.isnan(dtm_crop_5m), np.nanmean(dtm_crop_5m), dtm_crop_5m)
    
    try:
        # Resample this small crop
        from scipy.ndimage import zoom as scipy_zoom
        zoom_factor = abs(dtm_trans.a) / pixel_size
        bicubic_large = scipy_zoom(dtm_crop_5m, zoom=zoom_factor, order=3)
        
        # Extract the exact 1km region
        # Centre of bicubic_large in its own coords
        bc_centre_r = bicubic_large.shape[0] // 2
        bc_centre_c = bicubic_large.shape[1] // 2
        bc_r0 = max(0, bc_centre_r - km_px//2)
        bc_r1 = min(bicubic_large.shape[0], bc_r0 + km_px)
        bc_c0 = max(0, bc_centre_c - km_px//2)
        bc_c1 = min(bicubic_large.shape[1], bc_c0 + km_px)
        
        b_crop = bicubic_large[bc_r0:bc_r1, bc_c0:bc_c1]
        
        # Ensure shapes match (handle edge cases)
        if b_crop.shape != p_crop.shape:
            # Pad or trim to match
            min_r = min(b_crop.shape[0], p_crop.shape[0])
            min_c = min(b_crop.shape[1], p_crop.shape[1])
            b_crop = b_crop[:min_r, :min_c]
            p_crop = p_crop[:min_r, :min_c]
            v_crop = v_crop[:min_r, :min_c]
            sd_crop = sd_crop[:min_r, :min_c]
            km_px = min_r  # update for profile plot
        
        bicubic_available = True
    except MemoryError:
        print("  ⚠️  Not enough RAM for bicubic comparison in diagnostics (skipped)")
        b_crop = np.full_like(p_crop, np.nan)
        bicubic_available = False
    diff    = p_crop - b_crop

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle(
        f"Kriging vs Bicubic — 1km × 1km centre crop\n"
        f"Variogram: {VARIOGRAM_MODEL}  |  Search radius: {SEARCH_RADIUS_M}m  |  "
        f"Max neighbours: {MAX_NEIGHBOURS}",
        fontsize=12, fontweight='bold'
    )

    panels = [
        (b_crop if bicubic_available else None,
         "Bicubic resampled DTM (m)" if bicubic_available else "Bicubic unavailable\n(insufficient RAM)",
         "terrain",  None),
        (p_crop,  "Kriged DTM (m)",                             "terrain",  None),
        (diff if bicubic_available else None,
         "Difference: Kriging − Bicubic (m)" if bicubic_available else "—",
         "RdBu_r",   None),
        (sd_crop, "Kriging std dev (m)\n(prediction uncertainty)", "YlOrRd", (0, None)),
        (None, None, None, None),   # histogram placeholder
        (None, None, None, None),   # profile placeholder
    ]

    for i, (ax, (data, title, cmap, vlim)) in enumerate(zip(axes.flat, panels)):
        if data is None or title is None:
            ax.axis('off')
            if title == "—":  # Placeholder text for unavailable comparison
                ax.text(0.5, 0.5, 'Bicubic comparison\nskipped\n(RAM limit)',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=11, color='gray')
            continue
        vmin = np.nanpercentile(data, 2) if vlim is None else vlim[0]
        vmax = np.nanpercentile(data, 98) if (vlim is None or (vlim[1] is None)) else vlim[1]
        if vlim is not None and vlim[1] is None:
            vmax = np.nanpercentile(data, 98)
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Histogram comparison
    ax_hist = axes[1, 1]
    ax_hist.axis('on')
    valid_p = p_crop[~np.isnan(p_crop)]
    if bicubic_available:
        valid_b = b_crop[~np.isnan(b_crop)]
        if len(valid_b):
            ax_hist.hist(valid_b.ravel()[::100], bins=60, alpha=0.6,
                         color='steelblue', density=True,
                         label=f'Bicubic  μ={valid_b.mean():.2f}m')
    if len(valid_p):
        ax_hist.hist(valid_p.ravel()[::100], bins=60, alpha=0.6,
                     color='red', density=True,
                     label=f'Kriging  μ={valid_p.mean():.2f}m')
    ax_hist.set_xlabel('Elevation (m)', fontweight='bold')
    ax_hist.set_ylabel('Density', fontweight='bold')
    ax_hist.set_title('Elevation Distribution', fontweight='bold')
    ax_hist.legend(); ax_hist.grid(alpha=0.3)

    # Profile through centre row
    ax_prof = axes[1, 2]
    ax_prof.axis('on')
    mid_row = km_px // 2
    x_prof  = np.arange(km_px) * pixel_size
    if bicubic_available:
        ax_prof.plot(x_prof, b_crop[mid_row, :], 'b-', alpha=0.7,
                     linewidth=1.5, label='Bicubic')
    ax_prof.plot(x_prof, p_crop[mid_row, :], 'r-', alpha=0.7,
                 linewidth=1.5, label='Kriging')
    # Kriging confidence band
    ax_prof.fill_between(
        x_prof,
        p_crop[mid_row, :] - 1.96 * sd_crop[mid_row, :],
        p_crop[mid_row, :] + 1.96 * sd_crop[mid_row, :],
        alpha=0.2, color='red', label='Kriging 95% CI'
    )
    ax_prof.set_xlabel('Distance along profile (m)', fontweight='bold')
    ax_prof.set_ylabel('Elevation (m, WGS84)', fontweight='bold')
    ax_prof.set_title('Transect Profile (Centre Row)', fontweight='bold')
    ax_prof.legend(fontsize=9); ax_prof.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_diag), exist_ok=True)
    plt.savefig(output_diag, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Diagnostics: {output_diag}")

    # Print difference stats
    if bicubic_available:
        valid_diff = diff[~np.isnan(diff)]
        if len(valid_diff):
            print(f"\n  Kriging vs Bicubic (1km crop):")
            print(f"    Mean diff:   {valid_diff.mean():.4f} m")
            print(f"    Std diff:    {valid_diff.std():.4f} m")
            print(f"    Max abs diff: {np.abs(valid_diff).max():.4f} m")
            print(f"    RMSE:        {np.sqrt((valid_diff**2).mean()):.4f} m")


# =======================================================================
# MAIN
# =======================================================================

def run_kriging():

    print("="*70)
    print("KRIGING-BASED DTM UPSAMPLING")
    print(f"5m → 0.061m  |  Model: {VARIOGRAM_MODEL}  |  "
          f"Radius: {SEARCH_RADIUS_M}m  |  Neighbours: {MAX_NEIGHBOURS}")
    print("="*70)

    for label, path in [("Bluesky 5m DTM", bluesky_dtm_5m_file),
                         ("Orthophoto",    orthophoto_file)]:
        status = "✓" if os.path.exists(path) else "✗ MISSING"
        print(f"  {status}  {label}: {os.path.basename(path)}")

    if not all(os.path.exists(p) for p in [bluesky_dtm_5m_file, orthophoto_file]):
        print("\n✗ Input file(s) missing. Update paths at top of script.")
        return

    # Step 1
    (x_pts, y_pts, z_pts,
     dtm_data, dtm_trans, dtm_crs,
     dst_crs, dst_trans, dst_h, dst_w, pixel_size) = load_dtm_as_points()

    # Step 2
    ok_model, x_s, y_s, z_s = fit_variogram(x_pts, y_pts, z_pts)
    plot_variogram(ok_model, x_s, y_s, z_s)

    # Step 3
    result = block_krige_scene(
        ok_model, x_pts, y_pts, z_pts, dst_trans, dst_h, dst_w
    )
    
    if result[0] is None:
        print("\n✗ Kriging aborted due to memory constraints.")
        print("   See solutions printed above.")
        return

    pred, var = result

    # Save
    print("\n" + "="*70)
    print("Saving outputs")
    print("="*70)
    save_raster(pred, output_kriged,   dst_trans, dst_crs,
                desc="Kriged DTM (0.061m, WGS84 ellipsoid)")
    save_raster(var,  output_variance, dst_trans, dst_crs,
                desc="Kriging variance (m²)")

    # Step 4
    save_diagnostics(pred, var, dtm_data, dtm_trans, dst_trans, pixel_size)

    print("\n" + "="*70)
    print("✅  KRIGING COMPLETE")
    print("="*70)
    print(f"\n  Kriged DTM:  {output_kriged}")
    print(f"  Variance:    {output_variance}")
    print(f"  Variogram:   {output_variogram}")
    print(f"  Diagnostics: {output_diag}")
    print(f"\n  To use in batch_unified_comparison.py:")
    print(f"  bluesky_resampled_file = r\"{output_kriged}\"")
    print(f"\n  The variance file is your per-pixel uncertainty map.")
    print(f"  Divide by depth² to get a relative depth uncertainty for each tile.")


if __name__ == "__main__":
    run_kriging()
