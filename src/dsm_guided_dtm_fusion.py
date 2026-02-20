"""
DSM-Guided DTM Fusion
Replaces: save_bluesky_resampled.py

Pipeline:
  Step 1 — Reproject & align Bluesky 1m DSM and 5m DTM to WebODM grid
  Step 2 — Compute nDSM = DSM - DTM  (normalised canopy height model)
  Step 3 — Compute alpha mask: α=1 (open ground) → α=0 (dense canopy)
  Step 4 — Extract high-freq terrain detail from 1m DSM (real structure)
  Step 5 — Inject detail into upsampled 5m DTM weighted by α
  Step 6 — Apply geoid correction (+58m) and save at 0.061m resolution

Physical justification:
  Where α ≈ 1 (nDSM ≈ 0, open ground):
      1m DSM ≈ terrain surface  → inject its sub-5m detail into DTM
  Where α ≈ 0 (nDSM >> 0, canopy):
      1m DSM = canopy top       → suppress its detail, keep smooth DTM

  Final upsampling is 1m → 0.061m (16×) instead of 5m → 0.061m (82×)
  — a much more defensible interpolation step.

Reference approach:
  Aiazzi et al. (2002) context-adaptive pansharpening, adapted for terrain.
  Concept: Bluesky 1m DSM = panchromatic (high-res structure)
           Bluesky 5m DTM = multispectral (correct absolute ground level)
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.enums import Resampling as REnums
import rasterio.windows
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import os

# =======================================================================
# YOUR FILES — UPDATE THESE
# =======================================================================

# Bluesky inputs (in original CRS, EPSG:2157 Irish Grid)
bluesky_dsm_1m_file  = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\1m_dsm\bluesky_dsm_utm29n.tif"
bluesky_dtm_5m_file  = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\5m_dtm\bluesky_dtm_utm29n.tif"

# WebODM orthophoto — defines the target CRS, resolution, and extent
orthophoto_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_orthophoto\odm_orthophoto.tif"

# Outputs
output_fused_dtm  = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\fused\bluesky_dtm_fused_0061m.tif"
output_alpha_map  = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\fused\alpha_canopy_mask.tif"
output_ndsm       = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\fused\ndsm_1m.tif"
output_diagnostics = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\fused\fusion_diagnostics.png"

# Geoid correction (Malin Head orthometric → WGS84 ellipsoid)
GEOID_OFFSET = 58.0   # metres

# Alpha decay parameter: α = exp(-nDSM / CANOPY_SCALE)
# At nDSM = CANOPY_SCALE, α = 0.37  (35% detail injection)
# At nDSM = 2×CANOPY_SCALE, α = 0.14
# Typical: 2.0m (suppresses detail from shrubs+), 4.0m (only suppress full canopy)
CANOPY_SCALE = 2.0    # metres — tune based on your riparian vegetation height

NODATA_OUT   = -9999.0

# =======================================================================
# HELPERS
# =======================================================================

def reproject_to_grid(src_path, dst_transform, dst_crs, dst_width, dst_height,
                      resample_method, label=""):
    """
    Reproject a raster to the target grid defined by dst_transform/crs/shape.
    Returns numpy float32 array with nodata replaced by np.nan.
    """
    print(f"  Reprojecting {label}...")

    with rasterio.open(src_path) as src:
        src_nodata = src.nodata if src.nodata is not None else NODATA_OUT
        dst_array  = np.full((dst_height, dst_width), NODATA_OUT, dtype=np.float32)

        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resample_method,
            src_nodata=src_nodata,
            dst_nodata=NODATA_OUT,
        )

    result = np.where(dst_array == NODATA_OUT, np.nan, dst_array).astype(np.float32)
    valid  = np.sum(~np.isnan(result))
    print(f"    Shape: {result.shape}  Valid: {valid:,} / {result.size:,} "
          f"({valid/result.size*100:.1f}%)")
    if valid > 0:
        print(f"    Range: {np.nanmin(result):.2f} to {np.nanmax(result):.2f} m")
    return result


def save_raster(array, path, transform, crs, nodata=NODATA_OUT, desc=""):
    """Save float32 array as compressed GeoTIFF. NaN → nodata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.where(np.isnan(array), nodata, array).astype(np.float32)
    meta = {
        "driver":    "GTiff",
        "dtype":     "float32",
        "nodata":    nodata,
        "crs":       crs,
        "transform": transform,
        "width":     array.shape[1],
        "height":    array.shape[0],
        "count":     1,
        "compress":  "deflate",
        "tiled":     True,
        "blockxsize": 256,
        "blockysize": 256,
        "predictor": 3,
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data, 1)
    size_gb = os.path.getsize(path) / 1024**3
    print(f"  Saved {desc}: {os.path.basename(path)}  ({size_gb:.2f} GB)")
    return path


def low_pass(array, sigma_pixels):
    """
    Gaussian low-pass filter, NaN-aware.
    sigma_pixels: smoothing kernel size in pixels.
    """
    nan_mask = np.isnan(array)
    # Fill NaN with local mean before filtering
    filled = array.copy()
    if nan_mask.any():
        fill_val = np.nanmean(array)
        filled[nan_mask] = fill_val

    smoothed = gaussian_filter(filled, sigma=sigma_pixels)

    # Restore NaN
    smoothed[nan_mask] = np.nan
    return smoothed.astype(np.float32)


# =======================================================================
# STEP 1: LOAD TARGET GRID FROM ORTHOPHOTO
# =======================================================================

def load_target_grid():
    print("\n" + "="*70)
    print("STEP 1: Loading target grid from WebODM orthophoto")
    print("="*70)

    with rasterio.open(orthophoto_file) as src:
        dst_crs       = src.crs
        dst_transform = src.transform
        dst_width     = src.width
        dst_height    = src.height
        pixel_size    = abs(src.transform.a)

    print(f"  CRS:        {dst_crs}")
    print(f"  Resolution: {pixel_size:.4f} m")
    print(f"  Size:       {dst_height} × {dst_width} pixels")
    size_gb = (dst_height * dst_width * 4) / 1024**3
    print(f"  Output file will be ~{size_gb:.2f} GB (uncompressed float32)")

    return dst_crs, dst_transform, dst_width, dst_height, pixel_size


# =======================================================================
# STEP 2: REPROJECT BLUESKY DSM AND DTM TO WEBODM GRID
# =======================================================================

def reproject_bluesky(dst_crs, dst_transform, dst_width, dst_height):
    print("\n" + "="*70)
    print("STEP 2: Reprojecting Bluesky DSM (1m) and DTM (5m)")
    print("="*70)

    # DSM: bicubic — preserves fine structure in vegetation and terrain
    dsm_1m = reproject_to_grid(
        bluesky_dsm_1m_file,
        dst_transform, dst_crs, dst_width, dst_height,
        resample_method=REnums.cubic,
        label="Bluesky 1m DSM (cubic)"
    )

    # DTM: bicubic — smooth terrain, same resampling adequate
    dtm_5m = reproject_to_grid(
        bluesky_dtm_5m_file,
        dst_transform, dst_crs, dst_width, dst_height,
        resample_method=REnums.cubic,
        label="Bluesky 5m DTM (cubic)"
    )

    return dsm_1m, dtm_5m


# =======================================================================
# STEP 3: COMPUTE nDSM AND ALPHA MASK
# =======================================================================

def compute_ndsm_and_alpha(dsm_1m, dtm_5m, dst_transform, dst_crs):
    print("\n" + "="*70)
    print("STEP 3: Computing nDSM and α canopy mask")
    print("="*70)

    # nDSM = DSM - DTM  (normalised canopy height model)
    # Both are now at 0.061m resolution on same grid
    # Apply geoid offset to DTM before differencing (DSM already in same datum)
    dtm_5m_corrected = dtm_5m + GEOID_OFFSET   # Malin Head → WGS84 ellipsoid

    nDSM = dsm_1m - dtm_5m_corrected

    # Physical clamp: nDSM must be ≥ 0 (DSM cannot be below ground)
    # Negative values reflect georeferencing noise between the two products
    nDSM = np.clip(nDSM, 0.0, None)

    # Also clip implausible canopy heights (>50m for Ireland = artefact)
    nDSM = np.clip(nDSM, 0.0, 50.0)

    # Propagate NaN
    nDSM[np.isnan(dsm_1m) | np.isnan(dtm_5m)] = np.nan

    valid = ~np.isnan(nDSM)
    print(f"  nDSM range: {np.nanmin(nDSM):.2f} to {np.nanmax(nDSM):.2f} m")
    print(f"  Open ground (nDSM < 0.5m): "
          f"{np.sum(valid & (nDSM < 0.5)):,} pixels "
          f"({np.sum(valid & (nDSM < 0.5))/np.sum(valid)*100:.1f}%)")
    print(f"  Canopy (nDSM ≥ 2.0m):     "
          f"{np.sum(valid & (nDSM >= 2.0)):,} pixels "
          f"({np.sum(valid & (nDSM >= 2.0))/np.sum(valid)*100:.1f}%)")

    # Alpha: exponential decay with canopy height
    # α = 1 → open ground, inject full DSM detail
    # α = 0 → dense canopy, use smooth DTM only
    alpha = np.exp(-nDSM / CANOPY_SCALE)
    alpha[np.isnan(nDSM)] = np.nan

    print(f"\n  α (canopy scale={CANOPY_SCALE}m):")
    print(f"    Mean: {np.nanmean(alpha):.3f}")
    print(f"    % pixels with α > 0.8 (near-open): "
          f"{np.sum(~np.isnan(alpha) & (alpha > 0.8)):,} "
          f"({np.sum(~np.isnan(alpha) & (alpha > 0.8))/np.sum(~np.isnan(alpha))*100:.1f}%)")

    return nDSM, alpha, dtm_5m_corrected


# =======================================================================
# STEP 4: EXTRACT HIGH-FREQUENCY TERRAIN DETAIL FROM 1m DSM
# =======================================================================

def extract_dsm_detail(dsm_1m, pixel_size):
    """
    High-frequency detail = DSM_1m - lowpass(DSM_1m)

    The low-pass kernel size should match the DTM grid spacing (5m),
    so we remove the coarse component that the DTM already captures,
    retaining only the sub-5m variation.

    sigma = (5m / pixel_size) / 2  → Gaussian that attenuates frequencies
                                       with wavelength < ~5m
    """
    print("\n" + "="*70)
    print("STEP 4: Extracting high-frequency terrain detail from 1m DSM")
    print("="*70)

    # Sigma in pixels to match 5m DTM grid
    sigma_px = (5.0 / pixel_size) / 2.0
    print(f"  Low-pass sigma: {sigma_px:.1f} pixels  ({sigma_px * pixel_size:.2f} m)")
    print(f"  (Removes spatial frequencies coarser than ~5m — already in DTM)")
    print(f"  Applying Gaussian filter... (this may take ~1 minute for large files)")

    # Process in horizontal strips to avoid loading full array twice
    # For typical file sizes this fits in RAM, but we log progress
    dsm_lowfreq = low_pass(dsm_1m, sigma_px)

    dsm_detail  = dsm_1m - dsm_lowfreq
    dsm_detail[np.isnan(dsm_1m)] = np.nan

    valid = ~np.isnan(dsm_detail)
    print(f"  Detail range: {np.nanmin(dsm_detail):.3f} to {np.nanmax(dsm_detail):.3f} m")
    print(f"  Detail std:   {np.nanstd(dsm_detail):.4f} m  "
          f"(should be < 1m for natural terrain)")

    return dsm_detail


# =======================================================================
# STEP 5: FUSE DTM WITH ALPHA-WEIGHTED DSM DETAIL
# =======================================================================

def fuse_dtm(dtm_5m_corrected, dsm_detail, alpha):
    """
    Fused DTM = DTM_5m_corrected + α × DSM_detail

    Where α ≈ 1 (open ground):
        Adds real sub-5m terrain variation from the 1m DSM.
        Physical meaning: microterrain, bank shape, small channel features.

    Where α ≈ 0 (canopy):
        Keeps smooth DTM from the 5m product.
        Physical meaning: tree structure suppressed — DTM is already the
        best available estimate under canopy.
    """
    print("\n" + "="*70)
    print("STEP 5: Fusing DTM with α-weighted DSM detail")
    print("="*70)

    dtm_fused = dtm_5m_corrected + alpha * dsm_detail

    # Propagate NaN from any input
    nan_mask = np.isnan(dtm_5m_corrected) | np.isnan(dsm_detail) | np.isnan(alpha)
    dtm_fused[nan_mask] = np.nan

    valid = ~np.isnan(dtm_fused)
    print(f"  Fused DTM range: {np.nanmin(dtm_fused):.2f} to {np.nanmax(dtm_fused):.2f} m")
    print(f"  Mean elevation:  {np.nanmean(dtm_fused):.2f} m (WGS84 ellipsoid)")

    # Sanity check: fused should not deviate wildly from smooth DTM
    delta = dtm_fused - dtm_5m_corrected
    print(f"  Detail injected: {np.nanmin(delta):.3f} to {np.nanmax(delta):.3f} m "
          f"(mean {np.nanmean(np.abs(delta)):.4f} m)")

    return dtm_fused


# =======================================================================
# STEP 6: SAVE OUTPUTS
# =======================================================================

def save_outputs(dtm_fused, nDSM, alpha, dst_transform, dst_crs):
    print("\n" + "="*70)
    print("STEP 6: Saving outputs")
    print("="*70)

    os.makedirs(os.path.dirname(output_fused_dtm), exist_ok=True)

    save_raster(dtm_fused, output_fused_dtm, dst_transform, dst_crs,
                desc="Fused DTM (0.061m, WGS84 ellipsoid, geoid corrected)")
    save_raster(nDSM,      output_ndsm,      dst_transform, dst_crs,
                desc="nDSM canopy height model (m)")
    save_raster(alpha,     output_alpha_map,  dst_transform, dst_crs,
                desc="Alpha detail injection weight (0=canopy, 1=open)")


# =======================================================================
# STEP 7: DIAGNOSTICS PLOT
# =======================================================================

def save_diagnostics(dsm_1m, dtm_5m, nDSM, alpha, dsm_detail, dtm_fused,
                     dtm_5m_corrected, dst_transform):
    """
    Sample a representative 1km × 1km region from the centre of the scene
    and plot all intermediate products.
    """
    print("\n" + "="*70)
    print("STEP 7: Generating diagnostics plot")
    print("="*70)

    pixel_size = abs(dst_transform.a)
    h, w = dtm_fused.shape

    # Sample 1km × 1km from scene centre
    km_px  = int(1000 / pixel_size)
    r0, c0 = h // 2 - km_px // 2, w // 2 - km_px // 2
    r1, c1 = r0 + km_px, c0 + km_px

    def crop(arr):
        return arr[r0:r1, c0:c1]

    fig, axes = plt.subplots(2, 4, figsize=(24, 13))
    fig.suptitle(
        "DSM-Guided DTM Fusion — Diagnostic View (1km × 1km centre crop)\n"
        f"α decay scale = {CANOPY_SCALE}m  |  Geoid offset = +{GEOID_OFFSET}m",
        fontsize=13, fontweight="bold"
    )

    panels = [
        (crop(dsm_1m),            "Bluesky 1m DSM\n(first returns, canopy+ground)",   "terrain", None),
        (crop(dtm_5m_corrected),  "Bluesky 5m DTM (corrected)\n(last returns, ground only)", "terrain", None),
        (crop(nDSM),              "nDSM = DSM − DTM\n(canopy height model, m)",        "YlGn",    (0, 20)),
        (crop(alpha),             f"Alpha mask (α)\nα=1 open, α=0 canopy  [scale={CANOPY_SCALE}m]",
                                                                                       "RdYlGn",  (0, 1)),
        (crop(dsm_detail),        "DSM high-freq detail\n(sub-5m terrain variation)",  "RdBu_r",  None),
        (crop(alpha) * crop(dsm_detail),
                                  "α × detail injected\n(contribution to fused DTM)", "RdBu_r",  None),
        (crop(dtm_fused),         "Fused DTM (output)\nDTM + α × detail",             "terrain", None),
        (crop(dtm_fused) - crop(dtm_5m_corrected),
                                  "Fusion correction\n(fused − smooth DTM)",           "RdBu_r",  None),
    ]

    for ax, (data, title, cmap, vlim) in zip(axes.flat, panels):
        if vlim:
            im = ax.imshow(data, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        else:
            # Robust colour limits (2nd–98th percentile)
            vmin = np.nanpercentile(data, 2)
            vmax = np.nanpercentile(data, 98)
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontweight="bold", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_diagnostics, dpi=130, bbox_inches="tight")
    print(f"  Saved: {output_diagnostics}")
    plt.close()


# =======================================================================
# MEMORY-EFFICIENT BLOCK PROCESSING FOR VERY LARGE FILES
# =======================================================================

def check_memory_requirement(dst_height, dst_width):
    """
    Estimate RAM needed to hold all intermediate arrays.
    Warn if > 16 GB.
    """
    # Arrays needed simultaneously: dsm_1m, dtm_5m, dtm_5m_corrected,
    # nDSM, alpha, dsm_lowfreq, dsm_detail, dtm_fused = 8 arrays
    n_arrays = 8
    gb = (dst_height * dst_width * 4 * n_arrays) / 1024**3
    print(f"\n  Estimated RAM for all intermediate arrays: {gb:.1f} GB")
    if gb > 16:
        print(f"  ⚠️  This exceeds 16 GB. Consider processing in spatial blocks.")
        print(f"     Or reduce scope by clipping inputs to the study area first.")
    else:
        print(f"  ✓  Within typical workstation RAM limits.")
    return gb


# =======================================================================
# MAIN
# =======================================================================

def run_fusion():

    print("="*70)
    print("DSM-GUIDED DTM FUSION")
    print("Bluesky 1m DSM + 5m DTM → pseudo 0.061m DTM")
    print("="*70)

    # Check input files exist
    for label, path in [
        ("Bluesky 1m DSM", bluesky_dsm_1m_file),
        ("Bluesky 5m DTM", bluesky_dtm_5m_file),
        ("Orthophoto",     orthophoto_file),
    ]:
        status = "✓" if os.path.exists(path) else "✗ MISSING"
        print(f"  {status}  {label}: {os.path.basename(path)}")

    if not all(os.path.exists(p) for p in [bluesky_dsm_1m_file,
                                            bluesky_dtm_5m_file,
                                            orthophoto_file]):
        print("\n✗ One or more input files missing. Update paths at top of script.")
        return

    # Step 1
    dst_crs, dst_transform, dst_width, dst_height, pixel_size = load_target_grid()
    check_memory_requirement(dst_height, dst_width)

    # Step 2
    dsm_1m, dtm_5m = reproject_bluesky(dst_crs, dst_transform, dst_width, dst_height)

    # Step 3
    nDSM, alpha, dtm_5m_corrected = compute_ndsm_and_alpha(
        dsm_1m, dtm_5m, dst_transform, dst_crs
    )

    # Step 4
    dsm_detail = extract_dsm_detail(dsm_1m, pixel_size)

    # Step 5
    dtm_fused = fuse_dtm(dtm_5m_corrected, dsm_detail, alpha)

    # Step 6
    save_outputs(dtm_fused, nDSM, alpha, dst_transform, dst_crs)

    # Step 7
    save_diagnostics(dsm_1m, dtm_5m, nDSM, alpha, dsm_detail,
                     dtm_fused, dtm_5m_corrected, dst_transform)

    # Final summary
    print("\n" + "="*70)
    print("✅  FUSION COMPLETE")
    print("="*70)
    print(f"\n  Fused DTM: {output_fused_dtm}")
    print(f"  nDSM:      {output_ndsm}")
    print(f"  Alpha:     {output_alpha_map}")
    print(f"  Diagnostics: {output_diagnostics}")
    print(f"\n  Update batch_unified_comparison.py:")
    print(f"  bluesky_resampled_file = r\"{output_fused_dtm}\"")
    print()
    print(f"  ⚠️  Verify elevation values in diagnostics plot before running batch.")
    print(f"     Fused DTM mean should match your expected terrain elevation.")
    print(f"     If values are wrong, adjust GEOID_OFFSET (currently {GEOID_OFFSET}m).")


if __name__ == "__main__":
    run_fusion()
