"""
REALISTIC WebODM vs Bluesky Comparison
Acknowledges resolution limitations and provides honest assessment
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt

def realistic_comparison(webodm_file, bluesky_file, 
                        comparison_scale=10.0,
                        product_type='DSM'):
    """
    Honest comparison acknowledging resolution limitations
    
    Parameters:
    -----------
    webodm_file : str
        WebODM elevation file (6.1cm resolution)
    bluesky_file : str
        Bluesky elevation file (1m DSM or 5m DTM)
    comparison_scale : float
        Scale in meters for aggregation (default 10m = reasonable compromise)
    product_type : str
        'DSM' or 'DTM'
    """
    
    print("="*70)
    print(f"REALISTIC {product_type} COMPARISON: WebODM vs Bluesky")
    print("="*70)
    print("\n⚠️  IMPORTANT LIMITATIONS:")
    print("   • WebODM (6.1cm) vs Bluesky (1-5m) = huge resolution gap")
    print("   • Comparing 'smoothed WebODM' vs 'coarse Bluesky'")
    print("   • Results show TRENDS, not pixel-level accuracy")
    print("   • Narrow features (<5m) cannot be validated")
    print("="*70)
    
    # Load data
    print(f"\n1. Loading datasets...")
    with rasterio.open(webodm_file) as src:
        webodm_data_raw = src.read(1).astype(np.float32)
        webodm_res = abs(src.transform.a)
        webodm_crs = src.crs
        webodm_transform = src.transform
        webodm_nodata = src.nodata
        
        # CRITICAL FIX: Replace nodata with NaN
        if webodm_nodata is not None:
            webodm_data = np.where(webodm_data_raw == webodm_nodata, np.nan, webodm_data_raw)
        else:
            webodm_data = webodm_data_raw.copy()
    
    print(f"   WebODM: {webodm_data.shape} at {webodm_res:.3f}m resolution")
    print(f"   Valid WebODM pixels: {np.sum(~np.isnan(webodm_data)):,}")
    
    with rasterio.open(bluesky_file) as src:
        bluesky_data_raw = src.read(1).astype(np.float32)
        bluesky_res = abs(src.transform.a)
        bluesky_crs = src.crs
        bluesky_nodata = src.nodata
        
        # CRITICAL FIX: Mask nodata values
        if bluesky_nodata is not None:
            bluesky_data = np.where(bluesky_data_raw == bluesky_nodata, np.nan, bluesky_data_raw)
        else:
            bluesky_data = bluesky_data_raw.copy()
    
    print(f"   Bluesky: {bluesky_data.shape} at {bluesky_res:.1f}m resolution")
    print(f"   Resolution ratio: {bluesky_res/webodm_res:.1f}x coarser")
    print(f"   Valid Bluesky pixels: {np.sum(~np.isnan(bluesky_data)):,}")
    
    # Calculate how many WebODM pixels per comparison block
    pixels_per_block = int(comparison_scale / webodm_res)
    webodm_pixels_per_block = pixels_per_block ** 2
    
    print(f"\n2. Aggregation strategy:")
    print(f"   Comparison scale: {comparison_scale}m × {comparison_scale}m blocks")
    print(f"   Each block contains ~{webodm_pixels_per_block:,} WebODM pixels")
    print(f"   This smooths WebODM to match Bluesky's coarseness")
    
    # Aggregate WebODM to comparison scale
    print(f"\n3. Aggregating WebODM...")
    h, w = webodm_data.shape
    new_h = h // pixels_per_block
    new_w = w // pixels_per_block
    
    webodm_aggregated = np.full((new_h, new_w), np.nan, dtype=np.float32)
    
    # CRITICAL FIX: Properly mask nodata before aggregation
    if webodm_nodata is not None:
        webodm_masked = np.where(webodm_data == webodm_nodata, np.nan, webodm_data)
    else:
        webodm_masked = webodm_data.copy()
    
    for i in range(new_h):
        for j in range(new_w):
            block = webodm_masked[
                i*pixels_per_block:(i+1)*pixels_per_block,
                j*pixels_per_block:(j+1)*pixels_per_block
            ]
            # Only calculate mean if block has valid data
            valid_pixels = block[~np.isnan(block)]
            if len(valid_pixels) > 0:
                webodm_aggregated[i, j] = np.mean(valid_pixels)
    
    print(f"   Aggregated WebODM: {webodm_aggregated.shape}")
    print(f"   Valid blocks: {np.sum(~np.isnan(webodm_aggregated)):,}")
    
    # Crop Bluesky to matching size
    min_h = min(new_h, bluesky_data.shape[0])
    min_w = min(new_w, bluesky_data.shape[1])
    
    webodm_crop = webodm_aggregated[:min_h, :min_w]
    bluesky_crop = bluesky_data[:min_h, :min_w]
    
    print(f"   Comparison area: {webodm_crop.shape} blocks")
    
    # Calculate differences
    print(f"\n4. Calculating differences...")
    diff = webodm_crop - bluesky_crop
    
    valid_mask = ~(np.isnan(webodm_crop) | np.isnan(bluesky_crop))
    diff_valid = diff[valid_mask]
    
    if len(diff_valid) == 0:
        print("   ✗ ERROR: No overlapping valid data!")
        return None
    
    # Statistics
    print(f"\n" + "="*70)
    print(f"COMPARISON RESULTS ({comparison_scale}m SCALE)")
    print("="*70)
    
    print(f"\n📊 Elevation Statistics:")
    print(f"   WebODM  : {np.nanmean(webodm_crop):.2f} ± {np.nanstd(webodm_crop):.2f} m")
    print(f"   Bluesky : {np.nanmean(bluesky_crop):.2f} ± {np.nanstd(bluesky_crop):.2f} m")
    
    print(f"\n📏 Difference (WebODM - Bluesky):")
    mean_diff = np.mean(diff_valid)
    mae = np.mean(np.abs(diff_valid))
    rmse = np.sqrt(np.mean(diff_valid**2))
    std_diff = np.std(diff_valid)
    
    print(f"   Mean difference  : {mean_diff:.3f} m")
    print(f"   Mean abs error   : {mae:.3f} m")
    print(f"   RMSE             : {rmse:.3f} m")
    print(f"   Std deviation    : {std_diff:.3f} m")
    print(f"   Min difference   : {np.min(diff_valid):.3f} m")
    print(f"   Max difference   : {np.max(diff_valid):.3f} m")
    
    # Interpretation with HONESTY
    print(f"\n💡 INTERPRETATION:")
    
    if abs(mean_diff) < 0.20:
        print(f"   ✓ No major systematic offset (mean diff < 20cm)")
    elif abs(mean_diff) < 0.50:
        print(f"   ⚠ Moderate systematic offset: {mean_diff:.2f}m")
        print(f"     Could be: GPS bias, different vertical datums, or geoid model")
    else:
        print(f"   ✗ Large systematic offset: {mean_diff:.2f}m")
        print(f"     Investigate: CRS mismatch, elevation datum, georeferencing error")
    
    print(f"\n⚠️  CRITICAL LIMITATIONS:")
    print(f"   1. This compares {comparison_scale}m blocks, not individual pixels")
    print(f"   2. WebODM fine details are smoothed out")
    print(f"   3. Narrow river features (<{comparison_scale}m) cannot be assessed")
    
    if product_type == 'DTM':
        print(f"   4. TECHNOLOGY DIFFERENCE:")
        print(f"      • Bluesky LiDAR penetrates vegetation (sees ground)")
        print(f"      • WebODM photogrammetry blocked by leaves (interpolates ground)")
        print(f"      • Large differences under trees are EXPECTED")
    
    print(f"\n🎯 WHAT THIS TELLS YOU:")
    if abs(mean_diff) < 0.30 and rmse < 0.50:
        print(f"   • WebODM vertical georeferencing appears reasonable")
        print(f"   • No major systematic errors detected")
    else:
        print(f"   • Consider checking:")
        print(f"     - Vertical datum consistency (both using same ellipsoid/geoid?)")
        print(f"     - GCP elevation accuracy")
        print(f"     - WebODM camera parameters")
    
    print(f"\n🚫 WHAT THIS DOES NOT TELL YOU:")
    print(f"   • Accuracy of individual 6cm WebODM pixels")
    print(f"   • Accuracy along narrow river sections")
    print(f"   • Fine-scale topographic accuracy")
    print(f"   • Elevation under dense canopy (different technologies)")
    
    # Visualize
    print(f"\n5. Creating visualization...")
    create_honest_plots(webodm_crop, bluesky_crop, diff, 
                       comparison_scale, product_type, rmse, mean_diff)
    
    print(f"\n📄 FOR YOUR PAPER (HONEST VERSION):")
    print(f'   "As a coarse consistency check, WebODM-derived {product_type}')
    print(f'    was compared against Bluesky LiDAR at {comparison_scale}m scale,')
    print(f'    acknowledging the 82× resolution difference. This comparison')
    print(f'    revealed a mean offset of {mean_diff:.2f}m (RMSE {rmse:.2f}m),')
    
    if abs(mean_diff) < 0.30:
        print(f'    indicating reasonable vertical georeferencing. However,')
        print(f'    pixel-level validation was not feasible due to fundamental')
        print(f'    scale and technology differences between datasets."')
    else:
        print(f'    suggesting potential systematic offset requiring investigation.')
        print(f'    Direct pixel-level validation was not feasible due to')
        print(f'    fundamental scale and technology differences."')
    
    print("\n" + "="*70)
    
    return {
        'mean_diff': mean_diff,
        'mae': mae,
        'rmse': rmse,
        'std': std_diff,
        'blocks_compared': len(diff_valid),
        'comparison_scale': comparison_scale
    }

def create_honest_plots(webodm, bluesky, diff, scale, product_type, rmse, mean_diff):
    """Create visualization with honest labels about limitations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # WebODM
    im1 = axes[0, 0].imshow(webodm, cmap='terrain')
    axes[0, 0].set_title(f'WebODM {product_type}\n(Aggregated to {scale}m blocks)', 
                        fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], label='Elevation (m)', fraction=0.046)
    
    # Bluesky
    im2 = axes[0, 1].imshow(bluesky, cmap='terrain')
    axes[0, 1].set_title(f'Bluesky {product_type} Reference\n(Native resolution)', 
                        fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], label='Elevation (m)', fraction=0.046)
    
    # Difference
    vmax = max(abs(np.nanpercentile(diff, 5)), abs(np.nanpercentile(diff, 95)))
    im3 = axes[1, 0].imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1, 0].set_title(f'Difference Map\n(WebODM - Bluesky)', fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], label='Difference (m)', fraction=0.046)
    
    # Histogram
    diff_valid = diff[~np.isnan(diff)]
    axes[1, 1].hist(diff_valid, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    axes[1, 1].axvline(mean_diff, color='blue', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_diff:.2f}m')
    axes[1, 1].set_xlabel('Difference (m)', fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontweight='bold')
    axes[1, 1].set_title(f'Error Distribution\nRMSE: {rmse:.2f}m', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle(f'{product_type} Coarse Comparison: WebODM vs Bluesky\n'
                f'Scale: {scale}m blocks | NOT pixel-level validation',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{product_type.lower()}_coarse_comparison.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {product_type.lower()}_coarse_comparison.png")
    plt.close()

# if __name__ == "__main__":
    # # UPDATE THESE PATHS
    #
    # # Your WebODM DSM (high resolution - 6.1cm)
    # webodm_dsm = "C:\\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_dem\dsm_aligned.tif"
    #
    # # Bluesky DSM reference (1m resolution)
    # bluesky_dsm = "C:\\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\\bluesky\\1m_dsm\\bluesky_dsm_utm29n.tif"
    #
    # print("\n⚠️  DSM COARSE COMPARISON TOOL")
    # print("="*70)
    # print("Comparing WebODM (6.1cm) vs Bluesky DSM (1m)")
    # print("Resolution gap: 16x")
    # print("="*70)
    # print("\n📋 What this checks:")
    # print("  ✓ Vertical georeferencing (systematic offset?)")
    # print("  ✓ Large-scale elevation trends")
    # print("  ✓ Water surface elevations reasonable?")
    # print("\n📋 What this does NOT check:")
    # print("  ✗ Individual pixel accuracy")
    # print("  ✗ Fine details (<1m scale)")
    # print("  ✗ Narrow river features")
    # print("\n💡 Why DSM only?")
    # print("  • DSM compares surface-to-surface (both technologies measure same thing)")
    # print("  • 16x resolution gap is manageable (vs 82x for DTM)")
    # print("  • Validates water surface elevations (critical for flood forecasting!)")
    # print("  • DTM comparison problematic (different technologies + 82x gap)")
    # print("="*70)
    # print("\n")
    #
    # # Run comparison with realistic expectations
    # results = realistic_comparison(
    #     webodm_dsm,
    #     bluesky_dsm,
    #     comparison_scale=10.0,  # 10m blocks = reasonable compromise
    #     product_type='DSM'
    # )
    #
    # if results:
    #     print("\n✓ DSM comparison complete")
    #     print("\n🎯 INTERPRETATION:")
    #
    #     if abs(results['mean_diff']) < 0.30:
    #         print("  ✅ No major vertical offset detected")
    #         print("  ✅ Georeferencing appears reasonable")
    #         print("  ✅ Safe to proceed with WebODM DSM for water surface")
    #     elif abs(results['mean_diff']) < 0.60:
    #         print("  ⚠️  Moderate offset detected")
    #         print("  ⚠️  Check: CRS, vertical datum, GCP elevations")
    #         print("  ⚠️  May still be usable, but investigate cause")
    #     else:
    #         print("  ❌ Large offset detected!")
    #         print("  ❌ Likely issue: CRS mismatch, wrong datum, or GCP error")
    #         print("  ❌ Investigate before proceeding")
    #
    #     print("\n📄 FOR YOUR PAPER:")
    #     print('  "WebODM DSM was validated against Bluesky LiDAR DSM (1m)')
    #     print(f'   at 10m scale, showing mean offset of {results["mean_diff"]:.2f}m')
    #     print(f'   (RMSE {results["rmse"]:.2f}m). This coarse consistency check')
    #     print('   indicates [reasonable/problematic] vertical georeferencing.')
    #     print('   Pixel-level validation was not feasible due to 16× resolution')
    #     print('   mismatch, but the comparison validates water surface elevations')
    #     print('   at the scale relevant for discharge estimation."')
    #
    #     print("\n💡 REMEMBER:")
    #     print("  • This is NOT pixel-level validation")
    #     print("  • Consider RTK GPS survey for true accuracy assessment")
    #     print("  • Focus on water surface (DSM) - that's what matters for floods!")
    #     print("  • Skip DTM comparison - use Bluesky DTM directly in your workflow")

if __name__ == "__main__":
    # Bluesky DSM reference (1m resolution)
    bluesky_dsm = "C:\\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\\bluesky\\1m_dsm\\bluesky_dsm_utm29n.tif"
    with rasterio.open(bluesky_dsm) as src:
        print("CRS:", src.crs)
        print("Metadata:", src.tags())