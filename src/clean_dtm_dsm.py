"""
Clean DTM/DSM by fixing height violations and outliers
Use this AFTER validation to prepare data for analysis
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt

def clean_dtm_dsm(dsm_file, dtm_file, 
                  output_dsm='dsm_cleaned.tif',
                  output_dtm='dtm_cleaned.tif',
                  output_ndsm='ndsm_cleaned.tif',
                  max_height=50.0):
    """
    Clean DSM/DTM data by:
    1. Fixing violations where DSM < DTM
    2. Removing outliers (heights > max_height)
    3. Generating clean nDSM
    
    Parameters:
    -----------
    dsm_file : str
        Path to validated DSM
    dtm_file : str
        Path to validated DTM
    output_dsm : str
        Path for cleaned DSM
    output_dtm : str
        Path for cleaned DTM (usually unchanged)
    output_ndsm : str
        Path for clean nDSM output
    max_height : float
        Maximum realistic height (default 50m for Irish vegetation)
    """
    
    print("="*70)
    print("DTM/DSM DATA CLEANING")
    print("="*70)
    
    # Load files
    print("\nLoading files...")
    with rasterio.open(dsm_file) as src:
        dsm = src.read(1).astype(np.float32)
        dsm_profile = src.profile
        dsm_profile.update(dtype=rasterio.float32)
    
    with rasterio.open(dtm_file) as src:
        dtm = src.read(1).astype(np.float32)
        dtm_profile = src.profile
        dtm_profile.update(dtype=rasterio.float32)
    
    print(f"  DSM shape: {dsm.shape}")
    print(f"  DTM shape: {dtm.shape}")
    
    # Calculate original nDSM
    original_ndsm = dsm - dtm
    
    # Statistics BEFORE cleaning
    violations_before = np.sum(original_ndsm < -0.1)
    outliers_before = np.sum(original_ndsm > max_height)
    total_pixels = dsm.size
    
    print(f"\n📊 BEFORE CLEANING:")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Violations (DSM<DTM): {violations_before:,} ({violations_before/total_pixels*100:.2f}%)")
    print(f"  Outliers (>{max_height}m): {outliers_before:,} ({outliers_before/total_pixels*100:.2f}%)")
    
    # Create cleaned versions
    dsm_cleaned = dsm.copy()
    dtm_cleaned = dtm.copy()  # Usually don't modify DTM
    
    # Fix violations: where DSM < DTM, set DSM = DTM
    violation_mask = dsm_cleaned < dtm_cleaned
    dsm_cleaned[violation_mask] = dtm_cleaned[violation_mask]
    
    print(f"\n🔧 FIXING VIOLATIONS:")
    print(f"  Set DSM = DTM at {np.sum(violation_mask):,} locations")
    
    # Calculate cleaned nDSM
    ndsm_cleaned = dsm_cleaned - dtm_cleaned
    
    # Remove extreme outliers
    outlier_mask = ndsm_cleaned > max_height
    ndsm_cleaned[outlier_mask] = np.nan
    
    print(f"\n🔧 REMOVING OUTLIERS:")
    print(f"  Set to NaN at {np.sum(outlier_mask):,} locations (>{max_height}m)")
    
    # Statistics AFTER cleaning
    violations_after = np.sum(ndsm_cleaned < -0.1)
    outliers_after = np.sum(ndsm_cleaned > max_height)
    valid_pixels = np.sum(~np.isnan(ndsm_cleaned))
    
    print(f"\n📊 AFTER CLEANING:")
    print(f"  Valid pixels: {valid_pixels:,} ({valid_pixels/total_pixels*100:.2f}%)")
    print(f"  Violations (DSM<DTM): {violations_after:,} (should be 0)")
    print(f"  Outliers (>{max_height}m): {outliers_after:,} (should be 0)")
    print(f"  Removed: {total_pixels - valid_pixels:,} pixels ({(total_pixels-valid_pixels)/total_pixels*100:.3f}%)")
    
    # Updated statistics
    print(f"\n📈 CLEANED nDSM STATISTICS:")
    print(f"  Mean height: {np.nanmean(ndsm_cleaned):.2f} m")
    print(f"  Median height: {np.nanmedian(ndsm_cleaned):.2f} m")
    print(f"  Max height: {np.nanmax(ndsm_cleaned):.2f} m")
    print(f"  Std deviation: {np.nanstd(ndsm_cleaned):.2f} m")
    
    # Save cleaned files
    print(f"\n💾 SAVING CLEANED FILES:")
    
    # Save cleaned DSM
    print(f"  Saving: {output_dsm}")
    with rasterio.open(output_dsm, 'w', **dsm_profile) as dst:
        dst.write(dsm_cleaned, 1)
    
    # Save cleaned DTM (usually unchanged, but for completeness)
    print(f"  Saving: {output_dtm}")
    with rasterio.open(output_dtm, 'w', **dtm_profile) as dst:
        dst.write(dtm_cleaned, 1)
    
    # Save nDSM
    print(f"  Saving: {output_ndsm}")
    ndsm_profile = dsm_profile.copy()
    ndsm_profile.update(nodata=np.nan)
    with rasterio.open(output_ndsm, 'w', **ndsm_profile) as dst:
        dst.write(ndsm_cleaned, 1)
    
    # Create before/after comparison plot
    print(f"\n📊 Creating comparison plot...")
    create_comparison_plot(original_ndsm, ndsm_cleaned, max_height)
    
    print("\n" + "="*70)
    print("✅ CLEANING COMPLETE!")
    print("="*70)
    print(f"\nCleaned files ready for analysis:")
    print(f"  • {output_dsm}")
    print(f"  • {output_dtm}")
    print(f"  • {output_ndsm}")
    print(f"\nRemoved {(total_pixels-valid_pixels)/total_pixels*100:.3f}% of data")
    print(f"({violations_before:,} violations + {outliers_before:,} outliers)")
    print("="*70)
    
    return output_dsm, output_dtm, output_ndsm

def create_comparison_plot(original, cleaned, max_height):
    """Create before/after comparison visualization"""
    
    # Downsample for visualization if needed
    max_dimension = 2000
    height, width = original.shape
    
    if height > max_dimension or width > max_dimension:
        downsample_factor = max(height // max_dimension, width // max_dimension, 1)
        print(f"  Dataset is large ({height} × {width})")
        print(f"  Downsampling by factor of {downsample_factor} for visualization...")
        
        original_plot = original[::downsample_factor, ::downsample_factor]
        cleaned_plot = cleaned[::downsample_factor, ::downsample_factor]
        
        print(f"  Visualization size: {original_plot.shape}")
    else:
        original_plot = original
        cleaned_plot = cleaned
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Original data
    # Original nDSM
    im1 = axes[0, 0].imshow(original_plot, cmap='YlGnBu', vmin=0, vmax=max_height)
    axes[0, 0].set_title('BEFORE: nDSM (with violations)', fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], label='Height (m)', fraction=0.046)
    
    # Violations map
    violations = original_plot < -0.1
    axes[0, 1].imshow(violations, cmap='Reds', vmin=0, vmax=1)
    axes[0, 1].set_title(f'BEFORE: Violations\n{np.sum(original < -0.1):,} pixels', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Outliers map
    outliers = original_plot > max_height
    axes[0, 2].imshow(outliers, cmap='Oranges', vmin=0, vmax=1)
    axes[0, 2].set_title(f'BEFORE: Outliers (>{max_height}m)\n{np.sum(original > max_height):,} pixels', fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Cleaned data
    # Cleaned nDSM
    im4 = axes[1, 0].imshow(cleaned_plot, cmap='YlGnBu', vmin=0, vmax=max_height)
    axes[1, 0].set_title('AFTER: nDSM (cleaned)', fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], label='Height (m)', fraction=0.046)
    
    # Cleaned violations map
    violations_after = cleaned_plot < -0.1
    axes[1, 1].imshow(violations_after, cmap='Reds', vmin=0, vmax=1)
    axes[1, 1].set_title(f'AFTER: Violations\n{np.sum(cleaned < -0.1):,} pixels (should be 0)', fontweight='bold')
    axes[1, 1].axis('off')
    
    # Histogram comparison (sample if too large)
    original_valid = original[~np.isnan(original)]
    cleaned_valid = cleaned[~np.isnan(cleaned)]
    
    if len(original_valid) > 1000000:
        orig_sample = np.random.choice(original_valid, size=1000000, replace=False)
        clean_sample = np.random.choice(cleaned_valid, size=min(1000000, len(cleaned_valid)), replace=False)
    else:
        orig_sample = original_valid
        clean_sample = cleaned_valid
    
    axes[1, 2].hist(orig_sample, bins=50, alpha=0.5, label='Before', edgecolor='black')
    axes[1, 2].hist(clean_sample, bins=50, alpha=0.5, label='After', edgecolor='black')
    axes[1, 2].set_xlabel('Height (m)')
    axes[1, 2].set_ylabel('Pixel Count')
    axes[1, 2].set_title('Height Distribution Comparison', fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)
    axes[1, 2].set_xlim(-5, max_height)
    
    plt.suptitle('Data Cleaning: Before vs After\n(Plots downsampled for visualization)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cleaning_comparison.png', dpi=150, bbox_inches='tight')  # Reduced DPI
    print(f"  Saved: cleaning_comparison.png")
    
    return fig

if __name__ == "__main__":
    # UPDATE THESE PATHS
    dsm_input = "C:/Users/AdikariAdikari/PycharmProjects/DtmDsmAnalysis/dataset/odm_dem/dsm_aligned.tif"
    dtm_input = "C:/Users/AdikariAdikari/PycharmProjects/DtmDsmAnalysis/dataset/odm_dem/dtm_aligned.tif"
    
    # Output paths
    dsm_output = "C:/Users/AdikariAdikari/PycharmProjects/DtmDsmAnalysis/dataset/odm_dem/dsm_cleaned.tif"
    dtm_output = "C:/Users/AdikariAdikari/PycharmProjects/DtmDsmAnalysis/dataset/odm_dem/dtm_cleaned.tif"
    ndsm_output = "C:/Users/AdikariAdikari/PycharmProjects/DtmDsmAnalysis/dataset/odm_dem/ndsm_cleaned.tif"
    
    # Run cleaning
    cleaned_dsm, cleaned_dtm, cleaned_ndsm = clean_dtm_dsm(
        dsm_input, 
        dtm_input,
        dsm_output,
        dtm_output,
        ndsm_output,
        max_height=50.0  # Adjust for your area (50m reasonable for Ireland)
    )
    
    print("\n💡 NEXT STEPS:")
    print("  1. Use the cleaned files for your analysis")
    print("  2. For water depth: depth = water_surface - dtm_cleaned")
    print("  3. For canopy analysis: use ndsm_cleaned")
    print("  4. Run: python simple_height_analysis.py with cleaned files")
    print("\n  NOTE: The 'cleaned' DTM is usually identical to input DTM")
    print("        Main changes are in DSM (fixed violations) and nDSM (no outliers)")
    
    plt.show()
