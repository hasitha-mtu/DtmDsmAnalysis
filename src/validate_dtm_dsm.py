"""
DTM and DSM Validation Script
Comprehensive quality checks before using for hydraulic analysis
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource


class DTMDSMValidator:
    def __init__(self, dsm_file, dtm_file):
        self.dsm_file = dsm_file
        self.dtm_file = dtm_file
        self.issues = []
        self.warnings = []
        self.passed_checks = []
        
    def validate_all(self):
        """Run all validation checks"""
        print("="*70)
        print("DTM/DSM VALIDATION REPORT")
        print("="*70)
        
        # 1. File loading
        if not self.check_file_loading():
            return False
        
        # 2. Spatial alignment
        self.check_spatial_alignment()
        
        # 3. Value ranges
        self.check_value_ranges()
        
        # 4. Height logic (DSM >= DTM)
        self.check_height_logic()
        
        # 5. Resolution consistency
        self.check_resolution()
        
        # 6. Nodata handling
        self.check_nodata()
        
        # 7. Outliers and anomalies
        self.check_outliers()
        
        # 8. Data completeness
        self.check_completeness()
        
        # Print summary
        self.print_summary()
        
        # Create validation visualization
        self.create_validation_plots()
        
        return len(self.issues) == 0
    
    def check_file_loading(self):
        """Validate files can be loaded"""
        print("\n[1/8] Checking file loading...")
        
        try:
            with rasterio.open(self.dsm_file) as src:
                dsm_raw = src.read(1, masked=True)
                # Convert MaskedArray to regular array with NaN for compatibility
                self.dsm = np.ma.filled(dsm_raw, np.nan).astype(np.float32)
                self.dsm_profile = src.profile
                self.dsm_transform = src.transform
                self.dsm_crs = src.crs
                self.dsm_bounds = src.bounds
                self.dsm_nodata = src.nodata
            
            print(f"  ✓ DSM loaded successfully: {self.dsm_file}")
            print(f"    Shape: {self.dsm.shape}")
            print(f"    CRS: {self.dsm_crs}")
            
        except Exception as e:
            self.issues.append(f"Failed to load DSM: {str(e)}")
            print(f"  ✗ ERROR: Cannot load DSM - {str(e)}")
            return False
        
        try:
            with rasterio.open(self.dtm_file) as src:
                dtm_raw = src.read(1, masked=True)
                # Convert MaskedArray to regular array with NaN for compatibility
                self.dtm = np.ma.filled(dtm_raw, np.nan).astype(np.float32)
                self.dtm_profile = src.profile
                self.dtm_transform = src.transform
                self.dtm_crs = src.crs
                self.dtm_bounds = src.bounds
                self.dtm_nodata = src.nodata
            
            print(f"  ✓ DTM loaded successfully: {self.dtm_file}")
            print(f"    Shape: {self.dtm.shape}")
            print(f"    CRS: {self.dtm_crs}")
            
        except Exception as e:
            self.issues.append(f"Failed to load DTM: {str(e)}")
            print(f"  ✗ ERROR: Cannot load DTM - {str(e)}")
            return False
        
        self.passed_checks.append("File loading")
        return True
    
    def check_spatial_alignment(self):
        """Check if DSM and DTM are spatially aligned"""
        print("\n[2/8] Checking spatial alignment...")
        
        # Check CRS
        if self.dsm_crs != self.dtm_crs:
            self.issues.append(f"CRS mismatch: DSM={self.dsm_crs}, DTM={self.dtm_crs}")
            print(f"  ✗ ERROR: CRS mismatch!")
            print(f"    DSM CRS: {self.dsm_crs}")
            print(f"    DTM CRS: {self.dtm_crs}")
        else:
            print(f"  ✓ CRS match: {self.dsm_crs}")
        
        # Check dimensions
        if self.dsm.shape != self.dtm.shape:
            height_diff = abs(self.dsm.shape[0] - self.dtm.shape[0])
            width_diff = abs(self.dsm.shape[1] - self.dtm.shape[1])
            
            self.issues.append(f"Shape mismatch: DSM={self.dsm.shape}, DTM={self.dtm.shape}")
            print(f"  ✗ ERROR: Shape mismatch!")
            print(f"    DSM: {self.dsm.shape}")
            print(f"    DTM: {self.dtm.shape}")
            print(f"    Difference: {height_diff} rows, {width_diff} cols")
            
            # Check if it's a small mismatch (common with WebODM)
            if height_diff <= 2 and width_diff <= 2:
                print(f"\n  💡 This is a MINOR mismatch (≤2 pixels) - very common with WebODM!")
                print(f"     Solution: Use align_rasters.py to crop to common extent")
                print(f"     This will remove ~{height_diff * self.dsm.shape[1] + width_diff * self.dsm.shape[0]:,} pixels (<0.01%)")
        else:
            print(f"  ✓ Dimensions match: {self.dsm.shape}")
        
        # Check transform (origin and pixel size)
        dsm_origin = (self.dsm_transform.c, self.dsm_transform.f)
        dtm_origin = (self.dtm_transform.c, self.dtm_transform.f)
        
        origin_diff = np.sqrt((dsm_origin[0] - dtm_origin[0])**2 + 
                             (dsm_origin[1] - dtm_origin[1])**2)
        
        if origin_diff > 1.0:  # 1 meter tolerance
            self.warnings.append(f"Origin offset: {origin_diff:.2f}m")
            print(f"  ⚠ WARNING: Origin offset of {origin_diff:.2f}m")
        else:
            print(f"  ✓ Origins aligned (offset: {origin_diff:.4f}m)")
        
        # Check pixel size
        dsm_pixel = (abs(self.dsm_transform.a), abs(self.dsm_transform.e))
        dtm_pixel = (abs(self.dtm_transform.a), abs(self.dtm_transform.e))
        
        pixel_diff = max(abs(dsm_pixel[0] - dtm_pixel[0]), 
                        abs(dsm_pixel[1] - dtm_pixel[1]))
        
        if pixel_diff > 0.01:
            self.warnings.append(f"Pixel size mismatch: {pixel_diff:.4f}")
            print(f"  ⚠ WARNING: Pixel size difference of {pixel_diff:.4f}")
            print(f"    DSM: {dsm_pixel}")
            print(f"    DTM: {dtm_pixel}")
        else:
            print(f"  ✓ Pixel sizes match: {dsm_pixel[0]:.4f} x {dsm_pixel[1]:.4f}")
        
        if len([i for i in self.issues if 'mismatch' in i.lower()]) == 0:
            self.passed_checks.append("Spatial alignment")
    
    def check_value_ranges(self):
        """Check if elevation values are sensible"""
        print("\n[3/8] Checking value ranges...")
        
        dsm_min = np.nanmin(self.dsm)
        dsm_max = np.nanmax(self.dsm)
        dtm_min = np.nanmin(self.dtm)
        dtm_max = np.nanmax(self.dtm)
        
        print(f"  DSM range: {dsm_min:.2f} to {dsm_max:.2f} m")
        print(f"  DTM range: {dtm_min:.2f} to {dtm_max:.2f} m")
        
        # Check for negative elevations (might be OK for areas below sea level)
        if dsm_min < -100 or dtm_min < -100:
            self.warnings.append(f"Very low elevations detected (DSM: {dsm_min:.2f}, DTM: {dtm_min:.2f})")
            print(f"  ⚠ WARNING: Very low elevations (<-100m)")
        
        # Check for extremely high elevations
        if dsm_max > 5000 or dtm_max > 5000:
            self.warnings.append(f"Very high elevations detected (DSM: {dsm_max:.2f}, DTM: {dtm_max:.2f})")
            print(f"  ⚠ WARNING: Very high elevations (>5000m)")
        
        # Check elevation range
        dsm_range = dsm_max - dsm_min
        dtm_range = dtm_max - dtm_min
        
        print(f"  DSM elevation range: {dsm_range:.2f} m")
        print(f"  DTM elevation range: {dtm_range:.2f} m")
        
        if dsm_range > 1000 or dtm_range > 1000:
            self.warnings.append(f"Very large elevation range (DSM: {dsm_range:.2f}, DTM: {dtm_range:.2f})")
            print(f"  ⚠ WARNING: Large elevation range (>1000m) - check for outliers")
        else:
            print(f"  ✓ Elevation ranges appear reasonable")
        
        self.passed_checks.append("Value ranges")
    
    def check_height_logic(self):
        """Check if DSM >= DTM (surface should be at or above terrain)"""
        print("\n[4/8] Checking height logic (DSM >= DTM)...")
        
        # Calculate difference
        diff = self.dsm - self.dtm
        
        # Count violations (where DSM < DTM)
        violations = np.sum(diff < -0.1)  # 10cm tolerance
        total_valid = np.sum(~np.isnan(diff))
        violation_pct = (violations / total_valid) * 100 if total_valid > 0 else 0
        
        if violations > 0:
            print(f"  ⚠ Found {violations:,} pixels where DSM < DTM ({violation_pct:.2f}%)")
            
            if violation_pct > 5:
                self.issues.append(f"High DSM<DTM violations: {violation_pct:.2f}%")
                print(f"  ✗ ERROR: >5% violations - data quality issue!")
                print(f"    This suggests problems with WebODM point cloud generation.")
                print(f"    Consider: increasing min-num-features, checking GCP quality,")
                print(f"    or re-processing with higher quality settings.")
            elif violation_pct > 1:
                self.warnings.append(f"DSM<DTM violations: {violation_pct:.2f}%")
                print(f"  ⚠ WARNING: >1% violations - minor data quality issue")
                print(f"    Likely causes: GPS noise, moving objects during flight,")
                print(f"    or sparse point cloud in some areas.")
                print(f"    For most applications, this is acceptable (<5%).")
            else:
                print(f"  ✓ Minor violations (<1%) - likely measurement noise")
        else:
            print(f"  ✓ All pixels satisfy DSM >= DTM")
        
        # Check nDSM statistics
        ndsm = np.maximum(diff, 0)  # Set negative to 0
        
        print(f"\n  Height above ground (nDSM) statistics:")
        print(f"    Mean: {np.nanmean(ndsm):.2f} m")
        
        # Safe median calculation
        valid_ndsm = ndsm[~np.isnan(ndsm)]
        if len(valid_ndsm) > 0:
            print(f"    Median: {np.median(valid_ndsm):.2f} m")
        else:
            print(f"    Median: N/A (no valid data)")
        
        print(f"    Max: {np.nanmax(ndsm):.2f} m")
        print(f"    Std: {np.nanstd(ndsm):.2f} m")
        
        # Check for unrealistic heights
        very_tall = np.sum(ndsm > 50)
        if very_tall > 0:
            tall_pct = (very_tall / total_valid) * 100
            self.warnings.append(f"{very_tall:,} pixels with heights >50m ({tall_pct:.2f}%)")
            print(f"  ⚠ WARNING: {very_tall:,} pixels with heights >50m")
            print(f"    These are likely processing artifacts (birds, planes, noise).")
            print(f"    For Irish vegetation, expect max ~15-25m for tall trees.")
        
        if violation_pct < 5:
            self.passed_checks.append("Height logic")
        
        # Additional insight for user
        if violation_pct > 1:
            print(f"\n  📊 Height violation analysis:")
            violation_mask = diff < -0.1
            violation_depths = diff[violation_mask]
            print(f"    Worst violation: {np.nanmin(violation_depths):.2f} m")
            print(f"    Mean violation depth: {np.nanmean(violation_depths):.2f} m")
            print(f"    This means DSM is up to {abs(np.nanmin(violation_depths)):.2f}m BELOW DTM in some places.")
    
    def check_resolution(self):
        """Check resolution consistency"""
        print("\n[5/8] Checking resolution...")
        
        pixel_width = abs(self.dsm_transform.a)
        pixel_height = abs(self.dsm_transform.e)
        
        print(f"  Pixel size: {pixel_width:.4f} x {pixel_height:.4f}")
        
        # Check if square pixels
        if abs(pixel_width - pixel_height) > 0.001:
            self.warnings.append(f"Non-square pixels: {pixel_width:.4f} x {pixel_height:.4f}")
            print(f"  ⚠ WARNING: Non-square pixels")
        else:
            print(f"  ✓ Square pixels")
        
        # Typical UAV resolutions: 1-10 cm
        if pixel_width < 0.001 or pixel_width > 1.0:
            self.warnings.append(f"Unusual resolution: {pixel_width:.4f}m")
            print(f"  ⚠ WARNING: Unusual resolution for UAV data")
        else:
            print(f"  ✓ Resolution appropriate for UAV data")
        
        self.passed_checks.append("Resolution")
    
    def check_nodata(self):
        """Check nodata value handling"""
        print("\n[6/8] Checking nodata values...")
        
        dsm_nodata_count = np.sum(np.isnan(self.dsm))
        dtm_nodata_count = np.sum(np.isnan(self.dtm))
        total_pixels = self.dsm.size
        
        dsm_nodata_pct = (dsm_nodata_count / total_pixels) * 100
        dtm_nodata_pct = (dtm_nodata_count / total_pixels) * 100
        
        print(f"  DSM nodata: {dsm_nodata_count:,} pixels ({dsm_nodata_pct:.2f}%)")
        print(f"  DTM nodata: {dtm_nodata_count:,} pixels ({dtm_nodata_pct:.2f}%)")
        
        if dsm_nodata_pct > 20 or dtm_nodata_pct > 20:
            self.warnings.append(f"High nodata percentage (DSM: {dsm_nodata_pct:.1f}%, DTM: {dtm_nodata_pct:.1f}%)")
            print(f"  ⚠ WARNING: >20% nodata - may indicate processing issues")
        elif dsm_nodata_pct > 5 or dtm_nodata_pct > 5:
            print(f"  ⚠ Some nodata present - normal for edge areas")
        else:
            print(f"  ✓ Minimal nodata")
        
        self.passed_checks.append("Nodata handling")
    
    def check_outliers(self):
        """Detect statistical outliers"""
        print("\n[7/8] Checking for outliers...")
        
        # Calculate nDSM
        ndsm = self.dsm - self.dtm
        
        # Get valid data only (no NaN)
        valid_ndsm = ndsm[~np.isnan(ndsm)]
        
        if len(valid_ndsm) == 0:
            print("  ⚠ WARNING: No valid data for outlier detection")
            return
        
        # Use IQR method for outlier detection
        q1 = np.percentile(valid_ndsm, 25)
        q3 = np.percentile(valid_ndsm, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        
        outliers_low = np.sum(valid_ndsm < lower_bound)
        outliers_high = np.sum(valid_ndsm > upper_bound)
        total_valid = len(valid_ndsm)
        
        outlier_pct = ((outliers_low + outliers_high) / total_valid) * 100
        
        print(f"  Outlier bounds: {lower_bound:.2f} to {upper_bound:.2f} m")
        print(f"  Low outliers: {outliers_low:,} pixels")
        print(f"  High outliers: {outliers_high:,} pixels")
        print(f"  Total outliers: {outlier_pct:.2f}%")
        
        if outlier_pct > 5:
            self.warnings.append(f"High outlier percentage: {outlier_pct:.2f}%")
            print(f"  ⚠ WARNING: >5% outliers - check for artifacts")
            print(f"    Common causes: birds in flight, planes, processing errors")
        else:
            print(f"  ✓ Outlier percentage acceptable")
        
        # Additional diagnostics for high outliers
        if outliers_high > 0:
            high_outlier_values = valid_ndsm[valid_ndsm > upper_bound]
            print(f"\n  High outlier details:")
            print(f"    Max outlier value: {np.max(high_outlier_values):.2f} m")
            print(f"    Mean of high outliers: {np.mean(high_outlier_values):.2f} m")
            if np.max(high_outlier_values) > 100:
                print(f"    ⚠ Extreme values detected (>100m) - likely data corruption")
        
        self.passed_checks.append("Outlier detection")
    
    def check_completeness(self):
        """Check data completeness"""
        print("\n[8/8] Checking data completeness...")
        
        # Check for large continuous nodata regions
        dsm_valid = ~np.isnan(self.dsm)
        dtm_valid = ~np.isnan(self.dtm)
        both_valid = dsm_valid & dtm_valid
        
        completeness = (np.sum(both_valid) / self.dsm.size) * 100
        
        print(f"  Overlapping valid data: {completeness:.2f}%")
        
        if completeness < 80:
            self.issues.append(f"Low data completeness: {completeness:.2f}%")
            print(f"  ✗ ERROR: <80% valid data overlap")
        elif completeness < 95:
            self.warnings.append(f"Moderate data completeness: {completeness:.2f}%")
            print(f"  ⚠ WARNING: <95% valid data overlap")
        else:
            print(f"  ✓ Excellent data completeness")
        
        if completeness > 80:
            self.passed_checks.append("Data completeness")
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        
        print(f"\n✓ PASSED CHECKS ({len(self.passed_checks)}):")
        for check in self.passed_checks:
            print(f"  • {check}")
        
        if self.warnings:
            print(f"\n⚠ WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if self.issues:
            print(f"\n✗ CRITICAL ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  • {issue}")
            print("\n❌ VALIDATION FAILED - Fix critical issues before proceeding!")
            
            # Provide helpful fix suggestions
            print("\n📋 HOW TO FIX:")
            for issue in self.issues:
                if "Shape mismatch" in issue and "39453" in issue and "39454" in issue:
                    print("  1. Run: python align_rasters.py")
                    print("     This will crop both files to (39453, 63600)")
                    print("     Then validate the aligned files")
                elif "Shape mismatch" in issue:
                    print("  • Shape mismatch: Run align_rasters.py to crop to common extent")
                elif "CRS mismatch" in issue:
                    print("  • CRS mismatch: Reproject with gdalwarp -t_srs EPSG:XXXX")
                elif "DSM<DTM violations" in issue:
                    print("  • Height violations: Regenerate DSM/DTM in WebODM with higher point density")
        else:
            print("\n✅ VALIDATION PASSED - Files are ready for analysis!")
        
        print("="*70)
    
    def create_validation_plots(self):
        """Create validation visualization"""
        print("\nCreating validation plots...")
        
        # Calculate nDSM
        ndsm = self.dsm - self.dtm
        ndsm_clean = np.maximum(ndsm, 0)
        
        # CRITICAL: Downsample for visualization to prevent memory errors
        # For large datasets (>10M pixels), downsample to ~2000x2000
        max_dimension = 2000
        height, width = self.dsm.shape
        
        if height > max_dimension or width > max_dimension:
            # Calculate downsampling factor
            downsample_factor = max(height // max_dimension, width // max_dimension, 1)
            
            print(f"  Dataset is large ({height} × {width})")
            print(f"  Downsampling by factor of {downsample_factor} for visualization...")
            
            # Downsample all arrays
            dsm_plot = self.dsm[::downsample_factor, ::downsample_factor]
            dtm_plot = self.dtm[::downsample_factor, ::downsample_factor]
            ndsm_plot = ndsm[::downsample_factor, ::downsample_factor]
            ndsm_clean_plot = ndsm_clean[::downsample_factor, ::downsample_factor]
            
            print(f"  Visualization size: {dsm_plot.shape}")
        else:
            dsm_plot = self.dsm
            dtm_plot = self.dtm
            ndsm_plot = ndsm
            ndsm_clean_plot = ndsm_clean
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. DSM
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(dsm_plot, cmap='terrain')
        ax1.set_title('DSM (Digital Surface Model)', fontweight='bold', fontsize=11)
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, label='Elevation (m)', fraction=0.046)
        
        # 2. DTM
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(dtm_plot, cmap='terrain')
        ax2.set_title('DTM (Digital Terrain Model)', fontweight='bold', fontsize=11)
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, label='Elevation (m)', fraction=0.046)
        
        # 3. nDSM
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(ndsm_clean_plot, cmap='YlGnBu', vmin=0)
        ax3.set_title('nDSM (Height Above Ground)', fontweight='bold', fontsize=11)
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, label='Height (m)', fraction=0.046)
        
        # 4. Height violations (DSM < DTM)
        ax4 = fig.add_subplot(gs[1, 0])
        violations = ndsm_plot < -0.1
        ax4.imshow(violations, cmap='Reds', vmin=0, vmax=1)
        ax4.set_title(f'Height Violations (DSM<DTM)\n{np.sum(ndsm < -0.1):,} pixels', 
                     fontweight='bold', fontsize=11)
        ax4.axis('off')
        
        # 5. Nodata map
        ax5 = fig.add_subplot(gs[1, 1])
        nodata_map = np.isnan(dsm_plot) | np.isnan(dtm_plot)
        ax5.imshow(nodata_map, cmap='gray_r', vmin=0, vmax=1)
        ax5.set_title(f'Nodata Regions\n{np.sum(np.isnan(self.dsm) | np.isnan(self.dtm)):,} pixels', 
                     fontweight='bold', fontsize=11)
        ax5.axis('off')
        
        # 6. Height histogram
        ax6 = fig.add_subplot(gs[1, 2])
        # Use full resolution ndsm for histogram
        ndsm_valid = ndsm_clean[~np.isnan(ndsm_clean)]
        # Downsample histogram data if too large
        if len(ndsm_valid) > 1000000:
            hist_sample = np.random.choice(ndsm_valid, size=1000000, replace=False)
        else:
            hist_sample = ndsm_valid
        ax6.hist(hist_sample, bins=50, edgecolor='black', alpha=0.7)
        ax6.set_xlabel('Height Above Ground (m)', fontsize=10)
        ax6.set_ylabel('Pixel Count', fontsize=10)
        ax6.set_title('Height Distribution', fontweight='bold', fontsize=11)
        ax6.grid(alpha=0.3)
        
        # 7. Hillshade (use downsampled data)
        ax7 = fig.add_subplot(gs[2, 0])
        ls = LightSource(azdeg=315, altdeg=45)
        try:
            hillshade = ls.hillshade(ndsm_clean_plot, vert_exag=2, dx=1, dy=1)
            ax7.imshow(hillshade, cmap='gray')
        except:
            # If hillshade fails, just show the ndsm
            ax7.imshow(ndsm_clean_plot, cmap='gray')
        ax7.set_title('Hillshade (Topographic Relief)', fontweight='bold', fontsize=11)
        ax7.axis('off')
        
        # 8. Height classes
        ax8 = fig.add_subplot(gs[2, 1])
        height_classes = np.digitize(ndsm_clean_plot, bins=[0, 2, 5, 10, 15, 100])
        im8 = ax8.imshow(height_classes, cmap='RdYlGn_r', vmin=1, vmax=5)
        ax8.set_title('Height Classes\n(0-2, 2-5, 5-10, 10-15, >15m)', 
                     fontweight='bold', fontsize=11)
        ax8.axis('off')
        
        # 9. Statistics text
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        stats_text = f"""
VALIDATION STATISTICS

DSM Range: {np.nanmin(self.dsm):.2f} to {np.nanmax(self.dsm):.2f} m
DTM Range: {np.nanmin(self.dtm):.2f} to {np.nanmax(self.dtm):.2f} m

nDSM (Height Above Ground):
  Mean: {np.nanmean(ndsm_clean):.2f} m
  Median: {np.median(ndsm_clean[~np.isnan(ndsm_clean)]):.2f} m
  Max: {np.nanmax(ndsm_clean):.2f} m
  
Data Quality:
  Resolution: {abs(self.dsm_transform.a):.4f} m/pixel
  Valid pixels: {np.sum(~np.isnan(ndsm)):,}
  Nodata: {np.sum(np.isnan(ndsm)):,} ({np.sum(np.isnan(ndsm))/ndsm.size*100:.1f}%)
  
Validation Result:
  ✓ Passed: {len(self.passed_checks)}
  ⚠ Warnings: {len(self.warnings)}
  ✗ Issues: {len(self.issues)}
  
Note: Plots downsampled for visualization
Full resolution data used for statistics
"""
        ax9.text(0.1, 0.9, stats_text, transform=ax9.transAxes,
                fontfamily='monospace', fontsize=9, verticalalignment='top')
        
        plt.suptitle('DTM/DSM Validation Report', fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig('validation_report.png', dpi=150, bbox_inches='tight')  # Reduced DPI to save memory
        print("  Saved: validation_report.png")
        
        return fig


# Main execution
if __name__ == "__main__":
    # Update these paths
    dsm_file = "C:\\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_dem\dsm_aligned.tif"
    dtm_file = "C:\\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_dem\dtm_aligned.tif"
    
    print("Starting DTM/DSM validation...\n")
    
    validator = DTMDSMValidator(dsm_file, dtm_file)
    is_valid = validator.validate_all()
    
    plt.show()
    
    if is_valid:
        print("\n🎉 Validation successful! Your DTM and DSM are ready to use.")
        print("   You can proceed with water depth calculations and hydraulic analysis.")
    else:
        print("\n⚠️  Validation found critical issues. Please review and fix before proceeding.")
        print("   Check the validation_report.png for detailed diagnostics.")
