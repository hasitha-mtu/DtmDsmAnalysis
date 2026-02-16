"""
DEBUG: Investigate Bluesky vs WebODM CRS and overlap issues
Find out why comparison is failing
"""

import rasterio
from rasterio.warp import transform_bounds
import numpy as np

def debug_comparison_issues(webodm_file, bluesky_file):
    """
    Investigate why comparison is showing crazy values
    """
    
    print("="*70)
    print("DEBUGGING WEBODM vs BLUESKY COMPARISON ISSUES")
    print("="*70)
    
    # Load WebODM
    print("\n[1/5] Loading WebODM DSM...")
    with rasterio.open(webodm_file) as src:
        webodm_data = src.read(1)
        webodm_crs = src.crs
        webodm_bounds = src.bounds
        webodm_transform = src.transform
        webodm_nodata = src.nodata
        
        print(f"  Shape: {webodm_data.shape}")
        print(f"  CRS: {webodm_crs}")
        print(f"  Bounds: {webodm_bounds}")
        print(f"  NoData value: {webodm_nodata}")
        
        # Check elevations
        if webodm_nodata is not None:
            valid_data = webodm_data[webodm_data != webodm_nodata]
        else:
            valid_data = webodm_data[~np.isnan(webodm_data)]
        
        print(f"\n  Elevation Statistics:")
        print(f"    Min: {np.min(valid_data):.2f} m")
        print(f"    Max: {np.max(valid_data):.2f} m")
        print(f"    Mean: {np.mean(valid_data):.2f} m")
        
        if np.mean(valid_data) < 0 or np.mean(valid_data) > 500:
            print(f"  ⚠️  WARNING: Elevations look wrong!")
            print(f"      Expected for Cork: 77-143m")
            print(f"      Got: {np.mean(valid_data):.0f}m")
    
    # Load Bluesky
    print("\n[2/5] Loading Bluesky DSM...")
    with rasterio.open(bluesky_file) as src:
        bluesky_data = src.read(1)
        bluesky_crs = src.crs
        bluesky_bounds = src.bounds
        bluesky_transform = src.transform
        bluesky_nodata = src.nodata
        
        print(f"  Shape: {bluesky_data.shape}")
        print(f"  CRS: {bluesky_crs}")
        print(f"  Bounds: {bluesky_bounds}")
        print(f"  NoData value: {bluesky_nodata}")
        
        # Check elevations
        if bluesky_nodata is not None:
            valid_data_b = bluesky_data[bluesky_data != bluesky_nodata]
        else:
            valid_data_b = bluesky_data[~np.isnan(bluesky_data)]
        
        if len(valid_data_b) > 0:
            print(f"\n  Elevation Statistics:")
            print(f"    Min: {np.min(valid_data_b):.2f} m")
            print(f"    Max: {np.max(valid_data_b):.2f} m")
            print(f"    Mean: {np.mean(valid_data_b):.2f} m")
            print(f"    Valid pixels: {len(valid_data_b):,}")
        else:
            print(f"\n  ⚠️  ERROR: All Bluesky data is NODATA!")
            print(f"      Total pixels: {bluesky_data.size:,}")
            print(f"      NoData pixels: {np.sum(bluesky_data == bluesky_nodata):,}")
    
    # Check CRS compatibility
    print("\n[3/5] Checking CRS compatibility...")
    if webodm_crs == bluesky_crs:
        print(f"  ✓ CRS match: {webodm_crs}")
    else:
        print(f"  ⚠️  CRS MISMATCH:")
        print(f"    WebODM: {webodm_crs}")
        print(f"    Bluesky: {bluesky_crs}")
    
    # Check if both are projected (not geographic)
    if webodm_crs.is_geographic:
        print(f"  ⚠️  WebODM is in GEOGRAPHIC coordinates (lat/lon)")
        print(f"     Should be PROJECTED (meters)")
    else:
        print(f"  ✓ WebODM is projected (meters)")
    
    if bluesky_crs.is_geographic:
        print(f"  ⚠️  Bluesky is in GEOGRAPHIC coordinates (lat/lon)")
        print(f"     Should be PROJECTED (meters)")
    else:
        print(f"  ✓ Bluesky is projected (meters)")
    
    # Check overlap
    print("\n[4/5] Checking geographic overlap...")
    
    # If CRS different, transform WebODM bounds to Bluesky CRS
    if webodm_crs != bluesky_crs:
        print(f"  Transforming WebODM bounds to Bluesky CRS for comparison...")
        webodm_bounds_transformed = transform_bounds(
            webodm_crs, bluesky_crs, *webodm_bounds
        )
    else:
        webodm_bounds_transformed = webodm_bounds
    
    print(f"\n  WebODM bounds (in Bluesky CRS):")
    print(f"    West:  {webodm_bounds_transformed[0]:.2f}")
    print(f"    South: {webodm_bounds_transformed[1]:.2f}")
    print(f"    East:  {webodm_bounds_transformed[2]:.2f}")
    print(f"    North: {webodm_bounds_transformed[3]:.2f}")
    
    print(f"\n  Bluesky bounds:")
    print(f"    West:  {bluesky_bounds[0]:.2f}")
    print(f"    South: {bluesky_bounds[1]:.2f}")
    print(f"    East:  {bluesky_bounds[2]:.2f}")
    print(f"    North: {bluesky_bounds[3]:.2f}")
    
    # Check for overlap
    overlap_west = max(webodm_bounds_transformed[0], bluesky_bounds[0])
    overlap_east = min(webodm_bounds_transformed[2], bluesky_bounds[2])
    overlap_south = max(webodm_bounds_transformed[1], bluesky_bounds[1])
    overlap_north = min(webodm_bounds_transformed[3], bluesky_bounds[3])
    
    if overlap_west < overlap_east and overlap_south < overlap_north:
        overlap_width = overlap_east - overlap_west
        overlap_height = overlap_north - overlap_south
        print(f"\n  ✓ Files DO overlap!")
        print(f"    Overlap area: {overlap_width:.0f}m × {overlap_height:.0f}m")
        
        # Calculate percentage overlap
        webodm_area = (webodm_bounds_transformed[2] - webodm_bounds_transformed[0]) * \
                     (webodm_bounds_transformed[3] - webodm_bounds_transformed[1])
        overlap_area = overlap_width * overlap_height
        overlap_pct = (overlap_area / webodm_area) * 100
        print(f"    Overlap: {overlap_pct:.1f}% of WebODM area")
    else:
        print(f"\n  ✗ NO OVERLAP! Files don't intersect geographically!")
        print(f"    WebODM is completely outside Bluesky coverage area")
    
    # Diagnose issues
    print("\n[5/5] Diagnosis & Recommendations...")
    print("="*70)
    
    issues_found = []
    
    # Check WebODM elevations
    if np.mean(valid_data) < 0 or np.mean(valid_data) > 500:
        issues_found.append("webodm_elevation")
        print("\n❌ ISSUE 1: WebODM elevations are wrong")
        print(f"   Expected: 77-143m for Cork")
        print(f"   Got: {np.mean(valid_data):.0f}m")
        print("\n   POSSIBLE CAUSES:")
        print("   • Wrong vertical datum in WebODM processing")
        print("   • Wrong ellipsoid reference")
        print("   • GCP elevations incorrect")
        print("   • Camera calibration issue")
        print("\n   FIX:")
        print("   1. Check your validation results - were elevations 77-143m?")
        print("   2. If so, this is a file loading issue")
        print("   3. If not, WebODM processing has problems")
    
    # Check Bluesky nodata
    if len(valid_data_b) == 0:
        issues_found.append("bluesky_nodata")
        print("\n❌ ISSUE 2: Bluesky file is all NODATA")
        print(f"   All {bluesky_data.size:,} pixels = -9999")
        print("\n   POSSIBLE CAUSES:")
        print("   • No overlap with WebODM area")
        print("   • Wrong Bluesky file loaded")
        print("   • CRS mismatch causing misalignment")
        print("   • File corruption")
        print("\n   FIX:")
        print("   1. Check if files overlap (see overlap check above)")
        print("   2. Verify you loaded correct Bluesky file for your area")
        print("   3. Check Bluesky CRS matches what you expect")
    
    # Check overlap
    if overlap_west >= overlap_east or overlap_south >= overlap_north:
        issues_found.append("no_overlap")
        print("\n❌ ISSUE 3: No geographic overlap")
        print("   Your WebODM area is outside Bluesky coverage")
        print("\n   FIX:")
        print("   • Contact Bluesky to confirm coverage area")
        print("   • Check if you have the right Bluesky tile/file")
        print("   • Verify WebODM coordinates are correct")
    
    # Check CRS mismatch
    if webodm_crs != bluesky_crs:
        issues_found.append("crs_mismatch")
        print("\n⚠️  ISSUE 4: CRS mismatch")
        print(f"   WebODM: {webodm_crs}")
        print(f"   Bluesky: {bluesky_crs}")
        print("\n   FIX:")
        print("   • Reproject one file to match the other")
        print("   • Use gdalwarp: gdalwarp -t_srs EPSG:32629 input.tif output.tif")
    
    if not issues_found:
        print("\n✓ No obvious issues detected")
        print("  But comparison results still look wrong!")
        print("  Manual investigation needed")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    
    print("\n1. FIX WebODM elevations first (most critical!)")
    print("   • Check if validated file showed 77-143m")
    print("   • If yes: file loading issue")
    print("   • If no: WebODM processing issue")
    
    print("\n2. FIX Bluesky overlap")
    print("   • Verify you have correct Bluesky tile for Crookstown, Cork")
    print("   • Check Bluesky coverage map")
    
    print("\n3. AFTER fixing both:")
    print("   • Re-run realistic_bluesky_comparison.py")
    print("   • Should see reasonable elevations (not -8000m vs -9999m!)")
    
    return {
        'webodm_mean_elev': np.mean(valid_data),
        'bluesky_valid_pixels': len(valid_data_b),
        'crs_match': webodm_crs == bluesky_crs,
        'has_overlap': overlap_west < overlap_east and overlap_south < overlap_north,
        'issues': issues_found
    }

if __name__ == "__main__":
    # UPDATE THESE PATHS
    webodm_dsm = "C:\\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_dem\dsm_aligned.tif"
    bluesky_dsm = "C:\\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\\bluesky\\1m_dsm\\205999-1_DSM_1_Shape.tif"
    
    results = debug_comparison_issues(webodm_dsm, bluesky_dsm)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"WebODM mean elevation: {results['webodm_mean_elev']:.0f}m "
          f"(expected: 77-143m)")
    print(f"Bluesky valid pixels: {results['bluesky_valid_pixels']:,}")
    print(f"CRS match: {results['crs_match']}")
    print(f"Geographic overlap: {results['has_overlap']}")
    print(f"Issues found: {len(results['issues'])}")
    
    if results['issues']:
        print("\n❌ CANNOT PROCEED until these are fixed:")
        for issue in results['issues']:
            print(f"  • {issue}")
    else:
        print("\n⚠️  Files appear OK but comparison failed anyway")
        print("   Manual debugging required")
