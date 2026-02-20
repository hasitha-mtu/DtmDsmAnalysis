"""
Pre-Flight Check: DTM Comparison Workflow

Verifies all required files exist and have valid data before running comparison.
"""

import os
import rasterio
import numpy as np

print("="*70)
print("DTM COMPARISON - PRE-FLIGHT CHECK")
print("="*70)

# =======================================================================
# FILE PATHS
# =======================================================================

files_to_check = {
    'Orthophoto': r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_orthophoto\odm_orthophoto.tif",
    'DSM (water surface)': r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_dem\dsm_aligned.tif",
    'WebODM DTM': r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_dem\dtm_aligned.tif",
    'Bluesky DTM': r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\resampled\bluesky_dtm_0061m.tif",
    'Fusion DTM': r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\fused\bluesky_dtm_fused_0061m.tif",
    'Kriged DTM': r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\kriged\fusion_based_kriged_0061m.tif",
    'Working tiles list': "working_tiles.txt",
}

# =======================================================================
# CHECK FILES
# =======================================================================

print("\n1. FILE EXISTENCE CHECK")
print("-"*70)

all_exist = True
for name, filepath in files_to_check.items():
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    
    size_str = ""
    if exists and filepath.endswith('.tif'):
        size_mb = os.path.getsize(filepath) / 1024**2
        size_str = f" ({size_mb:.0f} MB)"
    
    print(f"  {status} {name:25s}: {os.path.basename(filepath)}{size_str}")
    
    if not exists:
        all_exist = False

if not all_exist:
    print("\n✗ Some files are missing!")
    print("\nTO FIX:")
    print("  - Bluesky DTM:  python save_bluesky_resampled.py")
    print("  - Fusion DTM:   python dsm_guided_dtm_fusion.py")
    print("  - Kriged DTM:   python kriging_fusion_based.py")
    print("  - Working tiles: python find_valid_from_files.py")
else:
    print("\n✓ All files exist")

# =======================================================================
# CHECK DATA COVERAGE
# =======================================================================

print("\n2. DATA COVERAGE CHECK")
print("-"*70)

dtm_files = {
    'WebODM': files_to_check['WebODM DTM'],
    'Bluesky': files_to_check['Bluesky DTM'],
    'Fusion': files_to_check['Fusion DTM'],
    'Kriged': files_to_check['Kriged DTM'],
}

coverage_ok = True

for name, filepath in dtm_files.items():
    if not os.path.exists(filepath):
        print(f"  ⊘ {name:8s}: File missing")
        coverage_ok = False
        continue
    
    try:
        with rasterio.open(filepath) as src:
            # Sample check (downsample for speed)
            sample = src.read(1)[::100, ::100]
            
            if src.nodata is not None:
                valid = (sample != src.nodata) & ~np.isnan(sample)
            else:
                valid = ~np.isnan(sample)
            
            coverage_pct = valid.sum() / sample.size * 100
            
            if coverage_pct < 10:
                status = "✗"
                msg = "UNUSABLE - No valid data"
                coverage_ok = False
            elif coverage_pct < 80:
                status = "⚠️ "
                msg = f"SPARSE - Limited coverage ({coverage_pct:.1f}%)"
            else:
                status = "✓"
                msg = f"GOOD - {coverage_pct:.1f}% coverage"
            
            print(f"  {status} {name:8s}: {msg}")
    
    except Exception as e:
        print(f"  ✗ {name:8s}: ERROR - {str(e)[:50]}")
        coverage_ok = False

if not coverage_ok:
    print("\n✗ Some DTMs have insufficient data coverage")
    print("\nTO FIX: Regenerate files with coverage < 80%")
else:
    print("\n✓ All DTMs have good coverage")

# =======================================================================
# CHECK SPATIAL ALIGNMENT
# =======================================================================

print("\n3. SPATIAL ALIGNMENT CHECK")
print("-"*70)

alignment_ok = True

# Check that all DTMs have same CRS
crs_list = []
res_list = []
names_list = []

for name, filepath in dtm_files.items():
    if os.path.exists(filepath):
        with rasterio.open(filepath) as src:
            crs_list.append(src.crs)
            res_list.append(abs(src.transform.a))
            names_list.append(name)

if len(set(str(c) for c in crs_list)) > 1:
    print(f"  ✗ CRS mismatch detected!")
    for name, crs in zip(names_list, crs_list):
        print(f"     {name}: {crs}")
    alignment_ok = False
else:
    print(f"  ✓ All DTMs use same CRS: {crs_list[0]}")

# Check resolutions are similar
res_array = np.array(res_list)
if res_array.max() - res_array.min() > 0.001:
    print(f"  ⚠️  Resolution mismatch detected:")
    for name, res in zip(names_list, res_list):
        print(f"     {name}: {res:.4f}m")
    print(f"  This may cause issues. Expected: 0.0610m")
else:
    print(f"  ✓ All DTMs have same resolution: {res_list[0]:.4f}m")

if not alignment_ok:
    print("\n✗ Spatial alignment issues detected")
else:
    print("\n✓ Spatial alignment OK")

# =======================================================================
# CHECK TILE LIST
# =======================================================================

print("\n4. TILE LIST CHECK")
print("-"*70)

tile_file = files_to_check['Working tiles list']
tiles_ok = False

if os.path.exists(tile_file):
    with open(tile_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    
    n_tiles = sum(1 for l in lines if l.startswith('Tile'))
    
    print(f"  ✓ Found {n_tiles} tiles in {tile_file}")
    
    if n_tiles < 10:
        print(f"  ⚠️  Only {n_tiles} tiles - is this correct?")
    else:
        tiles_ok = True
        print(f"  ✓ Tile count looks reasonable")
else:
    print(f"  ✗ {tile_file} not found")
    print(f"  Run: python find_valid_from_files.py")

# =======================================================================
# FINAL SUMMARY
# =======================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

all_checks = [all_exist, coverage_ok, alignment_ok, tiles_ok]

if all(all_checks):
    print("\n✅ ALL CHECKS PASSED")
    print("\nYou're ready to run:")
    print("  python batch_compare_all_dtms.py")
    print("\nThis will process all tiles and compare all 4 DTM methods.")
    print(f"Expected runtime: ~2-3 minutes")
else:
    print("\n⚠️  SOME CHECKS FAILED")
    print("\nFix the issues above before running batch_compare_all_dtms.py")
    
    if not all_exist:
        print("\nMissing files - check paths or regenerate")
    if not coverage_ok:
        print("\nInsufficient data coverage - regenerate DTM files")
    if not alignment_ok:
        print("\nSpatial alignment issues - verify CRS and resolution")
    if not tiles_ok:
        print("\nTile list issue - regenerate working_tiles.txt")

print("\n" + "="*70)
