"""
Diagnostic script: Check if DSM/DTM cover your tile's location
"""

import numpy as np
import rasterio
from rasterio.windows import Window

# =======================================================================
# YOUR FILES
# =======================================================================

orthophoto_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_orthophoto\odm_orthophoto.tif"
dsm_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_dem\dsm_aligned.tif"
dtm_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_dem\dtm_aligned.tif"

tile_number = 359
tile_size = 1024
stride = 768

print("="*70)
print("DIAGNOSTIC: DSM/DTM COVERAGE CHECK")
print("="*70)

# =======================================================================
# 1. CHECK ORTHOPHOTO
# =======================================================================

print("\n1. ORTHOPHOTO:")
with rasterio.open(orthophoto_file) as src:
    ortho_h, ortho_w = src.height, src.width
    ortho_bounds = src.bounds
    ortho_transform = src.transform
    
    print(f"   Dimensions: {ortho_h} × {ortho_w} pixels")
    print(f"   Bounds: {ortho_bounds}")
    print(f"   Resolution: {abs(ortho_transform.a):.3f}m")

# =======================================================================
# 2. CHECK DSM
# =======================================================================

print("\n2. DSM:")
with rasterio.open(dsm_file) as src:
    dsm_h, dsm_w = src.height, src.width
    dsm_bounds = src.bounds
    dsm_transform = src.transform
    dsm_nodata = src.nodata
    
    print(f"   Dimensions: {dsm_h} × {dsm_w} pixels")
    print(f"   Bounds: {dsm_bounds}")
    print(f"   Resolution: {abs(dsm_transform.a):.3f}m")
    print(f"   Nodata: {dsm_nodata}")
    
    # Read a sample
    dsm_data = src.read(1)
    valid_dsm = dsm_data != dsm_nodata if dsm_nodata is not None else np.ones_like(dsm_data, dtype=bool)
    
    print(f"   Valid pixels: {np.sum(valid_dsm):,} / {dsm_data.size:,} ({np.sum(valid_dsm)/dsm_data.size*100:.1f}%)")
    print(f"   Value range: {np.min(dsm_data[valid_dsm]):.2f} to {np.max(dsm_data[valid_dsm]):.2f}m")

# =======================================================================
# 3. CHECK DTM
# =======================================================================

print("\n3. DTM:")
with rasterio.open(dtm_file) as src:
    dtm_h, dtm_w = src.height, src.width
    dtm_bounds = src.bounds
    dtm_transform = src.transform
    dtm_nodata = src.nodata
    
    print(f"   Dimensions: {dtm_h} × {dtm_w} pixels")
    print(f"   Bounds: {dtm_bounds}")
    print(f"   Resolution: {abs(dtm_transform.a):.3f}m")
    print(f"   Nodata: {dtm_nodata}")
    
    # Read a sample
    dtm_data = src.read(1)
    valid_dtm = dtm_data != dtm_nodata if dtm_nodata is not None else np.ones_like(dtm_data, dtype=bool)
    
    print(f"   Valid pixels: {np.sum(valid_dtm):,} / {dtm_data.size:,} ({np.sum(valid_dtm)/dtm_data.size*100:.1f}%)")
    print(f"   Value range: {np.min(dtm_data[valid_dtm]):.2f} to {np.max(dtm_data[valid_dtm]):.2f}m")

# =======================================================================
# 4. CHECK TILE POSITION AND COVERAGE
# =======================================================================

print("\n4. TILE COVERAGE CHECK:")

# Calculate tile position
n_cols = (ortho_w - tile_size) // stride + 1
tile_row_idx = tile_number // n_cols
tile_col_idx = tile_number % n_cols
row_start = tile_row_idx * stride
col_start = tile_col_idx * stride

print(f"   Tile {tile_number}: row {tile_row_idx}, col {tile_col_idx}")
print(f"   Pixel position in orthophoto: ({row_start}, {col_start})")

# Get world coordinates
with rasterio.open(orthophoto_file) as src:
    left, top = rasterio.transform.xy(src.transform, row_start, col_start, offset='ul')
    right, bottom = rasterio.transform.xy(src.transform, row_start + tile_size, 
                                          col_start + tile_size, offset='ul')

tile_bounds = (left, bottom, right, top)
print(f"   Tile world bounds: {tile_bounds}")

# Check overlap with DSM
dsm_overlap_left = max(tile_bounds[0], dsm_bounds.left)
dsm_overlap_right = min(tile_bounds[2], dsm_bounds.right)
dsm_overlap_bottom = max(tile_bounds[1], dsm_bounds.bottom)
dsm_overlap_top = min(tile_bounds[3], dsm_bounds.top)

dsm_has_overlap = (dsm_overlap_left < dsm_overlap_right and 
                   dsm_overlap_bottom < dsm_overlap_top)

if dsm_has_overlap:
    overlap_area = (dsm_overlap_right - dsm_overlap_left) * (dsm_overlap_top - dsm_overlap_bottom)
    tile_area = (tile_bounds[2] - tile_bounds[0]) * (tile_bounds[3] - tile_bounds[1])
    overlap_pct = overlap_area / tile_area * 100
    
    print(f"\n   ✓ DSM overlaps tile!")
    print(f"     Overlap: {overlap_pct:.1f}%")
    print(f"     Overlap bounds: ({dsm_overlap_left:.2f}, {dsm_overlap_bottom:.2f}) to ({dsm_overlap_right:.2f}, {dsm_overlap_top:.2f})")
else:
    print(f"\n   ✗ DSM does NOT overlap tile!")
    print(f"     Tile bounds: {tile_bounds}")
    print(f"     DSM bounds: {dsm_bounds}")

# Check overlap with DTM
dtm_overlap_left = max(tile_bounds[0], dtm_bounds.left)
dtm_overlap_right = min(tile_bounds[2], dtm_bounds.right)
dtm_overlap_bottom = max(tile_bounds[1], dtm_bounds.bottom)
dtm_overlap_top = min(tile_bounds[3], dtm_bounds.top)

dtm_has_overlap = (dtm_overlap_left < dtm_overlap_right and 
                   dtm_overlap_bottom < dtm_overlap_top)

if dtm_has_overlap:
    overlap_area = (dtm_overlap_right - dtm_overlap_left) * (dtm_overlap_top - dtm_overlap_bottom)
    tile_area = (tile_bounds[2] - tile_bounds[0]) * (tile_bounds[3] - tile_bounds[1])
    overlap_pct = overlap_area / tile_area * 100
    
    print(f"\n   ✓ DTM overlaps tile!")
    print(f"     Overlap: {overlap_pct:.1f}%")
    print(f"     Overlap bounds: ({dtm_overlap_left:.2f}, {dtm_overlap_bottom:.2f}) to ({dtm_overlap_right:.2f}, {dtm_overlap_top:.2f})")
else:
    print(f"\n   ✗ DTM does NOT overlap tile!")
    print(f"     Tile bounds: {tile_bounds}")
    print(f"     DTM bounds: {dtm_bounds}")

# =======================================================================
# 5. SAMPLE TILE REGION FROM DSM/DTM
# =======================================================================

print("\n5. SAMPLING TILE REGION:")

window = Window(col_start, row_start, tile_size, tile_size)

# Sample DSM
print(f"\n   DSM at tile location:")
try:
    with rasterio.open(dsm_file) as src:
        dsm_tile = src.read(1, window=window)
        
        print(f"     Read window: col={col_start}, row={row_start}, width={tile_size}, height={tile_size}")
        print(f"     Data shape: {dsm_tile.shape}")
        print(f"     Nodata value: {src.nodata}")
        
        if src.nodata is not None:
            n_nodata = np.sum(dsm_tile == src.nodata)
            print(f"     Nodata pixels: {n_nodata:,} / {dsm_tile.size:,} ({n_nodata/dsm_tile.size*100:.1f}%)")
        
        valid = dsm_tile != src.nodata if src.nodata is not None else np.ones_like(dsm_tile, dtype=bool)
        
        if np.sum(valid) > 0:
            print(f"     Valid pixels: {np.sum(valid):,}")
            print(f"     Value range: {np.min(dsm_tile[valid]):.2f} to {np.max(dsm_tile[valid]):.2f}m")
        else:
            print(f"     ✗ ALL PIXELS ARE NODATA!")
            
except Exception as e:
    print(f"     ✗ ERROR reading DSM: {e}")

# Sample DTM
print(f"\n   DTM at tile location:")
try:
    with rasterio.open(dtm_file) as src:
        dtm_tile = src.read(1, window=window)
        
        print(f"     Read window: col={col_start}, row={row_start}, width={tile_size}, height={tile_size}")
        print(f"     Data shape: {dtm_tile.shape}")
        print(f"     Nodata value: {src.nodata}")
        
        if src.nodata is not None:
            n_nodata = np.sum(dtm_tile == src.nodata)
            print(f"     Nodata pixels: {n_nodata:,} / {dtm_tile.size:,} ({n_nodata/dtm_tile.size*100:.1f}%)")
        
        valid = dtm_tile != src.nodata if src.nodata is not None else np.ones_like(dtm_tile, dtype=bool)
        
        if np.sum(valid) > 0:
            print(f"     Valid pixels: {np.sum(valid):,}")
            print(f"     Value range: {np.min(dtm_tile[valid]):.2f} to {np.max(dtm_tile[valid]):.2f}m")
        else:
            print(f"     ✗ ALL PIXELS ARE NODATA!")
            
except Exception as e:
    print(f"     ✗ ERROR reading DTM: {e}")

# =======================================================================
# 6. DIAGNOSIS
# =======================================================================

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)

if not dsm_has_overlap or not dtm_has_overlap:
    print("\n✗ PROBLEM: DSM/DTM don't cover this tile's location!")
    print("   The tile is outside the DSM/DTM geographic extent.")
    print("\n   Solutions:")
    print("   1. Try a different tile (one that overlaps DSM/DTM)")
    print("   2. Check if DSM/DTM files are aligned with orthophoto")
    print("   3. Verify all files use same CRS (EPSG:32629)")
    
elif not dsm_has_overlap:
    print("\n✗ PROBLEM: DSM doesn't cover this tile!")
    
elif not dtm_has_overlap:
    print("\n✗ PROBLEM: DTM doesn't cover this tile!")
    
else:
    print("\n✓ Geographic overlap exists")
    print("   But tile region is all nodata in DSM/DTM")
    print("\n   Possible causes:")
    print("   1. Window coordinates are wrong")
    print("   2. This area wasn't processed by WebODM")
    print("   3. Low image coverage in this area")
    print("\n   Try a different tile number closer to the center of the orthophoto")

print("\n" + "="*70)
