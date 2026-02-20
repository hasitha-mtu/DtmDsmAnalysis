"""
Find a tile that has valid DSM/DTM data

Tests multiple tiles to find one that works
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

tile_size = 1024
stride = 768

print("="*70)
print("FINDING TILES WITH VALID DSM/DTM DATA")
print("="*70)

# Get dimensions
with rasterio.open(orthophoto_file) as src:
    ortho_w, ortho_h = src.width, src.height

with rasterio.open(dsm_file) as src:
    dsm_nodata = src.nodata

with rasterio.open(dtm_file) as src:
    dtm_nodata = src.nodata

# Calculate grid
n_cols = (ortho_w - tile_size) // stride + 1
n_rows = (ortho_h - tile_size) // stride + 1
n_total = n_rows * n_cols

print(f"\nOrthophoto: {ortho_h} × {ortho_w}")
print(f"Grid: {n_rows} rows × {n_cols} cols = {n_total:,} total tiles")

# Test tiles
test_tiles = [
    0,                    # Top-left
    n_cols // 2,         # Top-center
    n_cols - 1,          # Top-right
    (n_rows // 2) * n_cols,  # Middle-left
    (n_rows // 2) * n_cols + n_cols // 2,  # Center
    (n_rows // 2) * n_cols + n_cols - 1,  # Middle-right
    (n_rows - 1) * n_cols,  # Bottom-left
    (n_rows - 1) * n_cols + n_cols // 2,  # Bottom-center
    n_total - 1,         # Bottom-right
]

# Add your current tile
test_tiles.insert(0, 359)

print(f"\nTesting {len(test_tiles)} tiles...")
print("="*70)

working_tiles = []

for tile_num in test_tiles:
    if tile_num >= n_total:
        continue
        
    # Calculate position
    row_idx = tile_num // n_cols
    col_idx = tile_num % n_cols
    row_start = row_idx * stride
    col_start = col_idx * stride
    
    window = Window(col_start, row_start, tile_size, tile_size)
    
    # Test DSM
    try:
        with rasterio.open(dsm_file) as src:
            dsm_tile = src.read(1, window=window)
            
            if dsm_nodata is not None:
                valid_dsm = np.sum(dsm_tile != dsm_nodata)
            else:
                valid_dsm = dsm_tile.size
                
            valid_dsm_pct = valid_dsm / dsm_tile.size * 100
    except:
        valid_dsm = 0
        valid_dsm_pct = 0
    
    # Test DTM
    try:
        with rasterio.open(dtm_file) as src:
            dtm_tile = src.read(1, window=window)
            
            if dtm_nodata is not None:
                valid_dtm = np.sum(dtm_tile != dtm_nodata)
            else:
                valid_dtm = dtm_tile.size
                
            valid_dtm_pct = valid_dtm / dtm_tile.size * 100
    except:
        valid_dtm = 0
        valid_dtm_pct = 0
    
    # Print result
    status = "✓" if (valid_dsm > 0 and valid_dtm > 0) else "✗"
    
    location_names = {
        0: "top-left",
        n_cols // 2: "top-center",
        n_cols - 1: "top-right",
        (n_rows // 2) * n_cols: "middle-left",
        (n_rows // 2) * n_cols + n_cols // 2: "CENTER",
        (n_rows // 2) * n_cols + n_cols - 1: "middle-right",
        (n_rows - 1) * n_cols: "bottom-left",
        (n_rows - 1) * n_cols + n_cols // 2: "bottom-center",
        n_total - 1: "bottom-right",
        359: "YOUR TILE"
    }
    
    location = location_names.get(tile_num, "")
    
    print(f"{status} Tile {tile_num:5d} {location:12s}: "
          f"DSM {valid_dsm_pct:5.1f}%, DTM {valid_dtm_pct:5.1f}% valid")
    
    if valid_dsm > 0 and valid_dtm > 0:
        working_tiles.append(tile_num)

print("\n" + "="*70)
print("SUMMARY:")
print("="*70)

if len(working_tiles) > 0:
    print(f"\n✓ Found {len(working_tiles)} tiles with valid data!")
    print(f"\n  Working tiles: {working_tiles[:10]}")  # Show first 10
    
    # Recommend center tile
    center_tile = (n_rows // 2) * n_cols + n_cols // 2
    if center_tile in working_tiles:
        print(f"\n  ✓ RECOMMENDED: Try tile {center_tile} (center of orthophoto)")
    elif len(working_tiles) > 0:
        print(f"\n  ✓ RECOMMENDED: Try tile {working_tiles[0]}")
    
    print(f"\n  To use a different tile:")
    print(f"     1. Update tile_number in simple_tile_extraction.py")
    print(f"     2. Make sure you have the image and mask for that tile")
else:
    print(f"\n✗ NO tiles with valid DSM/DTM data found!")
    print(f"\n  This means:")
    print(f"     • DSM/DTM files don't cover the orthophoto area")
    print(f"     • Or DSM/DTM are all nodata")
    print(f"     • Or there's a CRS/alignment issue")
    
    print(f"\n  Debug steps:")
    print(f"     1. Run diagnose_coverage.py for detailed analysis")
    print(f"     2. Check if DSM/DTM files are correct")
    print(f"     3. Verify CRS matches (all should be EPSG:32629)")
    print(f"     4. Check DSM/DTM bounds vs orthophoto bounds")

print("\n" + "="*70)
