"""
Scan tiles to find which ones have valid DSM/DTM data

Tests tiles systematically to find working ones
"""

import numpy as np
import rasterio
from rasterio.windows import from_bounds as window_from_bounds
import os

# =======================================================================
# YOUR FILES
# =======================================================================

orthophoto_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_orthophoto\odm_orthophoto.tif"
dsm_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_dem\dsm_aligned.tif"
dtm_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_dem\dtm_aligned.tif"
masks_dir = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\masks"

tile_size = 1024
stride = 768

print("="*70)
print("SCANNING FOR TILES WITH VALID DSM/DTM DATA")
print("="*70)

# Get dimensions
with rasterio.open(orthophoto_file) as src:
    ortho_w, ortho_h = src.width, src.height
    ortho_transform = src.transform

with rasterio.open(dsm_file) as src:
    dsm_nodata = src.nodata
    dsm_transform = src.transform

with rasterio.open(dtm_file) as src:
    dtm_nodata = src.nodata
    dtm_transform = src.transform

# Calculate grid
n_cols = (ortho_w - tile_size) // stride + 1
n_rows = (ortho_h - tile_size) // stride + 1
n_total = n_rows * n_cols

print(f"\nOrthophoto: {ortho_h} × {ortho_w}")
print(f"Grid: {n_rows} rows × {n_cols} cols = {n_total:,} total tiles")
print(f"DSM nodata: {dsm_nodata}")
print(f"DTM nodata: {dtm_nodata}")

# Get list of available masks
if os.path.exists(masks_dir):
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.png')]
    print(f"\nAvailable mask files: {len(mask_files):,}")
else:
    mask_files = []
    print(f"\n⚠️  Masks directory not found: {masks_dir}")

# Sample tiles to test
test_tiles = []

# Add center region (most likely to have data)
center_row = n_rows // 2
center_col = n_cols // 2
for dr in range(-5, 6):
    for dc in range(-5, 6):
        tile_num = (center_row + dr) * n_cols + (center_col + dc)
        if 0 <= tile_num < n_total:
            test_tiles.append(tile_num)

# Add some random samples
import random
random.seed(42)
random_samples = random.sample(range(n_total), min(100, n_total))
test_tiles.extend(random_samples)

# Remove duplicates and sort
test_tiles = sorted(list(set(test_tiles)))

print(f"\nTesting {len(test_tiles)} tiles...")
print("="*70)

working_tiles = []
tiles_with_masks = []

for i, tile_num in enumerate(test_tiles):
    # Calculate position
    row_idx = tile_num // n_cols
    col_idx = tile_num % n_cols
    row_start = row_idx * stride
    col_start = col_idx * stride
    
    # Get world bounds
    left, top = rasterio.transform.xy(ortho_transform, row_start, col_start, offset='ul')
    right, bottom = rasterio.transform.xy(ortho_transform, 
                                          row_start + tile_size, 
                                          col_start + tile_size, 
                                          offset='ul')
    bounds = (left, bottom, right, top)
    
    # Test DSM
    try:
        with rasterio.open(dsm_file) as src:
            window = window_from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], 
                                       transform=src.transform)
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
            window = window_from_bounds(bounds[0], bounds[1], bounds[2], bounds[3],
                                       transform=src.transform)
            dtm_tile = src.read(1, window=window)
            
            if dtm_nodata is not None:
                valid_dtm = np.sum(dtm_tile != dtm_nodata)
            else:
                valid_dtm = dtm_tile.size
                
            valid_dtm_pct = valid_dtm / dtm_tile.size * 100
    except:
        valid_dtm = 0
        valid_dtm_pct = 0
    
    # Check if mask exists
    mask_exists = any(f"patch_{tile_num:04d}" in f for f in mask_files)
    
    # Report if valid
    if valid_dsm > 0 and valid_dtm > 0:
        status = "✓"
        working_tiles.append(tile_num)
        if mask_exists:
            tiles_with_masks.append(tile_num)
            status = "✓✓"  # Has data AND mask!
        
        if len(working_tiles) <= 20:  # Print first 20
            location = ""
            if tile_num == center_row * n_cols + center_col:
                location = "(CENTER)"
            
            mask_status = "HAS MASK" if mask_exists else "no mask"
            
            print(f"{status} Tile {tile_num:5d} {location:10s}: "
                  f"DSM {valid_dsm_pct:5.1f}%, DTM {valid_dtm_pct:5.1f}% | {mask_status}")
    
    # Progress
    if (i + 1) % 50 == 0:
        print(f"   ... scanned {i+1}/{len(test_tiles)} tiles "
              f"(found {len(working_tiles)} with data)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if len(working_tiles) > 0:
    print(f"\n✓ Found {len(working_tiles):,} tiles with valid DSM/DTM data")
    print(f"✓ Of these, {len(tiles_with_masks)} have mask files")
    
    if len(tiles_with_masks) > 0:
        print(f"\n🎯 RECOMMENDED TILES (have data + mask):")
        for tile_num in tiles_with_masks[:10]:
            row = tile_num // n_cols
            col = tile_num % n_cols
            print(f"   Tile {tile_num:5d} (row {row:2d}, col {col:2d})")
        
        print(f"\n📝 TO USE A DIFFERENT TILE:")
        print(f"   1. Pick a tile number from above")
        print(f"   2. Update simple_tile_extraction_FIXED.py:")
        print(f"      tile_number = {tiles_with_masks[0]}  # Change this!")
        print(f"   3. Make sure the image and mask files exist")
        
    else:
        print(f"\n⚠️  Tiles have DSM/DTM data but NO mask files found!")
        print(f"   First 10 tiles with data: {working_tiles[:10]}")
        print(f"   But none match your mask files in: {masks_dir}")
        
else:
    print(f"\n✗ NO tiles with valid DSM/DTM data found!")
    print(f"\n   This is very unusual. Possible issues:")
    print(f"   1. DSM/DTM files are completely empty (all nodata)")
    print(f"   2. WebODM processing failed")
    print(f"   3. Files are misaligned/corrupted")
    
    print(f"\n   Debug steps:")
    print(f"   1. Open DSM/DTM in QGIS - do you see elevation data?")
    print(f"   2. Check WebODM processing logs")
    print(f"   3. Verify files aren't corrupted")

# Also check tile 359 specifically
print(f"\n" + "="*70)
print(f"YOUR TILE (359) ANALYSIS:")
print("="*70)

tile_num = 359
row_idx = tile_num // n_cols
col_idx = tile_num % n_cols
row_start = row_idx * stride
col_start = col_idx * stride

left, top = rasterio.transform.xy(ortho_transform, row_start, col_start, offset='ul')
right, bottom = rasterio.transform.xy(ortho_transform, 
                                      row_start + tile_size, 
                                      col_start + tile_size, 
                                      offset='ul')
bounds = (left, bottom, right, top)

with rasterio.open(dsm_file) as src:
    window = window_from_bounds(bounds[0], bounds[1], bounds[2], bounds[3],
                               transform=src.transform)
    dsm_tile = src.read(1, window=window)
    
    if dsm_nodata is not None:
        valid_dsm = np.sum(dsm_tile != dsm_nodata)
        valid_dsm_pct = valid_dsm / dsm_tile.size * 100
    else:
        valid_dsm = dsm_tile.size
        valid_dsm_pct = 100.0

with rasterio.open(dtm_file) as src:
    window = window_from_bounds(bounds[0], bounds[1], bounds[2], bounds[3],
                               transform=src.transform)
    dtm_tile = src.read(1, window=window)
    
    if dtm_nodata is not None:
        valid_dtm = np.sum(dtm_tile != dtm_nodata)
        valid_dtm_pct = valid_dtm / dtm_tile.size * 100
    else:
        valid_dtm = dtm_tile.size
        valid_dtm_pct = 100.0

print(f"Position: row {row_idx}, col {col_idx} (pixel {row_start}, {col_start})")
print(f"DSM valid: {valid_dsm_pct:.1f}%")
print(f"DTM valid: {valid_dtm_pct:.1f}%")

if valid_dsm_pct == 0 or valid_dtm_pct == 0:
    print(f"\n✗ Tile 359 has NO valid elevation data")
    print(f"  This tile is outside the area WebODM processed")
    print(f"  Try a different tile from the recommended list above")
else:
    print(f"\n✓ Tile 359 has valid elevation data!")

print("\n" + "="*70)
