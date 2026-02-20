"""
Scan YOUR actual image/mask files and find which have valid DSM/DTM data

This is MUCH better - only tests tiles you actually have!
"""

import numpy as np
import rasterio
from rasterio.windows import from_bounds as window_from_bounds
import os
import re

# =======================================================================
# YOUR FOLDERS - UPDATE THESE!
# =======================================================================

images_dir = r"C:\Users\AdikariAdikari\PycharmProjects\river_segmentation\dataset\processed_1024\train\images"
masks_dir = r"C:\Users\AdikariAdikari\PycharmProjects\river_segmentation\dataset\processed_1024\train\masks"
orthophoto_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_orthophoto\odm_orthophoto.tif"
dsm_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_dem\dsm_aligned.tif"
dtm_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_dem\dtm_aligned.tif"

tile_size = 1024
stride = 768

print("="*70)
print("FINDING VALID TILES FROM YOUR ACTUAL FILES")
print("="*70)

# =======================================================================
# 1. SCAN YOUR FILES
# =======================================================================

print(f"\n1. Scanning your files...")

# Get image files
if not os.path.exists(images_dir):
    print(f"✗ Images directory not found: {images_dir}")
    exit(1)

image_files = [f for f in os.listdir(images_dir) 
               if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]

print(f"   Found {len(image_files):,} image files")

# Get mask files
if not os.path.exists(masks_dir):
    print(f"✗ Masks directory not found: {masks_dir}")
    exit(1)

mask_files = [f for f in os.listdir(masks_dir) 
              if f.endswith(('.png', '.PNG'))]

print(f"   Found {len(mask_files):,} mask files")

# =======================================================================
# 2. EXTRACT TILE NUMBERS
# =======================================================================

print(f"\n2. Extracting tile numbers from filenames...")

def extract_tile_number(filename):
    """
    Extract tile number from filename
    
    Examples:
    - "DJI_20250728101825_0553_V_patch_0359.jpg" → 359
    - "patch_0359.png" → 359
    - "tile_0359.jpg" → 359
    """
    # Try different patterns
    patterns = [
        r'patch[_-](\d+)',      # patch_0359, patch-0359
        r'tile[_-](\d+)',       # tile_0359, tile-0359
        r'_(\d{4})[._]',        # _0359. or _0359_
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    return None

# Extract tile numbers from images
image_tiles = {}
for filename in image_files:
    tile_num = extract_tile_number(filename)
    if tile_num is not None:
        image_tiles[tile_num] = filename

# Extract tile numbers from masks
mask_tiles = {}
for filename in mask_files:
    tile_num = extract_tile_number(filename)
    if tile_num is not None:
        mask_tiles[tile_num] = filename

print(f"   Extracted {len(image_tiles)} tile numbers from images")
print(f"   Extracted {len(mask_tiles)} tile numbers from masks")

# Find tiles that have both image and mask
tiles_with_both = sorted(set(image_tiles.keys()) & set(mask_tiles.keys()))

print(f"   {len(tiles_with_both)} tiles have BOTH image and mask")

if len(tiles_with_both) == 0:
    print(f"\n✗ No matching image/mask pairs found!")
    print(f"   Check that filenames match between images and masks")
    print(f"\n   Example image: {image_files[0] if image_files else 'N/A'}")
    print(f"   Example mask: {mask_files[0] if mask_files else 'N/A'}")
    exit(1)

print(f"\n   Tile range: {min(tiles_with_both)} to {max(tiles_with_both)}")
print(f"   First 10 tiles: {tiles_with_both[:10]}")

# =======================================================================
# 3. GET GEOREFERENCING INFO
# =======================================================================

print(f"\n3. Loading DSM/DTM info...")

with rasterio.open(orthophoto_file) as src:
    ortho_w, ortho_h = src.width, src.height
    ortho_transform = src.transform
    ortho_crs = src.crs

with rasterio.open(dsm_file) as src:
    dsm_nodata = src.nodata
    dsm_transform = src.transform

with rasterio.open(dtm_file) as src:
    dtm_nodata = src.nodata
    dtm_transform = src.transform

n_cols = (ortho_w - tile_size) // stride + 1
n_rows = (ortho_h - tile_size) // stride + 1

print(f"   Orthophoto: {ortho_h} × {ortho_w} pixels")
print(f"   Grid: {n_rows} rows × {n_cols} cols")
print(f"   DSM nodata: {dsm_nodata}")
print(f"   DTM nodata: {dtm_nodata}")

# =======================================================================
# 4. TEST YOUR TILES FOR DSM/DTM COVERAGE
# =======================================================================

print(f"\n4. Testing tiles for DSM/DTM coverage...")
print("="*70)

working_tiles = []

for i, tile_num in enumerate(tiles_with_both):
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
    except Exception as e:
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
    except Exception as e:
        valid_dtm = 0
        valid_dtm_pct = 0
    
    # Check if valid
    if valid_dsm > 0 and valid_dtm > 0:
        status = "✓"
        working_tiles.append({
            'tile_num': tile_num,
            'row': row_idx,
            'col': col_idx,
            'dsm_pct': valid_dsm_pct,
            'dtm_pct': valid_dtm_pct,
            'image_file': image_tiles[tile_num],
            'mask_file': mask_tiles[tile_num]
        })
        
        print(f"{status} Tile {tile_num:5d} (row {row_idx:2d}, col {col_idx:2d}): "
              f"DSM {valid_dsm_pct:5.1f}%, DTM {valid_dtm_pct:5.1f}%")
    else:
        print(f"✗ Tile {tile_num:5d} (row {row_idx:2d}, col {col_idx:2d}): "
              f"DSM {valid_dsm_pct:5.1f}%, DTM {valid_dtm_pct:5.1f}% - NO DATA")
    
    # Progress every 20 tiles
    if (i + 1) % 20 == 0:
        print(f"   ... tested {i+1}/{len(tiles_with_both)} tiles "
              f"(found {len(working_tiles)} working)")

# =======================================================================
# 5. SUMMARY AND RECOMMENDATIONS
# =======================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if len(working_tiles) > 0:
    print(f"\n✓ SUCCESS! Found {len(working_tiles)} tiles with valid DSM/DTM data")
    print(f"  Out of {len(tiles_with_both)} total tiles checked")
    
    # Sort by coverage (best first)
    working_tiles.sort(key=lambda x: (x['dsm_pct'] + x['dtm_pct']) / 2, reverse=True)
    
    print(f"\n🎯 TOP 10 RECOMMENDED TILES (best coverage):")
    print("="*70)
    
    for i, tile in enumerate(working_tiles[:10]):
        avg_coverage = (tile['dsm_pct'] + tile['dtm_pct']) / 2
        print(f"\n{i+1}. Tile {tile['tile_num']} (row {tile['row']}, col {tile['col']})")
        print(f"   Coverage: DSM {tile['dsm_pct']:.1f}%, DTM {tile['dtm_pct']:.1f}% (avg {avg_coverage:.1f}%)")
        print(f"   Image: {tile['image_file']}")
        print(f"   Mask:  {tile['mask_file']}")
    
    # Provide code to use best tile
    best_tile = working_tiles[0]
    
    print(f"\n" + "="*70)
    print(f"📝 TO USE THE BEST TILE:")
    print("="*70)
    print(f"\nUpdate simple_tile_extraction_FIXED.py with:\n")
    print(f"tile_number = {best_tile['tile_num']}")
    print(f"tile_image = r\"{images_dir}\\{best_tile['image_file']}\"")
    print(f"mask_image = r\"{masks_dir}\\{best_tile['mask_file']}\"")
    
    print(f"\nThen run:")
    print(f"python simple_tile_extraction_FIXED.py")
    
    # Save results to file
    output_file = "working_tiles.txt"
    with open(output_file, 'w') as f:
        f.write("TILES WITH VALID DSM/DTM DATA\n")
        f.write("="*70 + "\n\n")
        for tile in working_tiles:
            f.write(f"Tile {tile['tile_num']}: ")
            f.write(f"DSM {tile['dsm_pct']:.1f}%, DTM {tile['dtm_pct']:.1f}%\n")
            f.write(f"  Image: {tile['image_file']}\n")
            f.write(f"  Mask:  {tile['mask_file']}\n\n")
    
    print(f"\n✓ Full list saved to: {output_file}")
    
else:
    print(f"\n✗ NO tiles with valid DSM/DTM data found!")
    print(f"\n   Tested {len(tiles_with_both)} tiles, but NONE have elevation data")
    print(f"\n   Possible issues:")
    print(f"   1. DSM/DTM files are completely empty")
    print(f"   2. Your tiles are all outside the DSM/DTM coverage area")
    print(f"   3. WebODM processing failed for elevation")
    
    print(f"\n   Debug steps:")
    print(f"   1. Open DSM in QGIS - do you see ANY elevation data?")
    print(f"   2. Check WebODM processing logs")
    print(f"   3. Try reprocessing with more images/overlap")
    
    # Show tile distribution
    print(f"\n   Your tile distribution:")
    print(f"   Min tile: {min(tiles_with_both)}")
    print(f"   Max tile: {max(tiles_with_both)}")
    print(f"   Grid center would be: tile ~{(n_rows//2) * n_cols + n_cols//2}")

print("\n" + "="*70)
