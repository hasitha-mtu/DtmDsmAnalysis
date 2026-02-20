"""
Convert Bluesky DTM from ASC+PRJ → GeoTIFF and reproject to EPSG:32629

Handles:
  - .asc + .prj (Bluesky standard delivery format)
  - .tif already in EPSG:2157 (Irish Transverse Mercator)
  - Multiple ASC tiles in a folder (merges them into one)

Output: Single GeoTIFF in EPSG:32629 ready for depth extraction
"""

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
from rasterio.crs import CRS
import numpy as np
import os
import glob

# =======================================================================
# YOUR FILES - UPDATE THESE
# =======================================================================

# Option A: Single ASC file
bluesky_input = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\5m_dtm\205999-2_DTM_1_Shape.asc"

# Option B: Folder containing MULTIPLE ASC tiles (will merge them)
bluesky_input_folder = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\5m_dtm"

# Output GeoTIFF (reprojected to EPSG:32629)
bluesky_output = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\5m_dtm\bluesky_dtm_utm29n.tif"

# Geoid offset (Malin Head → WGS84 ellipsoid for Ireland)
GEOID_OFFSET = 58.0  # metres

# =======================================================================
# STEP 1: INSPECT YOUR FILES
# =======================================================================

def inspect_asc_file(asc_path):
    """Print info about an ASC file and its PRJ"""
    
    prj_path = os.path.splitext(asc_path)[0] + '.prj'
    
    print(f"\n{'='*70}")
    print(f"INSPECTING: {os.path.basename(asc_path)}")
    print(f"{'='*70}")
    
    # Check PRJ
    if os.path.exists(prj_path):
        print(f"\n✓ PRJ file found: {os.path.basename(prj_path)}")
        with open(prj_path) as f:
            prj_content = f.read().strip()
        print(f"  CRS: {prj_content[:120]}")
    else:
        print(f"\n⚠️  No PRJ file - will assume EPSG:2157 (Irish Grid)")
    
    # Open with rasterio
    with rasterio.open(asc_path) as src:
        data = src.read(1)
        nodata = src.nodata
        valid = data[data != nodata] if nodata is not None else data.flatten()
        
        print(f"\nFile properties:")
        print(f"  Shape:      {src.height} × {src.width} pixels")
        print(f"  Resolution: {abs(src.transform.a):.1f}m")
        print(f"  CRS:        {src.crs}")
        print(f"  Bounds:     {src.bounds}")
        print(f"  Nodata:     {nodata}")
        print(f"  Valid pixels: {len(valid):,} / {data.size:,} ({len(valid)/data.size*100:.1f}%)")
        
        if len(valid) > 0:
            print(f"\nElevation (Malin Head orthometric):")
            print(f"  Min:  {np.min(valid):.2f}m")
            print(f"  Max:  {np.max(valid):.2f}m")
            print(f"  Mean: {np.mean(valid):.2f}m")
            print(f"\nAfter +{GEOID_OFFSET}m geoid correction (WGS84 ellipsoid):")
            print(f"  Min:  {np.min(valid) + GEOID_OFFSET:.2f}m")
            print(f"  Max:  {np.max(valid) + GEOID_OFFSET:.2f}m")
            print(f"  Mean: {np.mean(valid) + GEOID_OFFSET:.2f}m")

# =======================================================================
# STEP 2: FIND ASC FILES IN FOLDER
# =======================================================================

def find_asc_files(folder):
    """Find all ASC files in a folder"""
    
    asc_files = sorted(
        glob.glob(os.path.join(folder, '*.asc')) +
        glob.glob(os.path.join(folder, '*.ASC'))
    )
    
    print(f"\nFound {len(asc_files)} ASC file(s) in: {folder}")
    for f in asc_files:
        size_mb = os.path.getsize(f) / 1024 / 1024
        prj = os.path.splitext(f)[0] + '.prj'
        prj_status = "✓ PRJ" if os.path.exists(prj) else "⚠️  no PRJ"
        print(f"  {os.path.basename(f):50s} {size_mb:6.1f}MB  {prj_status}")
    
    return asc_files

# =======================================================================
# STEP 3: CONVERT SINGLE ASC → INTERMEDIATE GeoTIFF
# =======================================================================

def asc_to_geotiff(asc_path, output_tif, assumed_crs='EPSG:2157'):
    """
    Convert ASC+PRJ to GeoTIFF (same CRS, just format change)
    """
    
    print(f"\n  Converting: {os.path.basename(asc_path)} → {os.path.basename(output_tif)}")
    
    with rasterio.open(asc_path) as src:
        
        crs = src.crs
        if crs is None:
            print(f"  ⚠️  No CRS in file - assuming {assumed_crs}")
            crs = CRS.from_string(assumed_crs)
        
        data = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999.0
        
        meta = src.meta.copy()
        meta.update({
            'driver': 'GTiff',
            'crs': crs,
            'dtype': 'float32',
            'nodata': nodata,
            'compress': 'deflate',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
        })
        
        with rasterio.open(output_tif, 'w', **meta) as dst:
            dst.write(data, 1)
    
    size_mb = os.path.getsize(output_tif) / 1024 / 1024
    print(f"  ✓ Written: {size_mb:.1f}MB")
    return output_tif

# =======================================================================
# STEP 4: MERGE MULTIPLE ASC TILES
# =======================================================================

def merge_asc_tiles(asc_files, merged_output, assumed_crs='EPSG:2157'):
    """Merge multiple ASC tiles into one GeoTIFF"""
    
    print(f"\n  Merging {len(asc_files)} tiles...")
    
    src_files = []
    for asc_path in asc_files:
        src = rasterio.open(asc_path)
        # Patch CRS if missing
        if src.crs is None:
            print(f"  ⚠️  {os.path.basename(asc_path)} has no CRS - assuming {assumed_crs}")
        src_files.append(src)
    
    mosaic, out_transform = merge(src_files, method='first')
    
    meta = src_files[0].meta.copy()
    crs = src_files[0].crs or CRS.from_string(assumed_crs)
    nodata = src_files[0].nodata if src_files[0].nodata is not None else -9999.0
    
    for src in src_files:
        src.close()
    
    meta.update({
        'driver': 'GTiff',
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'transform': out_transform,
        'crs': crs,
        'dtype': 'float32',
        'nodata': nodata,
        'compress': 'deflate',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
    })
    
    with rasterio.open(merged_output, 'w', **meta) as dst:
        dst.write(mosaic.astype(np.float32))
    
    size_mb = os.path.getsize(merged_output) / 1024 / 1024
    print(f"  ✓ Merged: {mosaic.shape[2]} × {mosaic.shape[1]} pixels, {size_mb:.1f}MB")
    return merged_output

# =======================================================================
# STEP 5: REPROJECT TO EPSG:32629
# =======================================================================

def reproject_to_utm29n(input_tif, output_tif, apply_geoid_offset=False):
    """
    Reproject from EPSG:2157 (Irish Grid) to EPSG:32629 (WGS84 UTM 29N)
    """
    
    target_crs = 'EPSG:32629'
    print(f"\n  Reprojecting to {target_crs}...")
    
    with rasterio.open(input_tif) as src:
        
        print(f"  Input:  {src.crs}  {src.height} × {src.width}  {abs(src.transform.a):.2f}m")
        
        transform, width, height = calculate_default_transform(
            src.crs, target_crs,
            src.width, src.height,
            *src.bounds
        )
        
        nodata = src.nodata if src.nodata is not None else -9999.0
        
        meta = src.meta.copy()
        meta.update({
            'driver': 'GTiff',
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'dtype': 'float32',
            'nodata': nodata,
            'compress': 'deflate',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
        })
        
        data = src.read(1).astype(np.float32)
        
        # Optionally bake geoid correction into values
        if apply_geoid_offset:
            valid = data != nodata
            data[valid] += GEOID_OFFSET
            print(f"  Applied +{GEOID_OFFSET}m geoid offset into file values")
        
        with rasterio.open(output_tif, 'w', **meta) as dst:
            reproject(
                source=data,
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
                src_nodata=nodata,
                dst_nodata=nodata
            )
    
    # Verify
    with rasterio.open(output_tif) as src:
        data = src.read(1)
        valid = data[data != src.nodata]
        print(f"  Output: {src.crs}  {src.height} × {src.width}  {abs(src.transform.a):.2f}m")
        if len(valid) > 0:
            label = "(WGS84 ellipsoid)" if apply_geoid_offset else "(Malin Head - apply +58m at runtime)"
            print(f"  Values: {np.min(valid):.2f} to {np.max(valid):.2f}m  {label}")
    
    return output_tif

# =======================================================================
# COMPLETE PIPELINE
# =======================================================================

def convert_bluesky_asc_to_utm_tif(input_path, output_tif, apply_geoid_offset=False):
    """
    Full pipeline: .asc or folder → merged GeoTIFF → EPSG:32629
    """
    
    print("="*70)
    print("BLUESKY DTM CONVERSION: ASC+PRJ → UTM29N GeoTIFF")
    print("="*70)
    
    intermediate_tif = output_tif.replace('.tif', '_itm_temp.tif')
    
    # ------------------------------------------------------------------
    # Detect input type and convert to intermediate GeoTIFF
    # ------------------------------------------------------------------
    
    if os.path.isdir(input_path):
        asc_files = find_asc_files(input_path)
        
        if len(asc_files) == 0:
            print(f"✗ No ASC files found in {input_path}")
            return None
        
        inspect_asc_file(asc_files[0])
        
        if len(asc_files) == 1:
            asc_to_geotiff(asc_files[0], intermediate_tif)
        else:
            merge_asc_tiles(asc_files, intermediate_tif)
    
    elif input_path.lower().endswith(('.asc',)):
        inspect_asc_file(input_path)
        asc_to_geotiff(input_path, intermediate_tif)
    
    elif input_path.lower().endswith('.tif'):
        print(f"\n✓ Already a GeoTIFF - skipping conversion")
        intermediate_tif = input_path
    
    else:
        print(f"✗ Unknown format: {input_path}")
        return None
    
    # ------------------------------------------------------------------
    # Reproject to EPSG:32629
    # ------------------------------------------------------------------
    
    reproject_to_utm29n(intermediate_tif, output_tif, apply_geoid_offset)
    
    # Clean up temp file
    if intermediate_tif != input_path and os.path.exists(intermediate_tif):
        os.remove(intermediate_tif)
    
    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    
    print(f"\n{'='*70}")
    print(f"✅ DONE")
    print(f"{'='*70}")
    print(f"\n  Output file: {output_tif}")
    print(f"  CRS: EPSG:32629 (matches WebODM)")
    
    if not apply_geoid_offset:
        print(f"\n  ⚠️  Geoid correction NOT baked in.")
        print(f"     bluesky_dtm_depth.py will add +{GEOID_OFFSET}m automatically.")
    else:
        print(f"\n  ✓ Geoid correction (+{GEOID_OFFSET}m) baked into values.")
        print(f"     Set GEOID_OFFSET = 0.0 in bluesky_dtm_depth.py.")
    
    print(f"\n  Update bluesky_dtm_depth.py:")
    print(f"  bluesky_dtm_file = r\"{output_tif}\"")
    
    return output_tif

# =======================================================================
# MAIN
# =======================================================================

if __name__ == "__main__":
    
    import sys
    
    # Auto-detect: is input a file or folder?
    if os.path.isfile(bluesky_input):
        input_path = bluesky_input
        print(f"\n  Mode: Single ASC file → {os.path.basename(bluesky_input)}")
    
    elif os.path.isdir(bluesky_input_folder):
        input_path = bluesky_input_folder
        print(f"\n  Mode: Folder of ASC tiles → will merge")
    
    else:
        print(f"\n✗ Input not found. Update paths at top of script:")
        print(f"     bluesky_input = '{bluesky_input}'")
        print(f"     bluesky_input_folder = '{bluesky_input_folder}'")
        sys.exit(1)
    
    convert_bluesky_asc_to_utm_tif(
        input_path=input_path,
        output_tif=bluesky_output,
        apply_geoid_offset=False   # Correction applied at runtime in bluesky_dtm_depth.py
    )
