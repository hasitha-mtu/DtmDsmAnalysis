"""
Convert Bluesky ASC format to GeoTIFF
Makes files faster to read and easier to work with
"""

import rasterio
import numpy as np
import os

def convert_asc_to_geotiff(asc_file, output_tif=None):
    """
    Convert ESRI ASCII Grid (.asc) to GeoTIFF (.tif)
    
    Parameters:
    -----------
    asc_file : str
        Path to .asc file (e.g., 'bluesky_dsm.asc')
    output_tif : str, optional
        Output path (default: same name with .tif extension)
    """
    
    if output_tif is None:
        output_tif = asc_file.replace('.asc', '.tif')
    
    print(f"Converting: {asc_file}")
    print(f"Output: {output_tif}")
    
    # Read ASC file (rasterio automatically finds .prj)
    with rasterio.open(asc_file) as src:
        data = src.read(1)
        profile = src.profile
        
        print(f"\nInput file info:")
        print(f"  Shape: {data.shape}")
        print(f"  CRS: {src.crs}")
        print(f"  Resolution: {src.res}")
        print(f"  Bounds: {src.bounds}")
        print(f"  NoData value: {src.nodata}")
        
        # Update profile for GeoTIFF output
        profile.update(
            driver='GTiff',
            compress='lzw',  # Compress to save space
            tiled=True,      # Tiled for faster reading
            blockxsize=256,
            blockysize=256
        )
    
    # Write GeoTIFF
    print(f"\nWriting GeoTIFF...")
    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(data, 1)
    
    # Check file sizes
    asc_size = os.path.getsize(asc_file) / (1024**2)  # MB
    tif_size = os.path.getsize(output_tif) / (1024**2)  # MB
    
    print(f"\n✓ Conversion complete!")
    print(f"  ASC size: {asc_size:.1f} MB")
    print(f"  TIF size: {tif_size:.1f} MB")
    print(f"  Space saved: {asc_size - tif_size:.1f} MB ({(1-tif_size/asc_size)*100:.1f}%)")
    
    return output_tif

def convert_bluesky_dataset(dsm_asc, dtm_asc):
    """
    Convert both Bluesky DSM and DTM files
    
    Parameters:
    -----------
    dsm_asc : str
        Path to Bluesky DSM .asc file
    dtm_asc : str
        Path to Bluesky DTM .asc file
    """
    
    print("="*70)
    print("CONVERTING BLUESKY DATA TO GEOTIFF")
    print("="*70)
    
    # Convert DSM
    print("\n[1/2] Converting DSM...")
    dsm_tif = convert_asc_to_geotiff(dsm_asc)
    
    # Convert DTM
    print("\n" + "="*70)
    print("\n[2/2] Converting DTM...")
    dtm_tif = convert_asc_to_geotiff(dtm_asc)
    
    print("\n" + "="*70)
    print("CONVERSION COMPLETE!")
    print("="*70)
    print(f"\nConverted files:")
    print(f"  DSM: {dsm_tif}")
    print(f"  DTM: {dtm_tif}")
    print(f"\nYou can now use these .tif files in your validation scripts.")
    print("They will load much faster than the original .asc files!")
    
    return dsm_tif, dtm_tif

def quick_check_asc(asc_file):
    """
    Quick check of ASC file contents
    """
    
    print("="*70)
    print(f"CHECKING: {asc_file}")
    print("="*70)
    
    with rasterio.open(asc_file) as src:
        data = src.read(1)
        
        print(f"\nFile Information:")
        print(f"  Format: {src.driver}")
        print(f"  Shape: {data.shape[0]} rows × {data.shape[1]} cols")
        print(f"  CRS: {src.crs}")
        print(f"  Resolution: {src.res[0]:.2f} × {src.res[1]:.2f} m")
        print(f"  Bounds: {src.bounds}")
        
        print(f"\nElevation Statistics:")
        valid_data = data[data != src.nodata]
        print(f"  Min: {np.min(valid_data):.2f} m")
        print(f"  Max: {np.max(valid_data):.2f} m")
        print(f"  Mean: {np.mean(valid_data):.2f} m")
        print(f"  Median: {np.median(valid_data):.2f} m")
        
        print(f"\nData Quality:")
        nodata_count = np.sum(data == src.nodata)
        nodata_pct = (nodata_count / data.size) * 100
        print(f"  Valid pixels: {data.size - nodata_count:,}")
        print(f"  NoData pixels: {nodata_count:,} ({nodata_pct:.1f}%)")
        
        # Check if .prj file exists
        prj_file = asc_file.replace('.asc', '.prj')
        if os.path.exists(prj_file):
            print(f"\n✓ Found .prj file: {prj_file}")
            print(f"  CRS properly defined")
        else:
            print(f"\n⚠ WARNING: No .prj file found!")
            print(f"  Expected: {prj_file}")
            print(f"  CRS may not be properly defined")
    
    print("="*70)

if __name__ == "__main__":
    # UPDATE THIS PATH
    
    # Your Bluesky DSM file only
    bluesky_dsm_asc = "C:\\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\\bluesky\\1m_dsm\\205999-1_DSM_1_Shape.asc"
    
    # Check what you have
    print("\n>>> CHECKING DSM FILE <<<")
    quick_check_asc(bluesky_dsm_asc)
    
    # Convert to GeoTIFF (recommended!)
    print("\n\n>>> CONVERTING DSM TO GEOTIFF <<<")
    user_input = input("\nConvert DSM to GeoTIFF? (y/n): ")
    
    if user_input.lower() == 'y':
        print("\nConverting DSM only (DTM comparison not recommended due to resolution/technology issues)...")
        dsm_tif = convert_asc_to_geotiff(bluesky_dsm_asc)
        
        print("\n" + "="*70)
        print("CONVERSION COMPLETE!")
        print("="*70)
        print(f"\nConverted file:")
        print(f"  DSM: {dsm_tif}")
        print(f"\n💡 NEXT STEPS:")
        print("   1. Use this .tif file for DSM validation:")
        print(f"      bluesky_dsm = '{dsm_tif}'")
        print("   2. Focus validation on DSM only (16x resolution gap)")
        print("   3. Skip DTM comparison (82x gap + technology differences)")
        print("   4. Run: python realistic_bluesky_comparison.py")
    else:
        print("\nSkipped conversion. You can still use .asc file directly:")
        print(f"   bluesky_dsm = '{bluesky_dsm_asc}'")
        print("\n   Note: .prj file must stay in same folder!")
        print(f"   Expected: {bluesky_dsm_asc.replace('.asc', '.prj')}")
