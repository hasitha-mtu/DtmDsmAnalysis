"""
Reproject Bluesky DSM from EPSG:2157 (Irish Grid) to EPSG:32629 (WGS84 UTM)
This fixes the CRS mismatch so comparison can work
"""

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np

def reproject_bluesky_to_webodm_crs(bluesky_input, bluesky_output):
    """
    Reproject Bluesky DSM from EPSG:2157 (ITM) to EPSG:32629 (WGS84 UTM 29N)
    
    Parameters:
    -----------
    bluesky_input : str
        Input Bluesky DSM in EPSG:2157
    bluesky_output : str
        Output Bluesky DSM in EPSG:32629
    """
    
    print("="*70)
    print("REPROJECTING BLUESKY DSM TO MATCH WEBODM CRS")
    print("="*70)
    
    target_crs = 'EPSG:32629'  # WGS 84 / UTM Zone 29N (WebODM CRS)
    
    print(f"\n1. Loading Bluesky DSM...")
    with rasterio.open(bluesky_input) as src:
        print(f"   Input file: {bluesky_input}")
        print(f"   Input CRS: {src.crs}")
        print(f"   Input shape: {src.shape}")
        print(f"   Input bounds: {src.bounds}")
        
        # Calculate transform and dimensions for target CRS
        print(f"\n2. Calculating reprojection parameters...")
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        
        print(f"   Target CRS: {target_crs}")
        print(f"   Target shape: ({height}, {width})")
        
        # Update metadata
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        print(f"\n3. Reprojecting (this may take a minute)...")
        
        # Create output file
        with rasterio.open(bluesky_output, 'w', **kwargs) as dst:
            # Reproject band 1
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,  # Good for elevation data
                src_nodata=src.nodata,
                dst_nodata=src.nodata
            )
    
    # Verify output
    print(f"\n4. Verifying reprojected file...")
    with rasterio.open(bluesky_output) as src:
        data = src.read(1)
        valid_data = data[data != src.nodata]
        
        print(f"   Output file: {bluesky_output}")
        print(f"   Output CRS: {src.crs}")
        print(f"   Output shape: {src.shape}")
        print(f"   Output bounds: {src.bounds}")
        print(f"\n   Elevation check:")
        print(f"     Min: {np.min(valid_data):.2f} m")
        print(f"     Max: {np.max(valid_data):.2f} m")
        print(f"     Mean: {np.mean(valid_data):.2f} m")
        print(f"     Valid pixels: {len(valid_data):,}")
    
    print(f"\n" + "="*70)
    print("✅ REPROJECTION COMPLETE!")
    print("="*70)
    print(f"\nReprojected file: {bluesky_output}")
    print(f"CRS: {target_crs} (matches WebODM)")
    print(f"\n💡 NEXT STEPS:")
    print(f"   1. Update realistic_bluesky_comparison.py:")
    print(f"      bluesky_dsm = '{bluesky_output}'")
    print(f"   2. Re-run comparison")
    print(f"   3. Should now see reasonable results!")
    print("="*70)
    
    return bluesky_output

if __name__ == "__main__":
    # UPDATE THESE PATHS
    
    # Your original Bluesky file (EPSG:2157)
    bluesky_input = "C:\\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\\bluesky\\1m_dsm\\205999-1_DSM_1_Shape.tif"  # or .asc (ITM)
    
    # Output file (will be EPSG:32629)
    bluesky_output = "C:\\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\\bluesky\\1m_dsm\\bluesky_dsm_utm29n.tif"  # Reprojected
    
    print("\n🔄 Reprojecting Bluesky from Irish Grid (ITM) to WGS84 UTM Zone 29N")
    print("   This is necessary because:")
    print("   • Bluesky uses EPSG:2157 (Irish national grid)")
    print("   • WebODM uses EPSG:32629 (WGS84 UTM)")
    print("   • They need to match for comparison")
    print("\n")
    
    output_file = reproject_bluesky_to_webodm_crs(bluesky_input, bluesky_output)
    
    print("\n✓ Done! Bluesky DSM is now in the same CRS as WebODM")
    print("  Both files now use EPSG:32629 (WGS 84 / UTM Zone 29N)")
