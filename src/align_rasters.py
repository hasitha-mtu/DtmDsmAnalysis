"""
Fix DTM/DSM shape mismatch by cropping to common extent
Handles small alignment issues from WebODM processing
"""

import rasterio

def align_rasters(dsm_file, dtm_file, output_dsm='dsm_aligned.tif', output_dtm='dtm_aligned.tif'):
    """
    Align DSM and DTM by cropping to smallest common extent
    
    Parameters:
    -----------
    dsm_file : str
        Path to DSM file
    dtm_file : str
        Path to DTM file
    output_dsm : str
        Output path for aligned DSM
    output_dtm : str
        Output path for aligned DTM
    """
    
    print("="*70)
    print("DTM/DSM ALIGNMENT TOOL")
    print("="*70)
    
    # Load DSM
    with rasterio.open(dsm_file) as src:
        dsm = src.read(1, masked=True)
        dsm_profile = src.profile
        dsm_shape = dsm.shape
        print(f"\nOriginal DSM shape: {dsm_shape}")
    
    # Load DTM
    with rasterio.open(dtm_file) as src:
        dtm = src.read(1, masked=True)
        dtm_profile = src.profile
        dtm_shape = dtm.shape
        print(f"Original DTM shape: {dtm_shape}")
    
    # Check if alignment needed
    if dsm_shape == dtm_shape:
        print("\n✓ Files already aligned! No action needed.")
        return dsm_file, dtm_file
    
    print(f"\n⚠ Shape mismatch detected:")
    print(f"  Height diff: {abs(dsm_shape[0] - dtm_shape[0])} pixels")
    print(f"  Width diff: {abs(dsm_shape[1] - dtm_shape[1])} pixels")
    
    # Calculate common extent (crop to smaller)
    common_height = min(dsm_shape[0], dtm_shape[0])
    common_width = min(dsm_shape[1], dtm_shape[1])
    
    print(f"\nAligning to common extent: ({common_height}, {common_width})")
    
    # Crop DSM
    dsm_cropped = dsm[:common_height, :common_width]
    pixels_removed_dsm = dsm.size - dsm_cropped.size
    
    # Crop DTM
    dtm_cropped = dtm[:common_height, :common_width]
    pixels_removed_dtm = dtm.size - dtm_cropped.size
    
    print(f"\nDSM: Removed {pixels_removed_dsm:,} pixels ({pixels_removed_dsm/dsm.size*100:.4f}%)")
    print(f"DTM: Removed {pixels_removed_dtm:,} pixels ({pixels_removed_dtm/dtm.size*100:.4f}%)")
    
    # Update profiles
    dsm_profile.update({
        'height': common_height,
        'width': common_width
    })
    
    dtm_profile.update({
        'height': common_height,
        'width': common_width
    })
    
    # Save aligned DSM
    print(f"\nSaving aligned DSM to: {output_dsm}")
    with rasterio.open(output_dsm, 'w', **dsm_profile) as dst:
        dst.write(dsm_cropped, 1)
    
    # Save aligned DTM
    print(f"Saving aligned DTM to: {output_dtm}")
    with rasterio.open(output_dtm, 'w', **dtm_profile) as dst:
        dst.write(dtm_cropped, 1)
    
    print("\n" + "="*70)
    print("ALIGNMENT COMPLETE!")
    print("="*70)
    print(f"\nAligned files:")
    print(f"  DSM: {output_dsm}")
    print(f"  DTM: {output_dtm}")
    print(f"  Shape: ({common_height}, {common_width})")
    print(f"\n✓ Files are now ready for validation!")
    print("  Run: python validate_dtm_dsm.py with the aligned files")
    print("="*70)
    
    return output_dsm, output_dtm


if __name__ == "__main__":
    # UPDATE THESE PATHS
    dsm_input = "C:/Users/AdikariAdikari/PycharmProjects/DtmDsmAnalysis/dataset/odm_dem/dsm.tif"
    dtm_input = "C:/Users/AdikariAdikari/PycharmProjects/DtmDsmAnalysis/dataset/odm_dem/dtm.tif"
    
    # Output files (will be created in same directory)
    dsm_output = "C:/Users/AdikariAdikari/PycharmProjects/DtmDsmAnalysis/dataset/odm_dem/dsm_aligned.tif"
    dtm_output = "C:/Users/AdikariAdikari/PycharmProjects/DtmDsmAnalysis/dataset/odm_dem/dtm_aligned.tif"
    
    # Run alignment
    aligned_dsm, aligned_dtm = align_rasters(
        dsm_input, 
        dtm_input, 
        dsm_output, 
        dtm_output
    )
    
    print("\n💡 TIP: The 1-pixel difference you had is completely normal!")
    print("   This often happens during WebODM processing due to:")
    print("   • Rounding in coordinate transformations")
    print("   • Edge effects in DEM generation")
    print("   • Slightly different processing parameters")
    print("\n   Cropping to common extent is the standard solution.")
