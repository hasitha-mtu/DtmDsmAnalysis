"""
All-in-one: Align DSM/DTM if needed, then validate
Convenient workflow for handling shape mismatches
"""

import rasterio
import numpy as np
import os

def quick_align_and_validate(dsm_file, dtm_file):
    """
    Check if alignment needed, align if so, then validate
    """
    
    print("="*70)
    print("QUICK ALIGN & VALIDATE")
    print("="*70)
    
    # Check shapes
    with rasterio.open(dsm_file) as src:
        dsm_shape = src.shape
        dsm_crs = src.crs
    
    with rasterio.open(dtm_file) as src:
        dtm_shape = src.shape
        dtm_crs = src.crs
    
    print(f"\nDSM: {dsm_shape} - {dsm_crs}")
    print(f"DTM: {dtm_shape} - {dtm_crs}")
    
    # Check if alignment needed
    needs_alignment = dsm_shape != dtm_shape
    
    if needs_alignment:
        print("\n⚠ Shape mismatch detected - running alignment...")
        
        # Generate output filenames
        dsm_dir = os.path.dirname(dsm_file)
        dtm_dir = os.path.dirname(dtm_file)
        
        dsm_aligned = os.path.join(dsm_dir, "dsm_aligned.tif")
        dtm_aligned = os.path.join(dtm_dir, "dtm_aligned.tif")
        
        # Run alignment
        from align_rasters import align_rasters
        aligned_dsm, aligned_dtm = align_rasters(
            dsm_file, dtm_file, 
            dsm_aligned, dtm_aligned
        )
        
        # Now validate the aligned files
        print("\n" + "="*70)
        print("Now validating aligned files...")
        print("="*70)
        
        from validate_dtm_dsm import DTMDSMValidator
        validator = DTMDSMValidator(aligned_dsm, aligned_dtm)
        is_valid = validator.validate_all()
        
        return is_valid, aligned_dsm, aligned_dtm
    
    else:
        print("\n✓ Shapes match - no alignment needed")
        print("\nRunning validation...")
        
        from validate_dtm_dsm import DTMDSMValidator
        validator = DTMDSMValidator(dsm_file, dtm_file)
        is_valid = validator.validate_all()
        
        return is_valid, dsm_file, dtm_file


if __name__ == "__main__":
    # UPDATE THESE PATHS
    dsm_file = "C:/Users/AdikariAdikari/PycharmProjects/DtmDsmAnalysis/dataset/odm_dem/dsm.tif"
    dtm_file = "C:/Users/AdikariAdikari/PycharmProjects/DtmDsmAnalysis/dataset/odm_dem/dtm.tif"
    
    is_valid, final_dsm, final_dtm = quick_align_and_validate(dsm_file, dtm_file)
    
    print("\n" + "="*70)
    print("FINAL RESULT")
    print("="*70)
    
    if is_valid:
        print("\n✅ SUCCESS! Your files are validated and ready to use:")
        print(f"   DSM: {final_dsm}")
        print(f"   DTM: {final_dtm}")
        print("\nNext steps:")
        print("   1. Use these files for your analysis")
        print("   2. Run: python simple_height_analysis.py (update paths)")
        print("   3. Or: python dsm_dtm_analysis.py (update paths)")
    else:
        print("\n⚠ Validation found issues - check the report above")
        print("   Files: {final_dsm}, {final_dtm}")
    
    print("="*70)
