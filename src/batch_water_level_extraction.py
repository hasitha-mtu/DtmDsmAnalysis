"""
Batch Water Level Extraction

Process ALL working tiles and extract water depths for each
Creates comprehensive analysis and summary statistics
"""

import numpy as np
import rasterio
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.transform import from_bounds
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import os
import json
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# =======================================================================
# YOUR FOLDERS - UPDATE THESE!
# =======================================================================

images_dir = r"C:\Users\AdikariAdikari\PycharmProjects\river_segmentation\dataset\processed_1024\train\images"
masks_dir = r"C:\Users\AdikariAdikari\PycharmProjects\river_segmentation\dataset\processed_1024\train\masks"
orthophoto_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_orthophoto\odm_orthophoto.tif"
dsm_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_dem\dsm_aligned.tif"
dtm_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_dem\dtm_aligned.tif"
output_dir = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\results"

tile_size = 1024
stride = 768

# Create output directory
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)

# =======================================================================
# FUNCTIONS
# =======================================================================

def calculate_tile_position(tile_number, stride, orthophoto_width):
    """Calculate tile position from tile number"""
    n_cols = (orthophoto_width - tile_size) // stride + 1
    tile_row_idx = tile_number // n_cols
    tile_col_idx = tile_number % n_cols
    row_start = tile_row_idx * stride
    col_start = tile_col_idx * stride
    return (row_start, col_start), (tile_row_idx, tile_col_idx)

def extract_water_levels_single_tile(mask_path, tile_position, ortho_transform, 
                                     ortho_crs, dsm_path, dtm_path):
    """Extract water levels from a single tile"""
    
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    mask = (mask > 127).astype(np.uint8)
    
    row_start, col_start = tile_position
    
    # Get world bounds
    left, top = rasterio.transform.xy(ortho_transform, row_start, col_start, offset='ul')
    right, bottom = rasterio.transform.xy(ortho_transform, 
                                          row_start + tile_size, 
                                          col_start + tile_size, 
                                          offset='ul')
    bounds = (left, bottom, right, top)
    
    # Extract water pixels
    water_coords = np.argwhere(mask > 0)
    if len(water_coords) == 0:
        return None  # No water in this tile
    
    rows, cols = water_coords[:, 0], water_coords[:, 1]
    
    # Sample DSM
    with rasterio.open(dsm_path) as src:
        window = window_from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], 
                                   transform=src.transform)
        dsm_tile = src.read(1, window=window)
        dsm_nodata = src.nodata
        
        if dsm_nodata is not None:
            dsm_tile = np.where(dsm_tile == dsm_nodata, np.nan, dsm_tile.astype(np.float32))
        
        # Resize if needed
        if dsm_tile.shape != (tile_size, tile_size):
            zoom_factors = (tile_size / dsm_tile.shape[0], tile_size / dsm_tile.shape[1])
            dsm_tile = zoom(dsm_tile, zoom_factors, order=1)
        
        water_surface = dsm_tile[rows, cols]
    
    # Sample DTM
    with rasterio.open(dtm_path) as src:
        window = window_from_bounds(bounds[0], bounds[1], bounds[2], bounds[3],
                                   transform=src.transform)
        dtm_tile = src.read(1, window=window)
        dtm_nodata = src.nodata
        
        if dtm_nodata is not None:
            dtm_tile = np.where(dtm_tile == dtm_nodata, np.nan, dtm_tile.astype(np.float32))
        
        # Resize if needed
        if dtm_tile.shape != (tile_size, tile_size):
            zoom_factors = (tile_size / dtm_tile.shape[0], tile_size / dtm_tile.shape[1])
            dtm_tile = zoom(dtm_tile, zoom_factors, order=1)
        
        riverbed = dtm_tile[rows, cols]
    
    # Calculate depths
    valid_dsm = ~np.isnan(water_surface)
    valid_dtm = ~np.isnan(riverbed)
    depth = water_surface - riverbed
    valid_mask = valid_dsm & valid_dtm & (depth > 0) & (depth < 10)
    
    if np.sum(valid_mask) == 0:
        return None  # No valid depths
    
    # Return results
    return {
        'mask': mask,
        'water_coords': water_coords,
        'water_surface': water_surface,
        'riverbed': riverbed,
        'depth': depth,
        'valid_mask': valid_mask,
        'bounds': bounds,
        'dsm_tile': dsm_tile,
        'dtm_tile': dtm_tile
    }

def create_summary_visualization(tile_num, image_path, mask_path, results, output_path):
    """Create summary visualization for a single tile"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Load image
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    
    # 1. Original with mask overlay
    overlay = img.copy()
    mask_colored = np.zeros_like(overlay)
    mask_colored[results['mask'] > 0] = [0, 255, 255]
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    axes[0, 0].imshow(overlay)
    axes[0, 0].set_title(f'Tile {tile_num} - Water Mask', fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. DSM
    im2 = axes[0, 1].imshow(results['dsm_tile'], cmap='terrain')
    axes[0, 1].set_title('DSM (Water Surface)', fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], label='Elevation (m)', fraction=0.046)
    
    # 3. Depth map
    depth_map = np.full(results['mask'].shape, np.nan)
    water_coords = results['water_coords']
    rows, cols = water_coords[:, 0], water_coords[:, 1]
    valid = results['valid_mask']
    depth_map[rows[valid], cols[valid]] = results['depth'][valid]
    
    im3 = axes[1, 0].imshow(depth_map, cmap='YlGnBu', vmin=0)
    axes[1, 0].set_title('Water Depth', fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], label='Depth (m)', fraction=0.046)
    
    # 4. Histogram
    if np.sum(valid) > 0:
        depths = results['depth'][valid]
        axes[1, 1].hist(depths, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(np.mean(depths), color='red', linestyle='--',
                          label=f'Mean: {np.mean(depths):.2f}m')
        axes[1, 1].set_xlabel('Depth (m)', fontweight='bold')
        axes[1, 1].set_ylabel('Pixel Count', fontweight='bold')
        axes[1, 1].set_title('Depth Distribution', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

# =======================================================================
# MAIN BATCH PROCESSING
# =======================================================================

def load_working_tiles_from_file(filepath):
    """
    Parse working_tiles.txt created by find_valid_from_files.py
    
    Format:
        Tile 359: DSM 98.5%, DTM 97.2%
          Image: DJI_20250728101825_0553_V_patch_0359.jpg
          Mask:  DJI_20250728101825_0553_V_patch_0359.png
    
    Returns list of dicts with tile_num, image_file, mask_file
    """
    tiles = []
    current = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Tile header line
            if line.startswith('Tile ') and ':' in line:
                if current:
                    tiles.append(current)
                tile_num = int(line.split()[1].rstrip(':'))
                current = {'tile_num': tile_num, 'image_file': None, 'mask_file': None}
            
            # Image filename line
            elif line.startswith('Image:'):
                current['image_file'] = line.split('Image:', 1)[1].strip()
            
            # Mask filename line
            elif line.startswith('Mask:'):
                current['mask_file'] = line.split('Mask:', 1)[1].strip()
        
        # Don't forget the last tile
        if current and current.get('image_file'):
            tiles.append(current)
    
    return tiles


# =======================================================================
# MAIN BATCH PROCESSING
# =======================================================================

def process_all_tiles():
    """Process all working tiles"""
    
    print("="*70)
    print("BATCH WATER LEVEL EXTRACTION")
    print("="*70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load georeferencing info
    with rasterio.open(orthophoto_file) as src:
        ortho_w, ortho_h = src.width, src.height
        ortho_transform = src.transform
        ortho_crs = src.crs
    
    print(f"\nOrthophoto: {ortho_h} × {ortho_w} pixels")
    print(f"Output directory: {output_dir}")
    
    # -----------------------------------------------------------------------
    # Load working tiles with exact filenames
    # -----------------------------------------------------------------------
    
    working_tiles_file = 'working_tiles.txt'
    
    if not os.path.exists(working_tiles_file):
        print(f"\n✗ working_tiles.txt not found!")
        print(f"  Run find_valid_from_files.py first to generate it.")
        return
    
    working_tiles = load_working_tiles_from_file(working_tiles_file)
    print(f"\n✓ Loaded {len(working_tiles)} tiles from working_tiles.txt")
    
    # Verify a sample to make sure filenames exist
    missing_images = 0
    missing_masks = 0
    for tile in working_tiles[:5]:
        img_path = os.path.join(images_dir, tile['image_file'])
        msk_path = os.path.join(masks_dir, tile['mask_file'])
        if not os.path.exists(img_path):
            missing_images += 1
            print(f"  ⚠️  Image not found: {tile['image_file']}")
        if not os.path.exists(msk_path):
            missing_masks += 1
            print(f"  ⚠️  Mask not found: {tile['mask_file']}")
    
    if missing_images > 0 or missing_masks > 0:
        print(f"\n  ⚠️  Warning: Some files are missing!")
        print(f"  Check that images_dir and masks_dir paths are correct:")
        print(f"  images_dir = {images_dir}")
        print(f"  masks_dir  = {masks_dir}")
    else:
        print(f"  ✓ File paths verified")
        print(f"  Example: {working_tiles[0]['image_file']}")
    
    # Initialize results storage
    all_results = []
    successful = 0
    failed = 0
    no_water = 0
    no_data = 0
    error_log = []  # Track what went wrong
    
    # Process each tile
    print(f"\n{'='*70}")
    print(f"PROCESSING {len(working_tiles)} TILES")
    print(f"{'='*70}\n")
    
    for tile_info in tqdm(working_tiles, desc="Processing tiles"):
        
        tile_num = tile_info['tile_num']
        
        # Use exact filenames from working_tiles.txt
        image_path = os.path.join(images_dir, tile_info['image_file'])
        mask_path = os.path.join(masks_dir, tile_info['mask_file'])
        
        # Check files exist
        if not os.path.exists(image_path):
            error_log.append(f"Tile {tile_num}: Image not found: {image_path}")
            failed += 1
            continue
        
        if not os.path.exists(mask_path):
            error_log.append(f"Tile {tile_num}: Mask not found: {mask_path}")
            failed += 1
            continue
        
        # Calculate tile position
        tile_position, grid_position = calculate_tile_position(tile_num, stride, ortho_w)
        
        # Extract water levels
        try:
            results = extract_water_levels_single_tile(
                mask_path, tile_position, ortho_transform, ortho_crs,
                dsm_file, dtm_file
            )
            
            if results is None:
                no_data += 1
                continue
            
            # Calculate statistics
            valid = results['valid_mask']
            
            # Skip if no water pixels in mask at all
            if np.sum(results['mask']) == 0:
                no_water += 1
                continue
            
            # Skip if no valid depths
            if np.sum(valid) == 0:
                no_data += 1
                continue
            
            depths = results['depth'][valid]
            surfaces = results['water_surface'][valid]
            riverbeds = results['riverbed'][valid]
            
            stats = {
                'tile_number': int(tile_num),
                'grid_row': int(grid_position[0]),
                'grid_col': int(grid_position[1]),
                'pixel_row': int(tile_position[0]),
                'pixel_col': int(tile_position[1]),
                'bounds_left': float(results['bounds'][0]),
                'bounds_bottom': float(results['bounds'][1]),
                'bounds_right': float(results['bounds'][2]),
                'bounds_top': float(results['bounds'][3]),
                'water_pixels': int(np.sum(results['mask'])),
                'valid_depths': int(np.sum(valid)),
                'coverage_pct': float(np.sum(valid) / np.sum(results['mask']) * 100),
                'depth_mean': float(np.mean(depths)),
                'depth_median': float(np.median(depths)),
                'depth_std': float(np.std(depths)),
                'depth_min': float(np.min(depths)),
                'depth_max': float(np.max(depths)),
                'surface_mean': float(np.mean(surfaces)),
                'surface_std': float(np.std(surfaces)),
                'riverbed_mean': float(np.mean(riverbeds)),
                'riverbed_std': float(np.std(riverbeds)),
                'image_file': tile_info['image_file'],
                'mask_file': tile_info['mask_file']
            }
            
            all_results.append(stats)
            successful += 1
            
            # Create visualization (every 10th successful tile)
            if successful % 10 == 1 or successful <= 3:
                viz_path = os.path.join(output_dir, 'visualizations', 
                                       f'tile_{tile_num:04d}.png')
                create_summary_visualization(tile_num, image_path, mask_path, 
                                           results, viz_path)
            
        except Exception as e:
            error_msg = f"Tile {tile_num}: {type(e).__name__}: {str(e)}"
            error_log.append(error_msg)
            failed += 1
            continue
    
    # Print error log sample
    if error_log:
        print(f"\n⚠️  First 10 errors encountered:")
        for err in error_log[:10]:
            print(f"   {err}")
    
    # =======================================================================
    # SAVE RESULTS
    # =======================================================================
    
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}\n")
    
    # Save as CSV
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(output_dir, 'water_depths_all_tiles.csv')
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved CSV: {csv_path}")
        
        # Save as JSON
        json_path = os.path.join(output_dir, 'water_depths_all_tiles.json')
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"✓ Saved JSON: {json_path}")
        
        # Create summary statistics
        summary = {
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_tiles_processed': len(working_tiles),
            'successful': successful,
            'failed': failed,
            'no_water': no_water,
            'no_data': no_data,
            'total_water_pixels': int(df['water_pixels'].sum()),
            'total_valid_depths': int(df['valid_depths'].sum()),
            'overall_depth_mean': float(df['depth_mean'].mean()),
            'overall_depth_median': float(df['depth_median'].median()),
            'overall_depth_std': float(df['depth_std'].mean()),
            'overall_surface_mean': float(df['surface_mean'].mean()),
            'overall_riverbed_mean': float(df['riverbed_mean'].mean()),
            'depth_range': {
                'min': float(df['depth_min'].min()),
                'max': float(df['depth_max'].max())
            }
        }
        
        summary_path = os.path.join(output_dir, 'summary_statistics.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved summary: {summary_path}")
        
        # Create summary plot
        create_summary_plots(df, output_dir)
        
    else:
        print("✗ No successful results to save")
    
    # =======================================================================
    # FINAL SUMMARY
    # =======================================================================
    
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE")
    print(f"{'='*70}\n")
    
    print(f"Total tiles: {len(working_tiles)}")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    print(f"  ⊘ No water: {no_water}")
    print(f"  ⊘ No DSM/DTM data: {no_data}")
    
    if all_results:
        print(f"\n📊 Overall Statistics:")
        print(f"  Total water pixels: {summary['total_water_pixels']:,}")
        print(f"  Total valid depths: {summary['total_valid_depths']:,}")
        print(f"  Mean depth: {summary['overall_depth_mean']:.2f}m")
        print(f"  Depth range: {summary['depth_range']['min']:.2f} to {summary['depth_range']['max']:.2f}m")
        print(f"  Mean water surface: {summary['overall_surface_mean']:.2f}m")
        print(f"  Mean riverbed: {summary['overall_riverbed_mean']:.2f}m")
    
    print(f"\n✓ All results saved to: {output_dir}")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def create_summary_plots(df, output_dir):
    """Create summary plots from all tiles"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Depth distribution (all tiles)
    axes[0, 0].hist(df['depth_mean'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Mean Depth (m)', fontweight='bold')
    axes[0, 0].set_ylabel('Tile Count', fontweight='bold')
    axes[0, 0].set_title('Distribution of Mean Depths Across Tiles', fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Coverage percentage
    axes[0, 1].hist(df['coverage_pct'], bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Valid Depth Coverage (%)', fontweight='bold')
    axes[0, 1].set_ylabel('Tile Count', fontweight='bold')
    axes[0, 1].set_title('Data Quality: Valid Depth Coverage', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Spatial distribution of depths
    scatter = axes[0, 2].scatter(df['grid_col'], df['grid_row'], 
                                c=df['depth_mean'], cmap='YlGnBu', s=50)
    axes[0, 2].set_xlabel('Grid Column', fontweight='bold')
    axes[0, 2].set_ylabel('Grid Row', fontweight='bold')
    axes[0, 2].set_title('Spatial Distribution of Mean Depths', fontweight='bold')
    axes[0, 2].invert_yaxis()
    plt.colorbar(scatter, ax=axes[0, 2], label='Mean Depth (m)')
    
    # 4. Water surface elevation
    axes[1, 0].hist(df['surface_mean'], bins=30, edgecolor='black', alpha=0.7, color='blue')
    axes[1, 0].set_xlabel('Water Surface Elevation (m)', fontweight='bold')
    axes[1, 0].set_ylabel('Tile Count', fontweight='bold')
    axes[1, 0].set_title('Water Surface Elevation Distribution', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # 5. Depth range per tile
    axes[1, 1].scatter(df['tile_number'], df['depth_max'] - df['depth_min'], alpha=0.6)
    axes[1, 1].set_xlabel('Tile Number', fontweight='bold')
    axes[1, 1].set_ylabel('Depth Range (m)', fontweight='bold')
    axes[1, 1].set_title('Depth Variability Across Tiles', fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    # 6. Summary statistics text
    axes[1, 2].axis('off')
    summary_text = f"""
SUMMARY STATISTICS

Tiles Processed: {len(df)}

Water Depths:
  Mean: {df['depth_mean'].mean():.2f} ± {df['depth_mean'].std():.2f} m
  Median: {df['depth_median'].median():.2f} m
  Range: {df['depth_min'].min():.2f} to {df['depth_max'].max():.2f} m

Elevations (WGS84 Ellipsoid):
  Water Surface: {df['surface_mean'].mean():.2f} ± {df['surface_mean'].std():.2f} m
  Riverbed: {df['riverbed_mean'].mean():.2f} ± {df['riverbed_mean'].std():.2f} m

Coverage:
  Total water pixels: {df['water_pixels'].sum():,}
  Total valid depths: {df['valid_depths'].sum():,}
  Mean coverage: {df['coverage_pct'].mean():.1f}%
    """
    
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                   fontfamily='monospace', fontsize=10, verticalalignment='top')
    
    plt.suptitle('Water Depth Analysis - All Tiles Summary', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    summary_plot_path = os.path.join(output_dir, 'summary_plots.png')
    plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved summary plots: {summary_plot_path}")
    plt.close()

# =======================================================================
# RUN
# =======================================================================

if __name__ == "__main__":
    process_all_tiles()
