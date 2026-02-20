"""
Unified Water Depth Extraction - All DTM Methods Comparison

Processes all tiles once and calculates water depths using 4 different DTMs:
  1. WebODM DTM (native photogrammetry)
  2. Bluesky 5m DTM (bicubic resampled to 0.061m)
  3. Fusion DTM (DSM-guided fusion, 5m DTM + 1m DSM)
  4. Kriged DTM (fusion-based ordinary kriging)

All DTMs are pre-resampled to 0.061m and geoid-corrected (+58m).
DSM from WebODM is used as water surface for all methods.
"""

import numpy as np
import rasterio
from rasterio.windows import from_bounds as window_from_bounds
import cv2
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# =======================================================================
# FILE PATHS
# =======================================================================

images_dir = r"C:\Users\AdikariAdikari\PycharmProjects\river_segmentation\dataset\processed_1024\train\images"
masks_dir = r"C:\Users\AdikariAdikari\PycharmProjects\river_segmentation\dataset\processed_1024\train\masks"
orthophoto_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_orthophoto\odm_orthophoto.tif"

# Water surface (same for all methods)
webodm_dsm_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_dem\dsm_aligned.tif"

# DTM files (all at 0.061m, geoid-corrected)
dtm_files = {
    'webodm': r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\odm_dem\dtm_aligned.tif",
    'bluesky': r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\resampled\bluesky_dtm_0061m.tif",
    'fusion': r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\fused\bluesky_dtm_fused_0061m.tif",
    'kriged': r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\kriged\fusion_based_kriged_0061m.tif",
}

working_tiles_file = "working_tiles.txt"
output_dir = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\results_comparison_all_dtms"

tile_size = 1024
stride = 768

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)

# =======================================================================
# UTILITY FUNCTIONS
# =======================================================================

def load_working_tiles(filepath):
    """Parse working_tiles.txt"""
    tiles = []
    current = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Tile ') and ':' in line:
                if current.get('image_file'):
                    tiles.append(current)
                tile_num = int(line.split()[1].rstrip(':'))
                current = {'tile_num': tile_num, 'image_file': None, 'mask_file': None}
            elif line.startswith('Image:'):
                current['image_file'] = line.split('Image:', 1)[1].strip()
            elif line.startswith('Mask:'):
                current['mask_file'] = line.split('Mask:', 1)[1].strip()
    
    if current.get('image_file'):
        tiles.append(current)
    
    return tiles


def tile_position_from_number(tile_number, stride, ortho_width):
    """Calculate tile position from tile number"""
    n_cols = (ortho_width - tile_size) // stride + 1
    row_idx = tile_number // n_cols
    col_idx = tile_number % n_cols
    return (row_idx * stride, col_idx * stride), (row_idx, col_idx)


def load_raster_window(filepath, bounds):
    """Load a raster window and handle nodata"""
    with rasterio.open(filepath) as src:
        window = window_from_bounds(*bounds, transform=src.transform)
        data = src.read(1, window=window).astype(np.float32)
        
        if src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)
        
        # Resize to tile_size if needed
        if data.shape != (tile_size, tile_size):
            from scipy.ndimage import zoom as scipy_zoom
            data = scipy_zoom(data,
                            (tile_size / data.shape[0],
                             tile_size / data.shape[1]), order=1)
        
        return data


# =======================================================================
# MULTI-DTM DEPTH EXTRACTION
# =======================================================================

def extract_depths_all_dtms(mask_path, tile_position, ortho_transform):
    """
    Extract water depths using all 4 DTM methods.
    
    Returns dict with results for each DTM method.
    """
    
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None, "mask_not_found"
    
    mask = (mask > 127).astype(np.uint8)
    if np.sum(mask) == 0:
        return None, "no_water"
    
    row_start, col_start = tile_position
    
    # World bounds of this tile
    left, top = rasterio.transform.xy(ortho_transform, row_start, col_start, offset='ul')
    right, bottom = rasterio.transform.xy(ortho_transform,
                                          row_start + tile_size,
                                          col_start + tile_size, offset='ul')
    bounds = (left, bottom, right, top)
    
    water_coords = np.argwhere(mask > 0)
    rows, cols = water_coords[:, 0], water_coords[:, 1]
    
    # ------------------------------------------------------------------
    # Load DSM (water surface - same for all methods)
    # ------------------------------------------------------------------
    dsm_tile = load_raster_window(webodm_dsm_file, bounds)
    water_surface = dsm_tile[rows, cols]
    
    # ------------------------------------------------------------------
    # Load all DTMs and calculate depths
    # ------------------------------------------------------------------
    results = {
        'mask': mask,
        'water_coords': water_coords,
        'dsm_tile': dsm_tile,
        'water_surface': water_surface,
        'bounds': bounds,
        'dtms': {},
    }
    
    for method_name, dtm_file in dtm_files.items():
        if not os.path.exists(dtm_file):
            print(f"  ⚠️  {method_name} DTM not found: {os.path.basename(dtm_file)}")
            continue
        
        try:
            dtm_tile = load_raster_window(dtm_file, bounds)
            riverbed = dtm_tile[rows, cols]
            depth = water_surface - riverbed
            
            valid_mask = (
                ~np.isnan(water_surface) &
                ~np.isnan(riverbed) &
                (depth > 0.0) &
                (depth < 5.0)  # Physical upper bound
            )
            
            results['dtms'][method_name] = {
                'dtm_tile': dtm_tile,
                'riverbed': riverbed,
                'depth': depth,
                'valid_mask': valid_mask,
                'n_valid': int(np.sum(valid_mask)),
            }
            
        except Exception as e:
            print(f"  ⚠️  {method_name} failed: {str(e)[:50]}")
            continue
    
    # Check if at least one method succeeded
    if not results['dtms']:
        return None, "no_valid_dtms"
    
    # Check if any method has valid depths
    any_valid = any(d['n_valid'] > 0 for d in results['dtms'].values())
    if not any_valid:
        return None, "no_valid_depths"
    
    return results, "ok"


# =======================================================================
# VISUALIZATION
# =======================================================================

def visualize_comparison(tile_num, image_path, results, output_path):
    """Create comparison visualization for all DTM methods"""
    
    n_methods = len(results['dtms'])
    fig, axes = plt.subplots(2, n_methods + 1, figsize=(5*(n_methods+1), 10))
    
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None \
          else np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    
    # Row 1, Col 0: Image + mask
    overlay = img.copy()
    colour = np.zeros_like(overlay)
    colour[results['mask'] > 0] = [0, 255, 255]
    overlay = cv2.addWeighted(overlay, 0.7, colour, 0.3, 0)
    axes[0, 0].imshow(overlay)
    axes[0, 0].set_title('Image + Water Mask', fontweight='bold', fontsize=10)
    axes[0, 0].axis('off')
    
    # Row 2, Col 0: DSM (water surface)
    im = axes[1, 0].imshow(results['dsm_tile'], cmap='terrain')
    axes[1, 0].set_title('DSM (Water Surface)', fontweight='bold', fontsize=10)
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, label='Elevation (m)')
    
    # Columns 1-N: Each DTM method
    for col_idx, (method_name, method_data) in enumerate(results['dtms'].items(), start=1):
        
        # Row 1: DTM
        im1 = axes[0, col_idx].imshow(method_data['dtm_tile'], cmap='terrain')
        axes[0, col_idx].set_title(f'{method_name.upper()} DTM', fontweight='bold', fontsize=10)
        axes[0, col_idx].axis('off')
        plt.colorbar(im1, ax=axes[0, col_idx], fraction=0.046, label='m')
        
        # Row 2: Depth map
        depth_map = np.full((tile_size, tile_size), np.nan)
        rows, cols = results['water_coords'][:, 0], results['water_coords'][:, 1]
        valid = method_data['valid_mask']
        depth_map[rows[valid], cols[valid]] = method_data['depth'][valid]
        
        im2 = axes[1, col_idx].imshow(depth_map, cmap='Blues', vmin=0, vmax=2)
        
        stats_text = (f"Valid: {method_data['n_valid']}\n"
                     f"Mean: {np.mean(method_data['depth'][valid]):.3f}m\n"
                     f"Max: {np.max(method_data['depth'][valid]):.3f}m" if valid.any() else "No data")
        
        axes[1, col_idx].set_title(f'{method_name.upper()} Depth\n{stats_text}',
                                   fontweight='bold', fontsize=9)
        axes[1, col_idx].axis('off')
        plt.colorbar(im2, ax=axes[1, col_idx], fraction=0.046, label='m')
    
    plt.suptitle(f'Tile {tile_num} - DTM Method Comparison', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =======================================================================
# BATCH PROCESSING
# =======================================================================

def batch_extract_all_methods():
    
    print("="*70)
    print("MULTI-DTM WATER DEPTH COMPARISON")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Verify files exist
    print("\nDTM Files:")
    for method, filepath in dtm_files.items():
        exists = "✓" if os.path.exists(filepath) else "✗"
        size_mb = os.path.getsize(filepath) / 1024**2 if os.path.exists(filepath) else 0
        print(f"  {exists} {method:8s}: {os.path.basename(filepath):40s} ({size_mb:.0f} MB)")
    
    # Load orthophoto metadata
    with rasterio.open(orthophoto_file) as src:
        ortho_w, ortho_h = src.width, src.height
        ortho_transform = src.transform
    
    # Load tile list
    if not os.path.exists(working_tiles_file):
        print(f"\n✗ {working_tiles_file} not found!")
        return
    
    working_tiles = load_working_tiles(working_tiles_file)
    print(f"\n✓ Loaded {len(working_tiles)} tiles from {working_tiles_file}")
    print(f"Output: {output_dir}")
    
    # ------------------------------------------------------------------
    # Process tiles
    # ------------------------------------------------------------------
    
    all_results = []
    successful = 0
    failed = no_water = no_data = 0
    error_log = []
    
    print(f"\n{'='*70}")
    print(f"PROCESSING {len(working_tiles)} TILES")
    print(f"{'='*70}\n")
    
    for tile_info in tqdm(working_tiles, desc="Extracting depths"):
        
        tile_num = tile_info['tile_num']
        image_path = os.path.join(images_dir, tile_info['image_file'])
        mask_path = os.path.join(masks_dir, tile_info['mask_file'])
        
        if not os.path.exists(mask_path):
            failed += 1
            continue
        
        tile_position, grid_pos = tile_position_from_number(tile_num, stride, ortho_w)
        
        try:
            results, status = extract_depths_all_dtms(mask_path, tile_position, ortho_transform)
            
            if status == "no_water":
                no_water += 1
                continue
            if status in ("no_valid_depths", "no_valid_dtms", "mask_not_found"):
                no_data += 1
                continue
            
            # Build result row with all methods
            row = {
                'tile_number': int(tile_num),
                'grid_row': int(grid_pos[0]),
                'grid_col': int(grid_pos[1]),
                'pixel_row': int(tile_position[0]),
                'pixel_col': int(tile_position[1]),
                'bounds_left': float(results['bounds'][0]),
                'bounds_bottom': float(results['bounds'][1]),
                'bounds_right': float(results['bounds'][2]),
                'bounds_top': float(results['bounds'][3]),
                'water_pixels': int(np.sum(results['mask'])),
                'image_file': tile_info['image_file'],
                'mask_file': tile_info['mask_file'],
            }
            
            # Add statistics for each DTM method
            for method_name, method_data in results['dtms'].items():
                v = method_data['valid_mask']
                prefix = f'{method_name}_'
                
                if v.any():
                    depths = method_data['depth'][v]
                    row[prefix + 'valid_pixels'] = int(np.sum(v))
                    row[prefix + 'coverage_pct'] = float(np.sum(v) / np.sum(results['mask']) * 100)
                    row[prefix + 'depth_mean'] = float(np.mean(depths))
                    row[prefix + 'depth_median'] = float(np.median(depths))
                    row[prefix + 'depth_std'] = float(np.std(depths))
                    row[prefix + 'depth_min'] = float(np.min(depths))
                    row[prefix + 'depth_max'] = float(np.max(depths))
                    row[prefix + 'riverbed_mean'] = float(np.nanmean(method_data['riverbed'][v]))
                    row[prefix + 'riverbed_std'] = float(np.nanstd(method_data['riverbed'][v]))
                else:
                    row[prefix + 'valid_pixels'] = 0
                    row[prefix + 'coverage_pct'] = 0.0
                    for stat in ['mean', 'median', 'std', 'min', 'max']:
                        row[prefix + 'depth_' + stat] = np.nan
                    row[prefix + 'riverbed_mean'] = np.nan
                    row[prefix + 'riverbed_std'] = np.nan
            
            all_results.append(row)
            successful += 1
            
            # Visualize every 10th successful tile
            if successful % 10 == 1 or successful <= 3:
                viz_path = os.path.join(output_dir, 'visualizations', f'tile_{tile_num:04d}.png')
                visualize_comparison(tile_num, image_path, results, viz_path)
        
        except Exception as e:
            error_log.append(f"Tile {tile_num}: {type(e).__name__}: {str(e)}")
            failed += 1
    
    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        csv_path = os.path.join(output_dir, 'water_depths_all_methods.csv')
        df.to_csv(csv_path, index=False)
        print(f"✓ CSV: {csv_path}")
        
        # Summary statistics
        summary = {
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_tiles': len(working_tiles),
            'successful': successful,
            'failed': failed,
            'no_water': no_water,
            'no_data': no_data,
            'methods': {}
        }
        
        for method in dtm_files.keys():
            prefix = f'{method}_'
            valid_col = prefix + 'valid_pixels'
            
            if valid_col in df.columns:
                method_df = df[df[valid_col] > 0]
                
                if len(method_df) > 0:
                    summary['methods'][method] = {
                        'tiles_with_data': len(method_df),
                        'total_valid_pixels': int(method_df[valid_col].sum()),
                        'mean_coverage_pct': float(method_df[prefix + 'coverage_pct'].mean()),
                        'depth_mean': float(method_df[prefix + 'depth_mean'].mean()),
                        'depth_median': float(method_df[prefix + 'depth_median'].median()),
                        'depth_std': float(method_df[prefix + 'depth_mean'].std()),
                        'depth_min': float(method_df[prefix + 'depth_min'].min()),
                        'depth_max': float(method_df[prefix + 'depth_max'].max()),
                    }
        
        summary_path = os.path.join(output_dir, 'summary_all_methods.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary: {summary_path}")
        
        # Create comparison plots
        create_comparison_plots(df, output_dir)
    
    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"Total tiles: {len(working_tiles)}")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    print(f"  ⊘ No water: {no_water}")
    print(f"  ⊘ No data: {no_data}")
    
    if all_results:
        print(f"\nMethod comparison (tiles with data):")
        for method, stats in summary['methods'].items():
            print(f"\n  {method.upper()}:")
            print(f"    Tiles: {stats['tiles_with_data']}")
            print(f"    Mean depth: {stats['depth_mean']:.3f}m")
            print(f"    Depth range: {stats['depth_min']:.3f} to {stats['depth_max']:.3f}m")
    
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results: {output_dir}")


# =======================================================================
# COMPARISON PLOTS
# =======================================================================

def create_comparison_plots(df, output_dir):
    """Create comparison plots across all DTM methods"""
    
    methods = [m for m in dtm_files.keys() if f'{m}_depth_mean' in df.columns]
    
    if len(methods) < 2:
        return
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Mean depth comparison (boxplot)
    ax1 = fig.add_subplot(gs[0, 0])
    depth_data = [df[f'{m}_depth_mean'].dropna() for m in methods]
    ax1.boxplot(depth_data, labels=[m.upper() for m in methods])
    ax1.set_ylabel('Mean Depth (m)', fontweight='bold')
    ax1.set_title('Depth Distribution by Method', fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # 2. Coverage comparison
    ax2 = fig.add_subplot(gs[0, 1])
    coverage_data = [df[f'{m}_coverage_pct'].dropna() for m in methods]
    ax2.boxplot(coverage_data, labels=[m.upper() for m in methods])
    ax2.set_ylabel('Coverage (%)', fontweight='bold')
    ax2.set_title('Data Coverage by Method', fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # 3. Tiles with valid data
    ax3 = fig.add_subplot(gs[0, 2])
    valid_counts = [(df[f'{m}_valid_pixels'] > 0).sum() for m in methods]
    ax3.bar([m.upper() for m in methods], valid_counts)
    ax3.set_ylabel('Number of Tiles', fontweight='bold')
    ax3.set_title('Tiles with Valid Data', fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')
    
    # 4-6. Pairwise scatter plots (first 3 method pairs)
    if len(methods) >= 2:
        for idx, (i, j) in enumerate([(0,1), (0,2), (1,2)][:3]):
            if i >= len(methods) or j >= len(methods):
                continue
            
            ax = fig.add_subplot(gs[1, idx])
            
            m1, m2 = methods[i], methods[j]
            mask = df[f'{m1}_depth_mean'].notna() & df[f'{m2}_depth_mean'].notna()
            
            if mask.sum() > 0:
                x = df.loc[mask, f'{m1}_depth_mean']
                y = df.loc[mask, f'{m2}_depth_mean']
                
                ax.scatter(x, y, alpha=0.5, s=20)
                
                # 1:1 line
                lims = [min(x.min(), y.min()), max(x.max(), y.max())]
                ax.plot(lims, lims, 'r--', linewidth=2, label='1:1')
                
                # Stats
                from scipy.stats import pearsonr
                r, _ = pearsonr(x, y)
                rmse = np.sqrt(((x - y)**2).mean())
                bias = (y - x).mean()
                
                ax.text(0.05, 0.95, f'r = {r:.3f}\nRMSE = {rmse:.3f}m\nBias = {bias:.3f}m',
                       transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_xlabel(f'{m1.upper()} Depth (m)', fontweight='bold')
                ax.set_ylabel(f'{m2.upper()} Depth (m)', fontweight='bold')
                ax.set_title(f'{m1.upper()} vs {m2.upper()}', fontweight='bold')
                ax.grid(alpha=0.3)
                ax.legend()
    
    # 7. Depth histogram overlays
    ax7 = fig.add_subplot(gs[2, 0])
    for m in methods:
        depths = df[f'{m}_depth_mean'].dropna()
        ax7.hist(depths, bins=30, alpha=0.5, label=m.upper())
    ax7.set_xlabel('Mean Depth (m)', fontweight='bold')
    ax7.set_ylabel('Count', fontweight='bold')
    ax7.set_title('Depth Distribution Comparison', fontweight='bold')
    ax7.legend()
    ax7.grid(alpha=0.3)
    
    # 8. RMSE matrix
    ax8 = fig.add_subplot(gs[2, 1])
    n = len(methods)
    rmse_matrix = np.zeros((n, n))
    
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i != j:
                mask = df[f'{m1}_depth_mean'].notna() & df[f'{m2}_depth_mean'].notna()
                if mask.sum() > 0:
                    rmse = np.sqrt(((df.loc[mask, f'{m1}_depth_mean'] - 
                                   df.loc[mask, f'{m2}_depth_mean'])**2).mean())
                    rmse_matrix[i, j] = rmse
    
    im = ax8.imshow(rmse_matrix, cmap='RdYlGn_r', vmin=0, vmax=rmse_matrix.max())
    ax8.set_xticks(range(n))
    ax8.set_yticks(range(n))
    ax8.set_xticklabels([m.upper() for m in methods], rotation=45)
    ax8.set_yticklabels([m.upper() for m in methods])
    ax8.set_title('RMSE Matrix (m)', fontweight='bold')
    
    for i in range(n):
        for j in range(n):
            if i != j:
                ax8.text(j, i, f'{rmse_matrix[i,j]:.3f}', ha='center', va='center', fontsize=10)
    
    plt.colorbar(im, ax=ax8, fraction=0.046)
    
    # 9. Method agreement table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    table_data = []
    table_data.append(['Method', 'Tiles', 'Mean(m)', 'Std(m)', 'Range(m)'])
    
    for m in methods:
        valid = df[f'{m}_depth_mean'].notna()
        if valid.sum() > 0:
            data = df.loc[valid, f'{m}_depth_mean']
            table_data.append([
                m.upper(),
                f'{valid.sum()}',
                f'{data.mean():.3f}',
                f'{data.std():.3f}',
                f'{data.min():.2f}-{data.max():.2f}'
            ])
    
    table = ax9.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.15, 0.20, 0.20, 0.30])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Header styling
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax9.set_title('Summary Statistics', fontweight='bold', pad=20)
    
    plt.suptitle('DTM Method Comparison - Water Depth Analysis', fontsize=16, fontweight='bold')
    
    plot_path = os.path.join(output_dir, 'comparison_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Comparison plots: {plot_path}")


if __name__ == "__main__":
    batch_extract_all_methods()
