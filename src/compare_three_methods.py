"""
Three-Way DTM Upsampling Method Comparison

Compare bicubic, DSM-guided fusion, and kriging methods on identical pixels.
All three outputs must be spatially aligned (same grid/extent).

Produces:
  - Statistical comparison table
  - Side-by-side visual comparison
  - Difference maps
  - Histogram and profile comparisons
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import pandas as pd
import os

# =======================================================================
# FILES — UPDATE THESE
# =======================================================================

bicubic_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\bicubic\bluesky_dtm_bicubic_matched_0061m.tif"
fusion_file  = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\fused\bluesky_dtm_fused_0061m.tif"
kriging_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\kriged\bluesky_dtm_kriged_masked_0061m.tif"

# Optional: Kriging variance for uncertainty visualization
kriging_variance_file = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\kriged\bluesky_dtm_kriging_variance_masked.tif"

output_comparison_plot = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\method_comparison_full.png"
output_stats_csv = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\dataset\bluesky\method_comparison_stats.csv"

# =======================================================================
# LOAD DATA
# =======================================================================

def load_all_methods():
    print("="*70)
    print("THREE-WAY DTM UPSAMPLING METHOD COMPARISON")
    print("="*70)

    methods = {}
    
    for name, path in [
        ("Bicubic", bicubic_file),
        ("Fusion",  fusion_file),
        ("Kriging", kriging_file),
    ]:
        if not os.path.exists(path):
            print(f"  ✗ {name}: MISSING")
            continue
        
        with rasterio.open(path) as src:
            data = src.read(1)
            trans = src.transform
            crs = src.crs
        
        methods[name] = {
            'data': data,
            'transform': trans,
            'crs': crs,
            'path': path
        }
        
        valid = ~np.isnan(data)
        print(f"  ✓ {name:8s}: {data.shape[0]}×{data.shape[1]} px, "
              f"{valid.sum():,} valid, "
              f"elev {np.nanmin(data):.2f}–{np.nanmax(data):.2f}m")

    # Kriging variance (optional)
    if os.path.exists(kriging_variance_file):
        with rasterio.open(kriging_variance_file) as src:
            methods['Kriging']['variance'] = src.read(1)
        print(f"  ✓ Kriging variance loaded")

    if len(methods) < 2:
        print("\n✗ Need at least 2 methods to compare!")
        return None

    # Check spatial alignment
    ref_shape = methods[list(methods.keys())[0]]['data'].shape
    for name, d in methods.items():
        if d['data'].shape != ref_shape:
            print(f"\n⚠️  WARNING: {name} has different shape {d['data'].shape} "
                  f"vs reference {ref_shape}")
            print("   All methods should have identical grids for valid comparison.")

    return methods

# =======================================================================
# STATISTICS
# =======================================================================

def compute_statistics(methods):
    print("\n" + "="*70)
    print("STATISTICS (on pixels valid in all methods)")
    print("="*70)

    # Find pixels valid in ALL methods
    all_valid = np.ones(methods[list(methods.keys())[0]]['data'].shape, dtype=bool)
    for name, d in methods.items():
        all_valid &= ~np.isnan(d['data'])

    n_common = all_valid.sum()
    print(f"  Common valid pixels: {n_common:,}")

    if n_common == 0:
        print("  ✗ No overlapping valid pixels!")
        return None

    stats = []
    for name, d in methods.items():
        data_valid = d['data'][all_valid]
        stats.append({
            'Method': name,
            'Mean (m)': data_valid.mean(),
            'Std (m)': data_valid.std(),
            'Min (m)': data_valid.min(),
            'Max (m)': data_valid.max(),
            'Median (m)': np.median(data_valid),
            'Valid pixels': n_common
        })

    df = pd.DataFrame(stats)
    print("\n" + df.to_string(index=False))

    # Pairwise differences
    print("\n" + "="*70)
    print("PAIRWISE DIFFERENCES")
    print("="*70)

    pairs = []
    names = list(methods.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            n1, n2 = names[i], names[j]
            d1 = methods[n1]['data'][all_valid]
            d2 = methods[n2]['data'][all_valid]
            diff = d1 - d2

            pairs.append({
                'Comparison': f'{n1} − {n2}',
                'Mean diff (m)': diff.mean(),
                'Std diff (m)': diff.std(),
                'RMSE (m)': np.sqrt((diff**2).mean()),
                'Max abs (m)': np.abs(diff).max(),
                'Correlation': np.corrcoef(d1, d2)[0, 1]
            })

    df_pairs = pd.DataFrame(pairs)
    print("\n" + df_pairs.to_string(index=False))

    # Save to CSV
    os.makedirs(os.path.dirname(output_stats_csv), exist_ok=True)
    with open(output_stats_csv, 'w') as f:
        f.write("# METHOD STATISTICS\n")
        df.to_csv(f, index=False)
        f.write("\n# PAIRWISE DIFFERENCES\n")
        df_pairs.to_csv(f, index=False)
    
    print(f"\n  ✓ Stats saved: {output_stats_csv}")

    return all_valid, df, df_pairs


# =======================================================================
# VISUALIZATION
# =======================================================================

def visualize_comparison(methods, all_valid):
    print("\n" + "="*70)
    print("VISUALIZATION")
    print("="*70)

    n_methods = len(methods)
    names = list(methods.keys())
    
    # Downsample for plotting if needed (avoid matplotlib 74GB allocations)
    MAX_PLOT_DIM = 2000
    ref_shape = methods[names[0]]['data'].shape
    
    if max(ref_shape) > MAX_PLOT_DIM:
        downsample_factor = max(ref_shape) // MAX_PLOT_DIM + 1
        print(f"  Downsampling by {downsample_factor}× for visualization "
              f"({ref_shape[0]}×{ref_shape[1]} → "
              f"{ref_shape[0]//downsample_factor}×{ref_shape[1]//downsample_factor})")
        
        methods_plot = {}
        for name in names:
            methods_plot[name] = methods[name]['data'][::downsample_factor, ::downsample_factor]
        all_valid_plot = all_valid[::downsample_factor, ::downsample_factor]
    else:
        methods_plot = {name: methods[name]['data'] for name in names}
        all_valid_plot = all_valid
    
    # Setup figure
    if n_methods == 3:
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    else:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    pixel_size = 0.061  # km
    
    # 1. Show each method
    for i, name in enumerate(names):
        data = methods_plot[name]
        ax = fig.add_subplot(gs[0, i])
        
        extent = [0, data.shape[1] * pixel_size / 1000,
                  0, data.shape[0] * pixel_size / 1000]
        
        im = ax.imshow(data, cmap='terrain', extent=extent)
        ax.set_title(f'{name} DTM', fontweight='bold', fontsize=12)
        ax.set_xlabel('km'); ax.set_ylabel('km')
        plt.colorbar(im, ax=ax, label='Elevation (m)', fraction=0.046)

    # 2. Difference maps (if 3 methods)
    if n_methods == 3:
        comparisons = [
            (names[0], names[1]),
            (names[0], names[2]),
            (names[1], names[2])
        ]
        
        for i, (n1, n2) in enumerate(comparisons):
            ax = fig.add_subplot(gs[1, i])
            
            d1 = methods_plot[n1]
            d2 = methods_plot[n2]
            diff = np.full_like(d1, np.nan)
            both = ~np.isnan(d1) & ~np.isnan(d2)
            diff[both] = d1[both] - d2[both]
            
            extent = [0, d1.shape[1] * pixel_size / 1000,
                      0, d1.shape[0] * pixel_size / 1000]
            
            vmax = np.nanpercentile(np.abs(diff), 95)
            im = ax.imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=extent)
            ax.set_title(f'Difference: {n1} − {n2}', fontweight='bold', fontsize=11)
            ax.set_xlabel('km'); ax.set_ylabel('km')
            plt.colorbar(im, ax=ax, label='m', fraction=0.046)

    # 3. Histograms (use full resolution data, not downsampled)
    ax_hist = fig.add_subplot(gs[-1, 0])
    for name in names:
        data_valid = methods[name]['data'][all_valid]
        ax_hist.hist(data_valid[::10], bins=50, alpha=0.5, density=True,
                     label=f'{name} (μ={data_valid.mean():.2f}m)')
    ax_hist.set_xlabel('Elevation (m)', fontweight='bold')
    ax_hist.set_ylabel('Density', fontweight='bold')
    ax_hist.set_title('Elevation Distribution', fontweight='bold')
    ax_hist.legend(); ax_hist.grid(alpha=0.3)

    # 4. Profile comparison (use full resolution)
    ax_prof = fig.add_subplot(gs[-1, 1])
    mid_row = methods[names[0]]['data'].shape[0] // 2
    x_prof = np.arange(methods[names[0]]['data'].shape[1]) * pixel_size
    
    for name in names:
        data = methods[name]['data']
        ax_prof.plot(x_prof, data[mid_row, :], alpha=0.7, linewidth=1.5, label=name)
    
    # Add kriging uncertainty band if available
    if 'Kriging' in methods and 'variance' in methods['Kriging']:
        krig_data = methods['Kriging']['data']
        krig_var = methods['Kriging']['variance']
        krig_std = np.sqrt(krig_var)
        ax_prof.fill_between(
            x_prof,
            krig_data[mid_row, :] - 1.96 * krig_std[mid_row, :],
            krig_data[mid_row, :] + 1.96 * krig_std[mid_row, :],
            alpha=0.2, label='Kriging 95% CI'
        )
    
    ax_prof.set_xlabel('Distance (m)', fontweight='bold')
    ax_prof.set_ylabel('Elevation (m)', fontweight='bold')
    ax_prof.set_title('Transect Profile (Centre Row)', fontweight='bold')
    ax_prof.legend(fontsize=9); ax_prof.grid(alpha=0.3)

    # 5. Scatter plots (method vs method) - use full resolution but subsample points
    if n_methods >= 2:
        ax_scatter = fig.add_subplot(gs[-1, 2])
        
        # Plot first two methods
        d1 = methods[names[0]]['data'][all_valid]
        d2 = methods[names[1]]['data'][all_valid]
        
        # Subsample for plotting (every 100th point)
        ax_scatter.scatter(d1[::100], d2[::100], alpha=0.3, s=5)
        
        # 1:1 line
        min_val = min(d1.min(), d2.min())
        max_val = max(d1.max(), d2.max())
        ax_scatter.plot([min_val, max_val], [min_val, max_val],
                       'r--', linewidth=2, label='1:1 line')
        
        r = np.corrcoef(d1, d2)[0, 1]
        rmse = np.sqrt(((d1 - d2)**2).mean())
        
        ax_scatter.set_xlabel(f'{names[0]} elevation (m)', fontweight='bold')
        ax_scatter.set_ylabel(f'{names[1]} elevation (m)', fontweight='bold')
        ax_scatter.set_title(f'{names[0]} vs {names[1]}\n'
                            f'r={r:.3f}, RMSE={rmse:.4f}m',
                            fontweight='bold')
        ax_scatter.legend(); ax_scatter.grid(alpha=0.3)

    plt.suptitle('DTM Upsampling Method Comparison (5m → 0.061m)',
                 fontsize=14, fontweight='bold')
    
    os.makedirs(os.path.dirname(output_comparison_plot), exist_ok=True)
    plt.savefig(output_comparison_plot, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Visualization: {output_comparison_plot}")


# =======================================================================
# MAIN
# =======================================================================

def run_comparison():
    methods = load_all_methods()
    if methods is None:
        return

    result = compute_statistics(methods)
    if result is None:
        return
    
    all_valid, df_stats, df_pairs = result
    
    visualize_comparison(methods, all_valid)

    print("\n" + "="*70)
    print("✅  COMPARISON COMPLETE")
    print("="*70)
    print(f"\n  Statistics: {output_stats_csv}")
    print(f"  Plot:       {output_comparison_plot}")
    print(f"\n  Key findings:")
    
    if len(methods) >= 2:
        names = list(methods.keys())
        mean_vals = [np.nanmean(methods[n]['data'][all_valid]) for n in names]
        print(f"    • Mean elevations: {', '.join([f'{n}={v:.3f}m' for n,v in zip(names, mean_vals)])}")
        
        if not df_pairs.empty:
            max_rmse = df_pairs['RMSE (m)'].max()
            max_pair = df_pairs.loc[df_pairs['RMSE (m)'].idxmax(), 'Comparison']
            print(f"    • Largest RMSE: {max_rmse:.4f}m ({max_pair})")


if __name__ == "__main__":
    run_comparison()
