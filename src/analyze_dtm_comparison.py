"""
Post-Processing Analysis: DTM Method Comparison

Generates detailed comparison reports from batch_compare_all_dtms.py output.
Creates method-specific performance metrics and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from scipy.stats import pearsonr, ttest_rel

# =======================================================================
# INPUT
# =======================================================================

results_dir = r"C:\Users\AdikariAdikari\PycharmProjects\DtmDsmAnalysis\results_comparison_all_dtms"
csv_file = os.path.join(results_dir, 'water_depths_all_methods.csv')
output_dir = os.path.join(results_dir, 'analysis')

os.makedirs(output_dir, exist_ok=True)

# =======================================================================
# LOAD DATA
# =======================================================================

print("="*70)
print("DTM METHOD COMPARISON ANALYSIS")
print("="*70)

if not os.path.exists(csv_file):
    print(f"\n✗ CSV file not found: {csv_file}")
    print("Run batch_compare_all_dtms.py first!")
    exit()

df = pd.read_csv(csv_file)
print(f"\n✓ Loaded {len(df)} tiles from {os.path.basename(csv_file)}")

# Detect available methods
methods = []
for col in df.columns:
    if col.endswith('_depth_mean'):
        method = col.replace('_depth_mean', '')
        methods.append(method)

print(f"✓ Detected {len(methods)} methods: {', '.join(methods)}")

# =======================================================================
# STATISTICAL COMPARISON
# =======================================================================

def compute_pairwise_stats(df, methods):
    """Compute pairwise comparison statistics"""
    
    stats = []
    
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i >= j:  # Skip diagonal and duplicates
                continue
            
            # Find tiles with valid data in both methods
            mask = (df[f'{m1}_depth_mean'].notna() & 
                   df[f'{m2}_depth_mean'].notna() &
                   (df[f'{m1}_valid_pixels'] > 10) &  # Minimum pixels threshold
                   (df[f'{m2}_valid_pixels'] > 10))
            
            if mask.sum() < 3:  # Need at least 3 samples
                continue
            
            depths1 = df.loc[mask, f'{m1}_depth_mean'].values
            depths2 = df.loc[mask, f'{m2}_depth_mean'].values
            
            # Compute statistics
            diff = depths2 - depths1
            
            r, p_corr = pearsonr(depths1, depths2)
            _, p_ttest = ttest_rel(depths1, depths2)
            
            stats.append({
                'method_1': m1,
                'method_2': m2,
                'n_tiles': mask.sum(),
                'correlation_r': r,
                'correlation_p': p_corr,
                'rmse': np.sqrt((diff**2).mean()),
                'mae': np.abs(diff).mean(),
                'bias': diff.mean(),
                'bias_std': diff.std(),
                'ttest_p': p_ttest,
                'max_abs_diff': np.abs(diff).max(),
            })
    
    return pd.DataFrame(stats)


print("\n" + "="*70)
print("PAIRWISE COMPARISON STATISTICS")
print("="*70)

stats_df = compute_pairwise_stats(df, methods)

if len(stats_df) > 0:
    print("\n" + stats_df.to_string(index=False))
    
    stats_path = os.path.join(output_dir, 'pairwise_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"\n✓ Saved: {stats_path}")

# =======================================================================
# METHOD PERFORMANCE SUMMARY
# =======================================================================

print("\n" + "="*70)
print("METHOD PERFORMANCE SUMMARY")
print("="*70)

performance = []

for method in methods:
    valid = df[f'{method}_valid_pixels'] > 0
    
    if valid.sum() == 0:
        print(f"\n{method.upper()}: No valid data")
        continue
    
    method_df = df[valid]
    
    perf = {
        'method': method,
        'tiles_with_data': valid.sum(),
        'total_tiles': len(df),
        'coverage_pct': valid.sum() / len(df) * 100,
        'mean_valid_pixels': method_df[f'{method}_valid_pixels'].mean(),
        'mean_coverage_per_tile': method_df[f'{method}_coverage_pct'].mean(),
        'depth_mean': method_df[f'{method}_depth_mean'].mean(),
        'depth_median': method_df[f'{method}_depth_median'].median(),
        'depth_std': method_df[f'{method}_depth_mean'].std(),
        'depth_min': method_df[f'{method}_depth_min'].min(),
        'depth_max': method_df[f'{method}_depth_max'].max(),
        'riverbed_mean': method_df[f'{method}_riverbed_mean'].mean(),
        'riverbed_std': method_df[f'{method}_riverbed_mean'].std(),
    }
    
    performance.append(perf)
    
    print(f"\n{method.upper()}:")
    print(f"  Tiles with data:    {perf['tiles_with_data']} / {perf['total_tiles']} ({perf['coverage_pct']:.1f}%)")
    print(f"  Avg pixels/tile:    {perf['mean_valid_pixels']:.0f}")
    print(f"  Depth (mean ± std): {perf['depth_mean']:.3f} ± {perf['depth_std']:.3f} m")
    print(f"  Depth range:        {perf['depth_min']:.3f} to {perf['depth_max']:.3f} m")
    print(f"  Riverbed elevation: {perf['riverbed_mean']:.2f} ± {perf['riverbed_std']:.2f} m")

perf_df = pd.DataFrame(performance)
perf_path = os.path.join(output_dir, 'method_performance.csv')
perf_df.to_csv(perf_path, index=False)
print(f"\n✓ Saved: {perf_path}")

# =======================================================================
# ADVANCED VISUALIZATIONS
# =======================================================================

print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# 1. Bland-Altman plots (difference vs average)
if len(methods) >= 2:
    fig, axes = plt.subplots(1, min(3, len(methods)-1), figsize=(18, 5))
    if len(methods) == 2:
        axes = [axes]
    
    plot_idx = 0
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            if plot_idx >= 3:
                break
            
            m1, m2 = methods[i], methods[j]
            
            mask = (df[f'{m1}_depth_mean'].notna() & 
                   df[f'{m2}_depth_mean'].notna())
            
            if mask.sum() > 0:
                d1 = df.loc[mask, f'{m1}_depth_mean']
                d2 = df.loc[mask, f'{m2}_depth_mean']
                
                avg = (d1 + d2) / 2
                diff = d2 - d1
                
                axes[plot_idx].scatter(avg, diff, alpha=0.5, s=20)
                axes[plot_idx].axhline(0, color='black', linestyle='-', linewidth=1)
                axes[plot_idx].axhline(diff.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {diff.mean():.3f}m')
                axes[plot_idx].axhline(diff.mean() + 1.96*diff.std(), color='red', linestyle=':', linewidth=1, label=f'±1.96 SD')
                axes[plot_idx].axhline(diff.mean() - 1.96*diff.std(), color='red', linestyle=':', linewidth=1)
                
                axes[plot_idx].set_xlabel(f'Average of {m1.upper()} and {m2.upper()} (m)', fontweight='bold')
                axes[plot_idx].set_ylabel(f'{m2.upper()} - {m1.upper()} (m)', fontweight='bold')
                axes[plot_idx].set_title(f'Bland-Altman: {m1.upper()} vs {m2.upper()}', fontweight='bold')
                axes[plot_idx].grid(alpha=0.3)
                axes[plot_idx].legend()
                
                plot_idx += 1
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bland_altman_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Bland-Altman plots")

# 2. Correlation heatmap
if len(methods) >= 2:
    corr_matrix = np.zeros((len(methods), len(methods)))
    
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            mask = df[f'{m1}_depth_mean'].notna() & df[f'{m2}_depth_mean'].notna()
            if mask.sum() > 2:
                r, _ = pearsonr(df.loc[mask, f'{m1}_depth_mean'], 
                              df.loc[mask, f'{m2}_depth_mean'])
                corr_matrix[i, j] = r
            else:
                corr_matrix[i, j] = np.nan
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(methods)))
    ax.set_yticks(range(len(methods)))
    ax.set_xticklabels([m.upper() for m in methods], rotation=45, ha='right')
    ax.set_yticklabels([m.upper() for m in methods])
    
    for i in range(len(methods)):
        for j in range(len(methods)):
            if not np.isnan(corr_matrix[i, j]):
                text = ax.text(j, i, f'{corr_matrix[i,j]:.3f}',
                             ha='center', va='center', color='black', fontweight='bold')
    
    ax.set_title('Correlation Matrix (Pearson r)', fontweight='bold', fontsize=14, pad=20)
    plt.colorbar(im, ax=ax, fraction=0.046, label='Correlation')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Correlation heatmap")

# 3. Violin plots
fig, ax = plt.subplots(figsize=(12, 6))
depth_data = []
labels = []

for m in methods:
    valid = df[f'{m}_depth_mean'].notna()
    if valid.sum() > 0:
        depth_data.append(df.loc[valid, f'{m}_depth_mean'].values)
        labels.append(m.upper())

parts = ax.violinplot(depth_data, positions=range(len(labels)), 
                      widths=0.7, showmeans=True, showmedians=True)

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_ylabel('Depth (m)', fontweight='bold', fontsize=12)
ax.set_title('Depth Distribution by DTM Method', fontweight='bold', fontsize=14)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'violin_plots.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Violin plots")

# 4. Spatial distribution of agreement
if len(methods) >= 2:
    m1, m2 = methods[0], methods[1]
    
    mask = df[f'{m1}_depth_mean'].notna() & df[f'{m2}_depth_mean'].notna()
    if mask.sum() > 0:
        diff = df.loc[mask, f'{m2}_depth_mean'] - df.loc[mask, f'{m1}_depth_mean']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        scatter = ax.scatter(df.loc[mask, 'pixel_col'], 
                           df.loc[mask, 'pixel_row'],
                           c=diff, cmap='RdBu_r', 
                           vmin=-0.5, vmax=0.5, s=50, alpha=0.7)
        
        ax.set_xlabel('Pixel Column', fontweight='bold', fontsize=12)
        ax.set_ylabel('Pixel Row', fontweight='bold', fontsize=12)
        ax.set_title(f'Spatial Distribution of Depth Difference\n({m2.upper()} - {m1.upper()})',
                    fontweight='bold', fontsize=14)
        ax.invert_yaxis()
        
        cbar = plt.colorbar(scatter, ax=ax, label='Depth Difference (m)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'spatial_agreement.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Spatial agreement map")

# =======================================================================
# SUMMARY REPORT
# =======================================================================

print("\n" + "="*70)
print("GENERATING SUMMARY REPORT")
print("="*70)

report = []
report.append("="*70)
report.append("DTM METHOD COMPARISON - SUMMARY REPORT")
report.append("="*70)
report.append("")

report.append("DATA OVERVIEW")
report.append("-"*70)
report.append(f"Total tiles processed: {len(df)}")
report.append(f"Methods compared: {', '.join([m.upper() for m in methods])}")
report.append("")

report.append("METHOD PERFORMANCE")
report.append("-"*70)
for _, row in perf_df.iterrows():
    report.append(f"\n{row['method'].upper()}:")
    report.append(f"  Coverage:         {row['coverage_pct']:.1f}% ({row['tiles_with_data']}/{row['total_tiles']} tiles)")
    report.append(f"  Mean depth:       {row['depth_mean']:.3f} ± {row['depth_std']:.3f} m")
    report.append(f"  Depth range:      {row['depth_min']:.3f} to {row['depth_max']:.3f} m")
    report.append(f"  Riverbed elev:    {row['riverbed_mean']:.2f} ± {row['riverbed_std']:.2f} m")

if len(stats_df) > 0:
    report.append("")
    report.append("PAIRWISE COMPARISON")
    report.append("-"*70)
    for _, row in stats_df.iterrows():
        report.append(f"\n{row['method_1'].upper()} vs {row['method_2'].upper()} (n={row['n_tiles']} tiles):")
        report.append(f"  Correlation:      r = {row['correlation_r']:.3f} (p = {row['correlation_p']:.4f})")
        report.append(f"  Agreement:        RMSE = {row['rmse']:.3f} m, MAE = {row['mae']:.3f} m")
        report.append(f"  Bias:             {row['bias']:.3f} ± {row['bias_std']:.3f} m")
        report.append(f"  Max difference:   {row['max_abs_diff']:.3f} m")
        
        sig = "significant" if row['ttest_p'] < 0.05 else "not significant"
        report.append(f"  Paired t-test:    p = {row['ttest_p']:.4f} ({sig})")

report.append("")
report.append("="*70)

report_text = "\n".join(report)
print(report_text)

report_path = os.path.join(output_dir, 'comparison_report.txt')
with open(report_path, 'w') as f:
    f.write(report_text)

print(f"\n✓ Full report saved: {report_path}")
print(f"✓ All outputs in: {output_dir}")
print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
