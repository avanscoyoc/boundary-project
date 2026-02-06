"""
Script 05: Analyze Results

Performs statistical analysis and creates visualizations for edge effects in
protected areas.

Workflow:
1. Load transect and WDPA datasets
2. Compute summary statistics and distributions
3. Fit statistical models (ANOVA, regression)
4. Create publication-quality figures
5. Export results to text files and plots

BEFORE RUNNING:
Ensure processed datasets exist:
- results/transect_df_{INDEX_NAME}.parquet
- results/wdpa_df_{INDEX_NAME}.parquet

OUTPUT:
- results/figures/{INDEX_NAME}_summary_statistics.txt
- results/figures/{INDEX_NAME}_anova_results.txt
- results/figures/{INDEX_NAME}_edge_extent_model.txt
- results/figures/{INDEX_NAME}_edge_intensity_model.txt
- results/figures/{INDEX_NAME}_*.png (various plots)

For advanced analysis, see src/7_analysis.ipynb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from modules import config
from modules.plotting import (
    classify_trend,
    save_summary_statistics,
    create_correlation_plot,
    create_distribution_plots
)

print("="*80)
print("SCRIPT 05: Analyze Results")
print("="*80)

# Validate configuration
print(f"\nCurrent index: {config.INDEX_NAME.upper()}")
print(f"Description: {config.INDEX_CONFIGS[config.INDEX_NAME]['description']}")

# Paths
output_dir = config.RESULTS_DIR
figures_dir = config.FIGURES_DIR
transect_path = output_dir / f'transect_df_{config.INDEX_NAME}.parquet'
wdpa_path = output_dir / f'wdpa_df_{config.INDEX_NAME}.parquet'

# Verify input files exist
if not transect_path.exists():
    print(f"\nERROR: Transect dataset not found: {transect_path}")
    print("Please run script 04_compute_edge_metrics.py first")
    exit(1)

if not wdpa_path.exists():
    print(f"\nERROR: WDPA dataset not found: {wdpa_path}")
    print("Please run script 04_compute_edge_metrics.py first")
    exit(1)

# Load data
print("\n" + "="*80)
print("Loading Datasets")
print("="*80)

print(f"Loading transect data: {transect_path}")
transect_df = pd.read_parquet(transect_path)
print(f"  Transects: {len(transect_df):,}")

print(f"Loading WDPA data: {wdpa_path}")
wdpa_df = pd.read_parquet(wdpa_path)
print(f"  Protected areas: {len(wdpa_df):,}")

# Basic data quality checks
print("\nData quality:")
print(f"  Missing transect values: {transect_df.isnull().sum().sum()}")
print(f"  Missing WDPA values: {wdpa_df.isnull().sum().sum()}")

# Step 1: Summary Statistics
print("\n" + "="*80)
print("STEP 1: Computing Summary Statistics")
print("="*80)

summary_output = figures_dir / f'{config.INDEX_NAME}_summary_statistics.txt'
save_summary_statistics(wdpa_df, summary_output)
print(f"Saved summary statistics to: {summary_output}")

# Show key metrics
if 'mean_cohens_d' in wdpa_df.columns:
    print("\nKey edge effect metrics:")
    print(f"  Mean Cohen's d: {wdpa_df['mean_cohens_d'].mean():.3f}")
    print(f"  Median Cohen's d: {wdpa_df['mean_cohens_d'].median():.3f}")
    print(f"  Std Cohen's d: {wdpa_df['mean_cohens_d'].std():.3f}")

# Classify trends
if 'mean_cohens_d' in wdpa_df.columns:
    wdpa_df['trend'] = wdpa_df['mean_cohens_d'].apply(classify_trend)
    trend_counts = wdpa_df['trend'].value_counts()
    print("\nEdge effect trends:")
    for trend, count in trend_counts.items():
        pct = count / len(wdpa_df) * 100
        print(f"  {trend}: {count} ({pct:.1f}%)")

# Step 2: Statistical Tests
print("\n" + "="*80)
print("STEP 2: Statistical Analysis")
print("="*80)

# ANOVA by biome
if 'biome_name' in wdpa_df.columns and 'mean_cohens_d' in wdpa_df.columns:
    print("\nANOVA: Edge effects by biome...")
    biome_groups = [group['mean_cohens_d'].dropna() 
                    for name, group in wdpa_df.groupby('biome_name')]
    f_stat, p_value = stats.f_oneway(*biome_groups)
    
    anova_output = figures_dir / f'{config.INDEX_NAME}_anova_results.txt'
    with open(anova_output, 'w') as f:
        f.write(f"ANOVA: Edge Effects ({config.INDEX_NAME.upper()}) by Biome\n")
        f.write("="*60 + "\n\n")
        f.write(f"F-statistic: {f_stat:.4f}\n")
        f.write(f"p-value: {p_value:.4e}\n\n")
        
        if p_value < 0.05:
            f.write("Result: Significant difference between biomes (p < 0.05)\n\n")
        else:
            f.write("Result: No significant difference between biomes (p >= 0.05)\n\n")
        
        f.write("Mean Cohen's d by biome:\n")
        for name, group in wdpa_df.groupby('biome_name'):
            mean_d = group['mean_cohens_d'].mean()
            f.write(f"  {name}: {mean_d:.3f}\n")
    
    print(f"  F={f_stat:.2f}, p={p_value:.2e}")
    print(f"  Saved to: {anova_output}")

# Simple regression models
if 'area_km2' in wdpa_df.columns and 'mean_cohens_d' in wdpa_df.columns:
    print("\nRegression: Edge extent (area) model...")
    
    # Log-transform area
    wdpa_df['log_area'] = np.log10(wdpa_df['area_km2'])
    
    # Simple linear regression
    from scipy.stats import linregress
    mask = wdpa_df[['log_area', 'mean_cohens_d']].notna().all(axis=1)
    slope, intercept, r_value, p_value, se = linregress(
        wdpa_df.loc[mask, 'log_area'],
        wdpa_df.loc[mask, 'mean_cohens_d']
    )
    
    extent_output = figures_dir / f'{config.INDEX_NAME}_edge_extent_model.txt'
    with open(extent_output, 'w') as f:
        f.write(f"Linear Regression: Edge Extent ({config.INDEX_NAME.upper()})\n")
        f.write("="*60 + "\n\n")
        f.write("Model: mean_cohens_d ~ log10(area_km2)\n\n")
        f.write(f"Slope: {slope:.4f}\n")
        f.write(f"Intercept: {intercept:.4f}\n")
        f.write(f"R²: {r_value**2:.4f}\n")
        f.write(f"p-value: {p_value:.4e}\n")
    
    print(f"  R²={r_value**2:.3f}, p={p_value:.2e}")
    print(f"  Saved to: {extent_output}")

# Step 3: Create Visualizations
print("\n" + "="*80)
print("STEP 3: Creating Visualizations")
print("="*80)

# Distribution plots
if 'mean_cohens_d' in wdpa_df.columns:
    print("\nCreating distribution plots...")
    create_distribution_plots(
        wdpa_df, 
        'mean_cohens_d',
        f"{config.INDEX_NAME.upper()} Edge Effects",
        figures_dir / f'{config.INDEX_NAME}_distribution.png'
    )
    print(f"  Saved: {config.INDEX_NAME}_distribution.png")

# Correlation plot
numeric_cols = wdpa_df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    print("\nCreating correlation heatmap...")
    # Select key columns for correlation
    corr_cols = [c for c in ['mean_cohens_d', 'area_km2', 'num_transects', 
                              'STATUS_YR', 'REP_AREA'] if c in numeric_cols]
    if len(corr_cols) > 1:
        create_correlation_plot(
            wdpa_df[corr_cols],
            figures_dir / f'{config.INDEX_NAME}_correlation.png'
        )
        print(f"  Saved: {config.INDEX_NAME}_correlation.png")

# Biome comparison plot
if 'biome_name' in wdpa_df.columns and 'mean_cohens_d' in wdpa_df.columns:
    print("\nCreating biome comparison plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Box plot by biome
    biome_data = wdpa_df.groupby('biome_name')['mean_cohens_d'].apply(list)
    positions = range(len(biome_data))
    
    bp = ax.boxplot(biome_data.values, positions=positions, patch_artist=True)
    
    # Style
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(biome_data.index, rotation=45, ha='right')
    ax.set_ylabel("Cohen's d")
    ax.set_title(f"{config.INDEX_NAME.upper()} Edge Effects by Biome")
    ax.axhline(0, color='red', linestyle='--', alpha=0.3, label='No effect')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(figures_dir / f'{config.INDEX_NAME}_biome_comparison.png', dpi=300)
    plt.close()
    
    print(f"  Saved: {config.INDEX_NAME}_biome_comparison.png")

# Area vs edge effect plot
if 'area_km2' in wdpa_df.columns and 'mean_cohens_d' in wdpa_df.columns:
    print("\nCreating area vs edge effect plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot with log-scale x-axis
    ax.scatter(wdpa_df['area_km2'], wdpa_df['mean_cohens_d'], 
              alpha=0.5, s=20)
    
    ax.set_xscale('log')
    ax.set_xlabel("Protected Area Size (km²)")
    ax.set_ylabel("Cohen's d")
    ax.set_title(f"{config.INDEX_NAME.upper()} Edge Effects vs PA Size")
    ax.axhline(0, color='red', linestyle='--', alpha=0.3, label='No effect')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(figures_dir / f'{config.INDEX_NAME}_area_vs_edge.png', dpi=300)
    plt.close()
    
    print(f"  Saved: {config.INDEX_NAME}_area_vs_edge.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"Results saved to: {figures_dir}")
print("\nGenerated files:")
for txt_file in sorted(figures_dir.glob(f'{config.INDEX_NAME}_*.txt')):
    print(f"  - {txt_file.name}")
for png_file in sorted(figures_dir.glob(f'{config.INDEX_NAME}_*.png')):
    print(f"  - {png_file.name}")

print("\nFor advanced analysis (time series, spatial patterns, etc.):")
print("  See: src/7_analysis.ipynb")
print("="*80)
