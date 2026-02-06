"""
Statistical modeling and visualization functions.

This module contains functions for:
- Classifying temporal trends in edge metrics
- Fitting mixed-effects models
- Generating publication figures
- Running ANOVA tests
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path


def classify_trend(group, metric='edge_extent', alpha=0.05):
    """
    Classify temporal trend for a protected area using linear regression.
    
    Parameters
    ----------
    group : DataFrame
        Time series data for single WDPA_PID with 'year' and metric columns
    metric : str, optional
        Column name to analyze. Default 'edge_extent'.
    alpha : float, optional
        Significance threshold for p-value. Default 0.05.
    
    Returns
    -------
    str
        Trend classification: 'sig_increase', 'sig_decrease', 'increase', 
        'decrease', or 'no_change'
    
    Notes
    -----
    Significant trends (p < alpha) are prefixed with 'sig_'.
    Non-significant trends indicate direction based on slope sign.
    """
    X = group['year'].values
    y = group[metric].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
    
    if p_value < alpha:
        if slope > 0:
            return 'sig_increase'
        else:
            return 'sig_decrease'
    else:
        if slope > 0:
            return 'increase'
        elif slope < 0:
            return 'decrease'
        else:
            return 'no_change'


def save_summary_statistics(df, output_path, index_name):
    """
    Compute and save summary statistics for edge metrics.
    
    Parameters
    ----------
    df : DataFrame
        WDPA-level dataset with edge metrics
    output_path : str or Path
        Output file path for statistics text file
    index_name : str
        Index name for labeling
    
    Returns
    -------
    None
        Writes statistics to text file
    """
    output_path = Path(output_path)
    
    with open(output_path, 'w') as f:
        f.write(f"Summary Statistics for {index_name.upper()}\n")
        f.write("="*80 + "\n\n")
        
        f.write("Edge Extent:\n")
        f.write(f"  Mean: {df['edge_extent'].mean():.4f}\n")
        f.write(f"  Median: {df['edge_extent'].median():.4f}\n")
        f.write(f"  Std: {df['edge_extent'].std():.4f}\n")
        f.write(f"  Min: {df['edge_extent'].min():.4f}\n")
        f.write(f"  Max: {df['edge_extent'].max():.4f}\n\n")
        
        f.write("Edge Intensity:\n")
        intensity_clean = df[np.isfinite(df['edge_intensity'])]
        f.write(f"  Mean: {intensity_clean['edge_intensity'].mean():.4f}\n")
        f.write(f"  Median: {intensity_clean['edge_intensity'].median():.4f}\n")
        f.write(f"  Std: {intensity_clean['edge_intensity'].std():.4f}\n")
        f.write(f"  Min: {intensity_clean['edge_intensity'].min():.4f}\n")
        f.write(f"  Max: {intensity_clean['edge_intensity'].max():.4f}\n\n")
        
        f.write("Protected Areas by Biome:\n")
        biome_counts = df.groupby('BIOME_NAME')['WDPA_PID'].nunique().sort_values(ascending=False)
        for biome, count in biome_counts.items():
            f.write(f"  {biome}: {count}\n")
    
    print(f"Summary statistics saved to {output_path}")


def fit_mixed_model_placeholder(df, formula, groups, output_path):
    """
    Placeholder for fitting mixed-effects model.
    
    Note: This is a placeholder. Full implementation requires statsmodels.
    See src/7_analysis.ipynb for complete implementation.
    
    Parameters
    ----------
    df : DataFrame
        Analysis dataset
    formula : str
        R-style formula for model
    groups : str
        Grouping variable for random effects
    output_path : str or Path
        Output file for model results
    
    Returns
    -------
    None
        Writes model summary to file
    """
    print(f"Mixed model fitting - see src/7_analysis.ipynb for full implementation")
    print(f"  Formula: {formula}")
    print(f"  Groups: {groups}")
    print(f"  Output: {output_path}")
    
    # Placeholder - full implementation in orchestration script
    with open(output_path, 'w') as f:
        f.write("Mixed-effects model placeholder\n")
        f.write(f"Formula: {formula}\n")
        f.write(f"Groups: {groups}\n")
        f.write("\nSee src/7_analysis.ipynb for full implementation\n")


def run_anova_placeholder(df, response, factors, output_path):
    """
    Placeholder for ANOVA tests.
    
    Note: This is a placeholder. Full implementation requires statsmodels.
    See src/7_analysis.ipynb for complete implementation.
    
    Parameters
    ----------
    df : DataFrame
        Analysis dataset
    response : str
        Response variable
    factors : list
        Categorical factors to test
    output_path : str or Path
        Output file for ANOVA results
    
    Returns
    -------
    None
        Writes ANOVA results to file
    """
    print(f"ANOVA testing - see src/7_analysis.ipynb for full implementation")
    print(f"  Response: {response}")
    print(f"  Factors: {factors}")
    print(f"  Output: {output_path}")
    
    # Placeholder
    with open(output_path, 'w') as f:
        f.write("ANOVA placeholder\n")
        f.write(f"Response: {response}\n")
        f.write(f"Factors: {', '.join(factors)}\n")
        f.write("\nSee src/7_analysis.ipynb for full implementation\n")


def create_correlation_plot(df, output_path, index_name):
    """
    Create correlation plot between edge_extent and edge_intensity.
    
    Parameters
    ----------
    df : DataFrame
        WDPA-level dataset
    output_path : str or Path
        Output file path for figure
    index_name : str
        Index name for title
    
    Returns
    -------
    None
        Saves figure to file
    """
    # Filter finite values
    clean_df = df[np.isfinite(df['edge_extent']) & np.isfinite(df['edge_intensity'])]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(clean_df['edge_extent'], clean_df['edge_intensity'], 
              alpha=0.3, s=10, edgecolors='none')
    ax.set_xlabel('Edge Extent')
    ax.set_ylabel('Edge Intensity (Cohen\'s d)')
    ax.set_title(f'{index_name.upper()}: Edge Extent vs Intensity')
    ax.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = clean_df[['edge_extent', 'edge_intensity']].corr().iloc[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation plot saved to {output_path}")


def create_distribution_plots(df, output_path, index_name):
    """
    Create distribution plots for edge_extent and edge_intensity.
    
    Parameters
    ----------
    df : DataFrame
        WDPA-level dataset
    output_path : str or Path
        Output file path for figure
    index_name : str
        Index name for title
    
    Returns
    -------
    None
        Saves figure to file
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Edge extent
    axes[0].hist(df['edge_extent'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Edge Extent')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{index_name.upper()}: Edge Extent Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Edge intensity  
    intensity_clean = df[np.isfinite(df['edge_intensity'])]
    axes[1].hist(intensity_clean['edge_intensity'], bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Edge Intensity (Cohen\'s d)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{index_name.upper()}: Edge Intensity Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Distribution plots saved to {output_path}")


# Note: Additional plotting functions for trend maps, stacked bar charts,
# covariate relationships, etc. are implemented in src/7_analysis.ipynb
# These can be migrated here as needed for the final workflow.
