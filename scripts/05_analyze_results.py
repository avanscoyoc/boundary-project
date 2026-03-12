"""
Script 05: Analyze Results

Performs statistical analysis and creates visualizations for edge effects in
protected areas. Mirrors the analysis in src/7_analysis.ipynb.

Workflow:
1. Load WDPA-level dataset
2. Recategorize biomes and classify temporal trends
3. Compute summary statistics
4. Run ANOVA for categorical predictors
5. Fit mixed-effects models for edge extent and edge intensity
6. Create publication figures (S1, S2, 3, 4a, 4b)

BEFORE RUNNING:
Ensure processed dataset exists:
  results/wdpa_df_{INDEX_NAME}.parquet
  (produced by script 04_compute_edge_metrics.py)

OUTPUT:
  results/figures/{INDEX_NAME}_summary_statistics.txt
  results/figures/{INDEX_NAME}_anova_results.txt
  results/figures/{INDEX_NAME}_edge_extent_model.txt
  results/figures/{INDEX_NAME}_edge_intensity_model.txt
  results/figures/{INDEX_NAME}_figureS1_distributions.png
  results/figures/{INDEX_NAME}_figureS2_correlation.png
  results/figures/{INDEX_NAME}_figure3_trends_by_biome.png
  results/figures/{INDEX_NAME}_figure4a_edge_extent_model.png
  results/figures/{INDEX_NAME}_figure4b_edge_intensity_model.png
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
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

print(f"\nCurrent index: {config.INDEX_NAME.upper()}")
print(f"Description: {config.INDEX_CONFIGS[config.INDEX_NAME]['description']}")

# Paths
# Input:  results/wdpa_df_{INDEX_NAME}.parquet  (from script 04)
# Output: results/figures/{INDEX_NAME}_*.txt / *.png
wdpa_path = config.RESULTS_DIR / f'wdpa_df_{config.INDEX_NAME}.parquet'
figures_dir = config.RESULTS_FIGURES
figures_dir.mkdir(parents=True, exist_ok=True)

if not wdpa_path.exists():
    print(f"\nERROR: WDPA dataset not found: {wdpa_path}")
    print("Please run script 04_compute_edge_metrics.py first")
    exit(1)

# Load data
print("\n" + "="*80)
print("Loading Dataset")
print("="*80)

print(f"Loading: {wdpa_path}")
wdpa_df = pd.read_parquet(wdpa_path)
print(f"  Rows: {len(wdpa_df):,}")
print(f"  Protected areas: {wdpa_df['WDPA_PID'].nunique():,}")

# Recategorize BIOME_NAME into 7 major groups
from modules.analysis import recategorize_biome
print(f"Biome Names: {wdpa_df['BIOME_NAME'].unique()}")
wdpa_df['BIOME_NAME'] = wdpa_df['BIOME_NAME'].apply(recategorize_biome)
print(f"Biome Names: {wdpa_df['BIOME_NAME'].unique()}")

# Classify temporal trend per WDPA_PID using linear regression over years on edge_extent
print("\nClassifying temporal trends per protected area...")
trend_df = wdpa_df.groupby('WDPA_PID').apply(classify_trend).reset_index()
trend_df.columns = ['WDPA_PID', 'trend']
wdpa_df = wdpa_df.merge(trend_df, on='WDPA_PID', how='left')
print(f"Trend distribution:\n{wdpa_df.groupby('WDPA_PID')['trend'].first().value_counts().to_string()}")

# Step 1: Summary Statistics
print("\n" + "="*80)
print("STEP 1: Summary Statistics")
print("="*80)

summary_output = figures_dir / f'{config.INDEX_NAME}_summary_statistics.txt'

output_lines = []
output_lines.append("=" * 60)
output_lines.append("SUMMARY STATISTICS")
output_lines.append("=" * 60)

n_wdpa = wdpa_df['WDPA_PID'].nunique()
output_lines.append(f"Total number of unique WDPA_PID: {n_wdpa:,}")

total_transects = wdpa_df.groupby('WDPA_PID')['n_trnst'].first().sum()
output_lines.append(f"Total number of transects: {total_transects:,}")

n_iso3 = wdpa_df['ISO3'].nunique()
output_lines.append(f"Total number of unique ISO3: {n_iso3:,}")

n_biome = wdpa_df['BIOME_NAME'].nunique()
output_lines.append(f"Total number of unique biomes: {n_biome:,}")

output_lines.append("\nNumber of WDPA_PID per biome:")
biome_counts = wdpa_df.groupby('BIOME_NAME')['WDPA_PID'].nunique().sort_values(ascending=False)
for biome, count in biome_counts.items():
    output_lines.append(f"  {biome}: {count:,}")

output_lines.append("\nTrend distribution:")
trend_counts = wdpa_df.groupby('WDPA_PID')['trend'].first().value_counts()
for trend, count in trend_counts.items():
    output_lines.append(f"  {trend}: {count:,}")

# Find what percent of PAs have low to no edges
wdpa_summary = wdpa_df.groupby('WDPA_PID').agg(
    edge_extent=('edge_extent', 'mean'),
    edge_intensity=('edge_intensity', 'mean')
).reset_index()
low_extent = (wdpa_summary['edge_extent'] < 0.1).sum()
output_lines.append(f"\nWDPA_PID with edge_extent < 10%: {low_extent:,} ({low_extent/len(wdpa_summary)*100:.1f}%)")
low_intensity = (wdpa_summary['edge_intensity'] < 0).sum()
output_lines.append(f"WDPA_PID with edge_intensity < 0: {low_intensity:,} ({low_intensity/len(wdpa_summary)*100:.1f}%)")
output_lines.append("=" * 60)

print('\n'.join(output_lines))
with open(summary_output, 'w') as f:
    f.write('\n'.join(output_lines))
print(f"Saved: {summary_output}")

# Step 2: ANOVA for categorical predictors
print("\n" + "="*80)
print("STEP 2: ANOVA")
print("="*80)

anova_df = wdpa_df[['edge_extent', 'IUCN_CAT', 'STATUS_YR', 'BIOME_NAME', 'AREA_DISSO']].dropna()
anova_output = figures_dir / f'{config.INDEX_NAME}_anova_results.txt'

with open(anova_output, 'w') as f:
    for factor, formula in [
        ('IUCN_CAT',  'edge_extent ~ C(IUCN_CAT)'),
        ('STATUS_YR', 'edge_extent ~ STATUS_YR'),
        ('BIOME_NAME','edge_extent ~ C(BIOME_NAME)'),
        ('AREA_DISSO','edge_extent ~ AREA_DISSO'),
    ]:
        model = ols(formula, data=anova_df).fit()
        result = anova_lm(model, typ=2)
        print(f"\nANOVA for {factor}:\n{result}")
        f.write(f"ANOVA for {factor}:\n{result}\n\n")

print(f"Saved: {anova_output}")

# Step 3: Mixed-effects models
print("\n" + "="*80)
print("STEP 3: Mixed-Effects Models")
print("="*80)

predictor_cols = ['AREA_DISSO', 'gHM_mean', 'elevation_mean', 'slope_mean', 'water_extent_pct']

for response, label, fig_label, out_stem in [
    ('edge_intensity', 'EDGE INTENSITY', 'Figure 4b', f'{config.INDEX_NAME}_edge_intensity_model'),
    ('edge_extent',    'EDGE EXTENT',    'Figure 4a', f'{config.INDEX_NAME}_edge_extent_model'),
]:
    print(f"\nFitting mixed model: {response}...")
    model_df = wdpa_df[[response] + predictor_cols + ['BIOME_NAME', 'year', 'WDPA_PID']].dropna()

    for col in predictor_cols:
        model_df[f'{col}_z'] = (model_df[col] - model_df[col].mean()) / model_df[col].std()

    formula = f'{response} ~ ' + ' + '.join(f'{c}_z' for c in predictor_cols)
    md = MixedLM.from_formula(formula, data=model_df, groups=model_df['BIOME_NAME'], re_formula='1')
    mdf = md.fit()
    print(mdf.summary())

    with open(figures_dir / f'{out_stem}.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"MIXED MODEL: {label}\n")
        f.write("=" * 60 + "\n\n")
        f.write(str(mdf.summary()))
    print(f"Saved: {figures_dir / f'{out_stem}.txt'}")

    coef_df = pd.DataFrame({
        'Variable': mdf.params.index[1:],
        'Coefficient': mdf.params.values[1:],
        'CI_lower': mdf.conf_int().iloc[1:, 0],
        'CI_upper': mdf.conf_int().iloc[1:, 1]
    })
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(coef_df['Coefficient'], range(len(coef_df)),
                xerr=[coef_df['Coefficient'] - coef_df['CI_lower'],
                      coef_df['CI_upper'] - coef_df['Coefficient']],
                fmt='o', capsize=5)
    ax.axvline(0, color='red', linestyle='--', linewidth=1)
    ax.set_yticks(range(len(coef_df)))
    ax.set_yticklabels(coef_df['Variable'])
    ax.set_xlabel('Coefficient (95% CI)')
    ax.set_title(f'{fig_label}: {response.replace("_", " ").title()} ~ Covariates')
    plt.tight_layout()
    fig_out = figures_dir / f'{out_stem}.png'
    plt.savefig(fig_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_out}")

# Step 4: Figures
print("\n" + "="*80)
print("STEP 4: Figures")
print("="*80)

# Figure S1: Distributions
print("\nFigure S1: Distributions...")
create_distribution_plots(wdpa_df, figures_dir / f'{config.INDEX_NAME}_figureS1_distributions.png', config.INDEX_NAME)

# Figure S2: Correlation
print("Figure S2: Correlation...")
create_correlation_plot(wdpa_df, figures_dir / f'{config.INDEX_NAME}_figureS2_correlation.png', config.INDEX_NAME)

# Figure 3: Trend stacked bar by biome
print("Figure 3: Trends by biome...")
trend_by_biome = wdpa_df.groupby(['BIOME_NAME', 'WDPA_PID'])['trend'].first().reset_index()
trend_counts_fig = trend_by_biome.groupby(['BIOME_NAME', 'trend']).size().unstack(fill_value=0)
col_order = ['sig_decrease', 'decrease', 'no_change', 'increase', 'sig_increase']
trend_counts_fig = trend_counts_fig[[c for c in col_order if c in trend_counts_fig.columns]]
trend_pcts = trend_counts_fig.div(trend_counts_fig.sum(axis=1), axis=0) * 100
biome_totals = trend_counts_fig.sum(axis=1)

fig, ax = plt.subplots(figsize=(10, 8))
trend_pcts.plot(kind='barh', stacked=True, ax=ax,
                color=['#d73027', '#fc8d59', '#afab9e', '#91bfdb', '#4575b4'])
ax.set_yticklabels([f"{b} (n={biome_totals[b]})" for b in trend_pcts.index])
for i, biome in enumerate(trend_pcts.index):
    cumulative = 0
    for col in trend_pcts.columns:
        val = trend_pcts.loc[biome, col]
        if val > 5:
            ax.text(cumulative + val/2, i, f'{val:.1f}%', ha='center', va='center', fontsize=8)
        cumulative += val
ax.set_xlabel('Percentage of Protected Areas')
ax.set_ylabel('Biome')
ax.legend(title='Trend', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_title('Edge Extent Trends by Biome')
plt.tight_layout()
fig3_out = figures_dir / f'{config.INDEX_NAME}_figure3_trends_by_biome.png'
plt.savefig(fig3_out, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {fig3_out}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"Results saved to: {figures_dir}")
print("="*80)
