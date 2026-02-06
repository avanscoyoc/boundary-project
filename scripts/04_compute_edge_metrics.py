"""
Script 04: Compute Edge Metrics

Processes raw satellite imagery data to compute edge detection metrics for each
protected area transect.

Workflow:
1. Sort and combine raw CSV chunks by WDPA_PID
2. Create transect-level dataset with Cohen's d edge effect measurements
3. Create protected area-level dataset with aggregated metrics
4. Save results as parquet files for efficient analysis

BEFORE RUNNING:
1. Download CSV files from Google Drive (see script 03 output)
2. Ensure they are in: results/raw/{INDEX_NAME}_raw/

OUTPUT:
- results/transect_df_{INDEX_NAME}.parquet
- results/wdpa_df_{INDEX_NAME}.parquet

Then run: python scripts/05_analyze_results.py
"""

import os
import pandas as pd
from modules import config
from modules.analysis import (
    sort_and_combine_csvs,
    create_transect_dataset,
    create_wdpa_dataset
)

print("="*80)
print("SCRIPT 04: Compute Edge Metrics")
print("="*80)

# Validate configuration
print(f"\nCurrent index: {config.INDEX_NAME.upper()}")
print(f"Description: {config.INDEX_CONFIGS[config.INDEX_NAME]['description']}")

# Paths
raw_dir = config.RESULTS_RAW / f"{config.INDEX_NAME}_raw"
output_dir = config.RESULTS_DIR
attributes_path = config.PROCESSED_ATTRIBUTES
transect_output = output_dir / f'transect_df_{config.INDEX_NAME}.parquet'
wdpa_output = output_dir / f'wdpa_df_{config.INDEX_NAME}.parquet'

# Verify input files exist
csv_files = list(raw_dir.glob('*.csv'))
if not csv_files:
    print(f"\nERROR: No CSV files found in {raw_dir}")
    print("Please download files from Google Drive first (see script 03 output)")
    exit(1)

print(f"\nFound {len(csv_files)} CSV files in {raw_dir}")

# Check for attributes file
if not attributes_path.exists():
    print(f"\nERROR: Attributes file not found: {attributes_path}")
    print("This file should contain WDPA metadata (biome, IUCN category, etc.)")
    exit(1)

print(f"Attributes file: {attributes_path}")

# Step 1: Sort and combine CSV chunks
print("\n" + "="*80)
print("STEP 1: Sorting and Combining CSV Files")
print("="*80)

print(f"Reading {len(csv_files)} CSV files...")
sorted_df = sort_and_combine_csvs(raw_dir)

print(f"Total samples: {len(sorted_df):,}")
print(f"Unique PAs: {sorted_df['WDPA_PID'].nunique():,}")
print(f"Unique transects: {sorted_df['transectID'].nunique():,}")

# Step 2: Create transect-level dataset
print("\n" + "="*80)
print("STEP 2: Computing Transect-Level Metrics")
print("="*80)

print("Computing Cohen's d for edge effects...")
transect_df = create_transect_dataset(sorted_df)

print(f"Transects processed: {len(transect_df):,}")
print(f"Protected areas: {transect_df['WDPA_PID'].nunique():,}")

# Step 3: Create WDPA-level dataset
print("\n" + "="*80)
print("STEP 3: Aggregating to Protected Area Level")
print("="*80)

print("Loading attributes and merging with transect metrics...")
wdpa_df = create_wdpa_dataset(transect_df, attributes_path)

print(f"Protected areas: {len(wdpa_df):,}")
print(f"Mean transects per PA: {wdpa_df['num_transects'].mean():.1f}")

# Show column names
print("\nTransect dataset columns:")
print(f"  {', '.join(transect_df.columns[:10])}...")

print("\nWDPA dataset columns:")
print(f"  {', '.join(wdpa_df.columns[:15])}...")

# Step 4: Save results
print("\n" + "="*80)
print("STEP 4: Saving Results")
print("="*80)

print(f"Saving transect dataset to: {transect_output}")
transect_df.to_parquet(transect_output, index=False)

print(f"Saving WDPA dataset to: {wdpa_output}")
wdpa_df.to_parquet(wdpa_output, index=False)

# Verify saves
transect_size = transect_output.stat().st_size / (1024**2)  # MB
wdpa_size = wdpa_output.stat().st_size / (1024**2)  # MB

print(f"\nTransect file: {transect_size:.1f} MB")
print(f"WDPA file: {wdpa_size:.1f} MB")

print("\n" + "="*80)
print("PROCESSING COMPLETE")
print("="*80)
print(f"Transect dataset: {transect_output}")
print(f"WDPA dataset: {wdpa_output}")
print("\nNext step: python scripts/05_analyze_results.py")
print("="*80)
