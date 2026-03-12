"""
Script 04: Compute Edge Metrics

Processes raw satellite imagery data to compute edge detection metrics for each
protected area transect.

Workflow:
1. Sort and combine raw CSV chunks by WDPA_PID
2. Create transect-level dataset with Cohen's D edge effect measurements
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

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
attributes_path = config.DATA_PROCESSED / "attributes_final.csv"
transect_output = output_dir / f'transect_df_{config.INDEX_NAME}.parquet' 
wdpa_output = output_dir / f'wdpa_df_{config.INDEX_NAME}2.parquet'

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
sorted_files = sort_and_combine_csvs(raw_dir, config.INDEX_NAME, config.START_YEAR, config.END_YEAR)
print(f"Sorted data written to {len(sorted_files)} parquet files")

# Step 2: Create transect-level dataset
print("\n" + "="*80)
print("STEP 2: Computing Transect-Level Metrics")
print("="*80)

print("Pivoting to transect level and computing edge detection...")
transect_files = create_transect_dataset(raw_dir, config.INDEX_NAME, config.START_YEAR, config.END_YEAR)
print(f"Transect chunks created: {len(transect_files)}")

# Step 3: Create WDPA-level dataset
print("\n" + "="*80)
print("STEP 3: Aggregating to Protected Area Level")
print("="*80)

transect_dir = raw_dir / "transect_chunks"
print("Loading attributes and merging with transect metrics...")
create_wdpa_dataset(transect_dir, attributes_path, config.INDEX_NAME, wdpa_output)

wdpa_size = wdpa_output.stat().st_size / (1024**2)
print(f"WDPA file: {wdpa_size:.1f} MB")

print("\n" + "="*80)
print("PROCESSING COMPLETE")
print("="*80)
print(f"Transect chunks: {transect_dir}/")
print(f"WDPA dataset: {wdpa_output}")
print("\nNext step: python scripts/05_analyze_results.py")
print("="*80)
