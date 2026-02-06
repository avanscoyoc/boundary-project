import os
import pandas as pd
import numpy as np
from pathlib import Path

# ===== CONFIGURATION: SELECT INDEX =====
INDEX_NAME = 'ndvi'  # Must match the index processed in 4_gee_tasks.ipynb
START_YEAR = 2001
END_YEAR = 2022  # Exclusive
# =======================================

print(f"Processing {INDEX_NAME.upper()} results...")

# Index-specific paths
folder_path = Path(f"results/{INDEX_NAME}_raw") 

# Verify input folder exists
if not folder_path.exists():
    raise FileNotFoundError(
        f"Results folder not found: {folder_path}\n"
        f"Make sure you've downloaded results from Google Drive to this location."
    )

id_vars = ['WDPA_PID', 'transectID', 'pointID', 'max_extent', 'gHM', 'elevation', 'slope']
value_vars = [str(y) for y in range(START_YEAR, END_YEAR)]
usecols = id_vars + value_vars

dtypes = {
    'WDPA_PID': 'string',
    'transectID': 'string',
    'pointID': 'int8',
    'max_extent': 'float32',  # float32 to allow NaNs
    'gHM': 'float32',
    'elevation': 'float32',
    'slope': 'float32',
}

# 1. Read and append all CSVs
print(f"Reading CSV files from {folder_path}...")
all_files = sorted([folder_path / f for f in os.listdir(folder_path) if f.endswith(".csv")])
print(f"Found {len(all_files)} CSV files")

df_list = [pd.read_csv(file, usecols=usecols, dtype=dtypes, low_memory=False) for file in all_files]
df = pd.concat(df_list, ignore_index=True)
del df_list

# Replace modis values with NaN
df[value_vars] = df[value_vars].replace(-9999.0, np.nan)

print(f"Loaded {len(df):,} rows")

# 2. Sort the full DataFrame
print("Sorting by WDPA_PID, transectID, pointID...")
df = df.sort_values(['WDPA_PID', 'transectID', 'pointID'])

# 3. Split into 10 files, keeping WDPA_PID together
print("Splitting into 10 parquet files...")
unique_pids = df['WDPA_PID'].unique()
n_splits = 10
split_sizes = [len(unique_pids) // n_splits + (1 if x < len(unique_pids) % n_splits else 0) for x in range(n_splits)]

start = 0
for i, size in enumerate(split_sizes):
    pids = unique_pids[start:start+size]
    split_df = df[df['WDPA_PID'].isin(pids)]
    out_file = folder_path / f"{INDEX_NAME}_sorted_chunk_{i}.parquet"
    split_df.to_parquet(out_file, engine="pyarrow", index=False)
    print(f"  Saved split {i}: {len(split_df):,} rows")
    start += size

print(f"\nDone! Sorted results saved to {folder_path}/")