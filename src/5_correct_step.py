import os
import pandas as pd
from pathlib import Path

folder_path = Path("../results")
out_dir = Path("../data/results_sorted_chunks")
out_dir.mkdir(exist_ok=True)  # directory for chunked parquet

id_vars = ['WDPA_PID', 'transectID', 'pointID', 'max_extent', 'gHM', 'elevation', 'slope']
value_vars = [str(y) for y in range(2001, 2022)]
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
all_files = sorted(folder_path / f for f in os.listdir(folder_path) if f.endswith(".csv"))
df_list = [pd.read_csv(file, usecols=usecols, dtype=dtypes, low_memory=False) for file in all_files]
df = pd.concat(df_list, ignore_index=True)
del df_list

# 2. Sort the full DataFrame
df = df.sort_values(['WDPA_PID', 'transectID', 'pointID'])

# 3. Split into 10 files, keeping WDPA_PID together
unique_pids = df['WDPA_PID'].unique()
n_splits = 10
split_sizes = [len(unique_pids) // n_splits + (1 if x < len(unique_pids) % n_splits else 0) for x in range(n_splits)]

start = 0
for i, size in enumerate(split_sizes):
    pids = unique_pids[start:start+size]
    split_df = df[df['WDPA_PID'].isin(pids)]
    out_file = out_dir / f"results_sorted_split_{i}.parquet"
    split_df.to_parquet(out_file, engine="pyarrow", index=False)
    start += size