import os
import pandas as pd
from pathlib import Path

folder_path = Path("../results")
out_dir = Path("../data/results_long_chunks")
out_dir.mkdir(exist_ok=True)  # directory for chunked parquet

id_vars = ['WDPA_PID', 'transectID', 'pointID', 'max_extent', 'gHM', 'elevation', 'slope']
value_vars = [str(y) for y in range(2001, 2022)]
usecols = id_vars + value_vars

chunksize = 100_000
dtypes = {
    'WDPA_PID': 'string',
    'transectID': 'string',
    'pointID': 'int8',
    'max_extent': 'float32',  # float32 to allow NaNs
    'gHM': 'float32',
    'elevation': 'float32',
    'slope': 'float32',
}

all_files = sorted(folder_path / f for f in os.listdir(folder_path) if f.endswith(".csv"))

chunk_counter = 0
for file in all_files:
    for chunk in pd.read_csv(file, chunksize=chunksize, usecols=usecols, dtype=dtypes, low_memory=False):
        long = (
            chunk
            .melt(id_vars=id_vars, value_vars=value_vars, var_name='year', value_name='value')
            .dropna(subset=['value'])
        )
        long['year'] = long['year'].astype('int16')
        long['value'] = long['value'].astype('float32')

        # write each chunk as a separate parquet file
        chunk_file = out_dir / f"results_long_chunk_{chunk_counter}.parquet"
        long.to_parquet(chunk_file, engine="pyarrow", index=False)
        chunk_counter += 1

        del chunk, long