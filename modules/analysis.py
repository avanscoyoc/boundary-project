"""
Data processing and edge detection analysis functions.

This module contains functions for:
- Sorting and concatenating GEE results
- Pivoting data from long to wide format  
- Computing edge detection metrics
- Aggregating to WDPA-year level with Cohen's d effect sizes
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path


def cohens_d(x, y):
    """
    Compute Cohen's d effect size between two samples.
    
    Parameters
    ----------
    x : array-like
        First sample (e.g., values at boundary point)
    y : array-like
        Second sample (e.g., values at comparison point)
    
    Returns
    -------
    float
        Cohen's d effect size (standardized mean difference)
    
    Notes
    -----
    Uses pooled standard deviation with (n-1) degrees of freedom.
    """
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std


def sort_and_combine_csvs(input_dir, index_name, start_year, end_year, output_chunks=10):
    """
    Read, concatenate, sort, and split GEE CSV results into parquet chunks.
    
    Processes all CSV files from GEE exports, replaces sentinel values with NaN,
    sorts by WDPA_PID/transectID/pointID, and splits into chunks while keeping
    each WDPA_PID together.
    
    Parameters
    ----------
    input_dir : str or Path
        Directory containing raw CSV files from GEE
    index_name : str
        Index name (ndvi, lai, fpar, ndbi) for column selection
    start_year : int
        First year in the time series
    end_year : int
        Last year in the time series (inclusive)
    output_chunks : int, optional
        Number of output parquet files. Default 10.
    
    Returns
    -------
    list
        Paths to created parquet chunk files
    
    Notes
    -----
    - Replaces -9999 with NaN for gradient values
    - Uses memory-efficient dtypes
    - Deletes intermediate sorted chunks after processing
    """
    input_dir = Path(input_dir)
    
    print(f"Reading CSV files from {input_dir}...")
    
    # Define columns
    id_vars = ['WDPA_PID', 'transectID', 'pointID', 'max_extent', 'gHM', 'elevation', 'slope']
    value_vars = [str(y) for y in range(start_year, end_year + 1)]
    usecols = id_vars + value_vars
    
    dtypes = {
        'WDPA_PID': 'string',
        'transectID': 'string',
        'pointID': 'int8',
        'max_extent': 'float32',
        'gHM': 'float32',
        'elevation': 'float32',
        'slope': 'float32',
    }
    
    # Read all CSVs
    all_files = sorted([input_dir / f for f in os.listdir(input_dir) if f.endswith(".csv")])
    print(f"Found {len(all_files)} CSV files")
    
    df_list = [pd.read_parquet(file, usecols=usecols, dtype=dtypes, low_memory=False) for file in all_files]
    df = pd.concat(df_list, ignore_index=True)
    del df_list
    
    # Replace sentinel values with NaN
    df[value_vars] = df[value_vars].replace(-9999.0, np.nan)
    print(f"Loaded {len(df):,} rows")
    
    # Sort
    print("Sorting by WDPA_PID, transectID, pointID...")
    df = df.sort_values(['WDPA_PID', 'transectID', 'pointID'])
    
    # Split into chunks keeping WDPA_PIDs together
    print(f"Splitting into {output_chunks} parquet files...")
    unique_pids = df['WDPA_PID'].unique()
    n_splits = output_chunks
    split_sizes = [len(unique_pids) // n_splits + (1 if x < len(unique_pids) % n_splits else 0) 
                   for x in range(n_splits)]
    
    output_files = []
    start = 0
    for i, size in enumerate(split_sizes):
        pids = unique_pids[start:start+size]
        split_df = df[df['WDPA_PID'].isin(pids)]
        out_file = input_dir / f"{index_name}_sorted_chunk_{i}.parquet"
        split_df.to_parquet(out_file, engine="pyarrow", index=False)
        output_files.append(str(out_file))
        print(f"  Saved split {i}: {len(split_df):,} rows")
        start += size
    
    return output_files


def create_transect_dataset(input_dir, index_name, start_year, end_year, output_dir=None):
    """
    Transform sorted point data to transect-level dataset with edge detection.
    
    Reads sorted parquet chunks, pivots from long (years) to wide (pointID columns),
    computes edge detection metrics, and adds environmental covariates.
    
    Parameters
    ----------
    input_dir : str or Path
        Directory containing sorted parquet chunks
    index_name : str
        Index name for file naming
    start_year : int
        First year in time series
    end_year : int
        Last year in time series (inclusive)
    output_dir : str or Path, optional
        Output directory for transect chunks. If None, creates subdirectory.
    
    Returns
    -------
    list
        Paths to created transect chunk files
    
    Notes
    -----
    Edge detection logic: edge present when pt_0 > min(pt_2, pt_1) AND 
    pt_0 > min(pt_m2, pt_m1), indicating higher values at boundary than both sides.
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir / "transect_chunks"
    else:
        output_dir = Path(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean old chunks
    for old_file in glob.glob(str(output_dir / f"{index_name}_transect_chunk_*.parquet")):
        os.remove(old_file)
    
    print("Pivoting data in pandas chunks...")
    files = sorted(glob.glob(str(input_dir / f"{index_name}_sorted_chunk_*.parquet")))
    print(f"Found {len(files)} input files")
    
    output_files = []
    
    for idx, f in enumerate(files):
        print(f"Processing chunk {idx+1}/{len(files)}: {os.path.basename(f)}")
        df = pd.read_parquet(f)
        
        # Pivot long: year/value
        long = (
            df
            .melt(id_vars=['WDPA_PID', 'transectID', 'pointID', 'max_extent', 'gHM', 
                          'elevation', 'slope'],
                  value_vars=[str(y) for y in range(start_year, end_year + 1)],
                  var_name='year', value_name='value')
            .dropna(subset=['value'])
        )
        long['year'] = long['year'].astype('int16')
        long['value'] = long['value'].astype('float32')
        
        # Pivot wide: pointID to columns
        pts = long.pivot_table(
            index=['WDPA_PID', 'year', 'transectID'],
            columns='pointID', values='value', aggfunc='first'
        )
        pts = pts.rename(columns={2: 'pt_2', 1: 'pt_1', 0: 'pt_0', -1: 'pt_m1', -2: 'pt_m2'})
        pts = pts.dropna(subset=['pt_2', 'pt_1', 'pt_0', 'pt_m1', 'pt_m2'])
        
        # Compute covariates
        long_valid = long.set_index(['WDPA_PID', 'year', 'transectID'])
        long_valid = long_valid[long_valid.index.isin(pts.index)].reset_index()
        
        max_extent_df = long_valid.groupby(['WDPA_PID','year','transectID'])['max_extent'].max()
        gHM_outer = long_valid[long_valid['pointID'].isin([1,2])].groupby(
            ['WDPA_PID','year','transectID'])['gHM'].mean()
        pt0_data = long_valid[long_valid['pointID']==0].set_index(
            ['WDPA_PID','year','transectID'])[['elevation','slope']]
        
        covars = pd.DataFrame({
            'trnst_max_extent': max_extent_df,
            'gHM_mean_outer': gHM_outer,
            'elevation_pt0': pt0_data['elevation'],
            'slope_pt0': pt0_data['slope']
        })
        
        # Join and compute edge
        chunk_df = pts.join(covars)
        chunk_df['edge'] = (
            (chunk_df['pt_0'] > chunk_df[['pt_2','pt_1']].min(axis=1)) &
            (chunk_df['pt_0'] > chunk_df[['pt_m2','pt_m1']].min(axis=1))
        ).astype('int8')
        
        # Save chunk
        output_file = output_dir / f"{index_name}_transect_chunk_{idx}.parquet"
        chunk_df.to_parquet(output_file, engine='pyarrow', index=True)
        output_files.append(str(output_file))
        
        del df, long, long_valid, pts, covars, max_extent_df, gHM_outer, pt0_data, chunk_df
        print(f"  Chunk {idx+1} processed and saved")
    
    # Clean up intermediate sorted files
    print("Deleting intermediate sorted chunk files...")
    for sorted_file in files:
        os.remove(sorted_file)
        print(f"  Deleted {os.path.basename(sorted_file)}")
    
    return output_files


def create_wdpa_dataset(transect_dir, attributes_path, index_name, output_path):
    """
    Aggregate transect-level data to WDPA-year level with edge metrics.
    
    Computes Cohen's d effect sizes for all point comparisons, derives edge
    intensity (minimum d) and edge extent (proportion of transects with edges),
    and merges with protected area attributes.
    
    Parameters
    ----------
    transect_dir : str or Path
        Directory containing transect chunk parquet files
    attributes_path : str or Path
        Path to attributes CSV file
    index_name : str
        Index name for file selection
    output_path : str or Path
        Output path for WDPA-level parquet file
    
    Returns
    -------
    str
        Path to created WDPA-level parquet file
    
    Notes
    -----
    Edge intensity = min(D02, D01, D0m1, D0m2) where D is Cohen's d.
    Edge extent = proportion of transects where edge == 1.
    """
    transect_dir = Path(transect_dir)
    
    print("Computing WDPA-level statistics from transect chunks...")
    wdpa_list = []
    
    transect_files = sorted(glob.glob(str(transect_dir / f"{index_name}_transect_chunk_*.parquet")))
    
    for chunk_num, transect_file in enumerate(transect_files):
        print(f"  Processing WDPA stats from chunk {chunk_num+1}/{len(transect_files)}...")
        transect_chunk = pd.read_parquet(transect_file).reset_index()
        
        for (wdpa, year), group in transect_chunk.groupby(['WDPA_PID','year']):
            d02 = cohens_d(group['pt_0'], group['pt_2'])
            d01 = cohens_d(group['pt_0'], group['pt_1'])
            d0m1 = cohens_d(group['pt_0'], group['pt_m1'])
            d0m2 = cohens_d(group['pt_0'], group['pt_m2'])
            edge_intensity = min(d02, d01, d0m1, d0m2)
            edge_extent = (group['edge']==1).sum() / len(group)
            
            wdpa_list.append({
                'WDPA_PID': wdpa,
                'year': year,
                'n_trnst': len(group),
                'D02': d02,
                'D01': d01,
                'D0m1': d0m1,
                'D0m2': d0m2,
                'edge_intensity': edge_intensity,
                'edge_extent': edge_extent,
                'gHM_mean': group['gHM_mean_outer'].mean(),
                'elevation_mean': group['elevation_pt0'].mean(),
                'slope_mean': group['slope_pt0'].mean(),
                'water_extent_pct': (group['trnst_max_extent'] == 1).sum() / len(group) * 100
            })
        del transect_chunk
    
    print("Merging with attributes...")
    wdpa_df = pd.DataFrame(wdpa_list)
    attributes = pd.read_csv(attributes_path)
    wdpa_df = pd.merge(wdpa_df, attributes, on='WDPA_PID', how='left')
    
    print(f"Saving {output_path}...")
    wdpa_df.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"WDPA-level dataset saved to {output_path}")
    
    return str(output_path)


def recategorize_biome(biome):
    """
    Recategorize biomes into 7 major groups.
    
    Parameters
    ----------
    biome : str
        Original RESOLVE Ecoregions biome name
    
    Returns
    -------
    str
        Recategorized biome name
    """
    if biome == "Mangroves":
        return "Mangrove"
    elif biome == "N/A":
        return "Rock & Ice"
    elif biome in ["Deserts & Xeric Shrublands"]:
        return "Desert"
    elif biome in ["Tropical & Subtropical Coniferous Forests",
                   "Tropical & Subtropical Moist Broadleaf Forests",
                   "Tropical & Subtropical Dry Broadleaf Forests"]:
        return "Tropical-Forests"
    elif biome in ["Mediterranean Forests, Woodlands & Scrub",
                   "Temperate Conifer Forests",
                   "Temperate Broadleaf & Mixed Forests"]:
        return "Temperate-Forests"
    elif biome in ["Boreal Forests/Taiga"]:
        return "Boreal-Forests"
    elif biome in ["Tropical & Subtropical Grasslands, Savannas & Shrublands",
                   "Temperate Grasslands, Savannas & Shrublands",
                   "Montane Grasslands & Shrublands",
                   "Flooded Grasslands & Savannas"]:
        return "Grassland-Shrubland"
    else:
        return biome
