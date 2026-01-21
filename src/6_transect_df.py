import pandas as pd
import numpy as np
import glob
import os

#########################################################################
# Create transectID level dataset (pandas chunked approach)
#########################################################################

print("Pivoting data in pandas chunks...")
files = sorted(glob.glob("../data/results_sorted_chunks/*.parquet"))
transect_chunks_dir = "../data/transect_chunks"

# Create output directory for transect chunks
os.makedirs(transect_chunks_dir, exist_ok=True)
# Clean old chunks if they exist
for old_file in glob.glob(os.path.join(transect_chunks_dir, "*.parquet")):
    os.remove(old_file)

# Process each chunk and save to transect_chunks directory
for idx, f in enumerate(files):
    print(f"Processing chunk {idx+1}/{len(files)}: {os.path.basename(f)}")
    df = pd.read_parquet(f)
    
    # Pivot long: year/value
    long = (
        df
        .melt(id_vars=['WDPA_PID', 'transectID', 'pointID', 'max_extent', 'gHM', 'elevation', 'slope'],
              value_vars=[str(y) for y in range(2001, 2022)],
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
    
    # Compute covariates efficiently using vectorized operations
    # Filter to transects that survived the dropna
    long_valid = long.set_index(['WDPA_PID', 'year', 'transectID'])
    long_valid = long_valid[long_valid.index.isin(pts.index)].reset_index()
    
    # Max extent - just take max per transect
    max_extent_df = long_valid.groupby(['WDPA_PID','year','transectID'])['max_extent'].max()
    
    # gHM mean for outer points (pointID 1 and 2)
    gHM_outer = long_valid[long_valid['pointID'].isin([1,2])].groupby(['WDPA_PID','year','transectID'])['gHM'].mean()
    
    # Elevation and slope at point 0
    pt0_data = long_valid[long_valid['pointID']==0].set_index(['WDPA_PID','year','transectID'])[['elevation','slope']]
    
    # Combine covariates
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
    
    # Save chunk (this is the transect_df in chunks - user needs this)
    output_file = os.path.join(transect_chunks_dir, f"transect_chunk_{idx:02d}.parquet")
    chunk_df.to_parquet(output_file, engine='pyarrow', index=True)
    
    del df, long, long_valid, pts, covars, max_extent_df, gHM_outer, pt0_data, chunk_df
    print(f"  Chunk {idx+1} processed and saved")

print(f"Transect dataset saved in chunks to {transect_chunks_dir}/")

#########################################################################
# Create WDPA level dataset
#########################################################################

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

print("\nComputing WDPA-level statistics from transect chunks...")
wdpa_list = []

# Read each transect chunk and compute WDPA stats
transect_files = sorted(glob.glob(os.path.join(transect_chunks_dir, "*.parquet")))
for chunk_num, transect_file in enumerate(transect_files):
    print(f"  Processing WDPA stats from chunk {chunk_num+1}/{len(transect_files)}...")
    transect_chunk = pd.read_parquet(transect_file)
    
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
attributes = pd.read_csv("../data/attributes_final.csv")
wdpa_df = pd.merge(wdpa_df, attributes, on='WDPA_PID', how='left')

print("Saving wdpa_df.parquet...")
wdpa_df.to_parquet("../data/wdpa_df.parquet", engine='pyarrow', index=False)
print("Done!")