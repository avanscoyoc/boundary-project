import pandas as pd
import numpy as np
import glob
  

#########################################################################
# Create transectID level dataset (pandas chunked approach)
#########################################################################

print("Pivoting data in pandas chunks...")
files = glob.glob("../data/results_sorted_chunks/*.parquet")
result_list = []
covar_list = []
for f in files:
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
    result_list.append(pts)
    covars = long.groupby(['WDPA_PID','year','transectID']).agg(
        trnst_max_extent=('max_extent','max'),
        gHM_mean_outer=('gHM', lambda x: x[long.loc[x.index,'pointID'].isin([1,2])].mean()),
        elevation_pt0=('elevation', lambda x: x[long.loc[x.index,'pointID']==0].iloc[0]),
        slope_pt0=('slope', lambda x: x[long.loc[x.index,'pointID']==0].iloc[0])
    )
    covar_list.append(covars)
final_pts = pd.concat(result_list)
final_covars = pd.concat(covar_list)
print("Concatenated pivoted data shape:", final_pts.shape)

transect_df = final_pts.join(final_covars)

print("Computing edge...")
transect_df['edge'] = (
    (transect_df['pt_0'] > transect_df[['pt_2','pt_1']].min(axis=1)) &
    (transect_df['pt_0'] > transect_df[['pt_m2','pt_m1']].min(axis=1))
).astype(int)

print("Saving transect_df.csv...")
transect_df.to_csv("../data/transect_df.csv", index=True)

#########################################################################
# Create WDPA level dataset
#########################################################################

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

print("Computing WDPA-level statistics...")
wdpa_list = []
for (wdpa, year), group in transect_df.groupby(['WDPA_PID','year']):
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
        'slope_mean': group['slope_pt0'].mean()
    })
del transect_df

print("Merging with attributes...")
wdpa_df = pd.DataFrame(wdpa_list)
attributes = pd.read_csv("../data/attributes_final.csv")
wdpa_df = pd.merge(wdpa_df, attributes, on='WDPA_PID', how='left')

print("Saving wdpa_df.csv...")
wdpa_df.to_csv("../data/wdpa_df.csv", index=False)
print("Done!")