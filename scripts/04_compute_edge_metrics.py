"""
Script 04: Compute Edge Metrics

Turns the Earth Engine exports into transect-year data and PA-year edge metrics
with per-PA trends.

Workflow (one asset at a time; every PA lies within a single asset):
1. Read the asset's CSV parts and check every exported point arrived exactly once
2. Build one row per complete transect-year (all 5 points have a value):
   edge flag, water_extent and in_other_pa flags, covariates
3. Compute PA-year metrics for four versions: all transects, without water,
   without other-PA, without either flag
4. Merge PA attributes (from pas_selected.gpkg), recategorize biomes, and add
   per-PA OLS trends for every extent and intensity metric

Edge: pt_0 > min(pt_1, pt_2) and pt_0 > min(pt_m1, pt_m2).
Edge intensity: mean of the four Cohen's d values (pt_0 vs each position),
computed on edge transects only.

BEFORE RUNNING:
Download all CSVs from the Google Drive folder (see script 03 output) into data/03_gee/

OUTPUT:
- data/04_transect_years/ndvi_NNN.parquet  (one per asset)
- results/pa_edge_metrics_ndvi.csv           (FINAL: one row per PA-year)
- results/pipeline_counts.json               (section '04_ndvi')
"""

import sys
import gc
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import pyogrio
from modules import config
from modules.counts import update_counts
from modules.analysis import (
    find_asset_csvs, read_asset_csvs, build_transect_years, pa_year_metrics,
    recategorize_biome, add_trends,
)

t_start = time.time()
print("=" * 80)
print("SCRIPT 04: Compute Edge Metrics")
print("=" * 80)

years = list(range(config.START_YEAR, config.END_YEAR + 1))

# Inputs
for path in [config.TRANSECT_POINTS, config.PAS_SELECTED]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run script 02 first.")
asset_csvs, unrecognized = find_asset_csvs(config.GEE_CSV_DIR)
if not asset_csvs:
    raise FileNotFoundError(
        f"No CSV files found in {config.GEE_CSV_DIR}. Download them from Google Drive "
        f"folder '{config.GEE_DRIVE_FOLDER}' first (see script 03 output)."
    )
if unrecognized:
    print(f"WARNING: skipping {len(unrecognized)} CSVs with unexpected names: {unrecognized[:5]}")

# Expected points (script 02). Upload files hold CHUNK_SIZE whole PAs in order,
# so a PA's asset number is its position in the file // CHUNK_SIZE.
print(f"\nReading {config.TRANSECT_POINTS}...")
points = pd.read_parquet(config.TRANSECT_POINTS, columns=['WDPA_PID', 'transectID', 'pointID', 'in_other_pa'])
pa_order = pd.unique(points['WDPA_PID'])
asset_of_pa = pd.Series([f'{i // config.CHUNK_SIZE:03d}' for i in range(len(pa_order))], index=pa_order)
points['asset'] = points['WDPA_PID'].map(asset_of_pa)
n_generated = int((points['pointID'] == 0).sum())
print(f"Expected: {len(points):,} points, {n_generated:,} transects, {len(pa_order):,} PAs, "
      f"{points['asset'].nunique()} assets")

missing_assets = sorted(set(points['asset']) - set(asset_csvs))
if missing_assets:
    raise FileNotFoundError(f"No CSVs found for assets {missing_assets} in {config.GEE_CSV_DIR}")

# Process one asset at a time
config.TRANSECT_YEARS_DIR.mkdir(parents=True, exist_ok=True)
for old in config.TRANSECT_YEARS_DIR.glob(f'{config.INDEX_NAME}_*.parquet'):
    old.unlink()

metrics = []
completeness = {'points_expected': len(points), 'points_received': 0, 'points_missing': 0,
                'points_unexpected': 0}
per_year = {y: 0 for y in years}
transect_counts = {'generated': n_generated, 'received': 0, 'missing_points': 0}
key = ['WDPA_PID', 'transectID', 'pointID']

for asset, files in asset_csvs.items():
    df = read_asset_csvs(files, years)
    expected = points.loc[points['asset'] == asset, key]
    check = expected.merge(df[key], on=key, how='outer', indicator=True)['_merge'].value_counts()
    completeness['points_received'] += len(df)
    completeness['points_missing'] += int(check.get('left_only', 0))
    completeness['points_unexpected'] += int(check.get('right_only', 0))

    flags = points.loc[(points['asset'] == asset) & (points['pointID'] == 0),
                       ['WDPA_PID', 'transectID', 'in_other_pa']]
    ty, c = build_transect_years(df, years, flags)
    transect_counts['received'] += c['transects_received']
    transect_counts['missing_points'] += c['transects_missing_points']
    for y, n in c['complete_per_year'].items():
        per_year[int(y)] += int(n)

    ty.to_parquet(config.TRANSECT_YEARS_DIR / f'{config.INDEX_NAME}_{asset}.parquet', index=False)
    metrics.append(pa_year_metrics(ty))
    print(f"  asset {asset}: {len(files)} CSVs, {len(df):,} points, "
          f"{int(check.get('left_only', 0)):,} missing, {len(ty):,} transect-years")
    del df, ty, expected, check, flags
    gc.collect()

if completeness['points_missing'] or completeness['points_unexpected']:
    print(f"WARNING: {completeness['points_missing']:,} expected points missing and "
          f"{completeness['points_unexpected']:,} unexpected points in the CSVs. "
          f"Check that every CSV part was downloaded.")

# PA-year table with attributes and trends
print("\nMerging attributes and computing trends...")
pa = pd.concat(metrics, ignore_index=True)
attributes = pyogrio.read_dataframe(config.PAS_SELECTED, read_geometry=False)
attributes['WDPA_PID'] = attributes['WDPA_PID'].astype(str)
pa = pa.merge(attributes, on='WDPA_PID', how='left')
pa['BIOME_NAME'] = pa['BIOME_NAME'].apply(recategorize_biome)
# Share of the PA's transects with data at all 5 points that year
pa['pct_complete'] = 100 * pa['n_trnst'] / pa['N_TRNST']
pa = add_trends(pa.sort_values(['WDPA_PID', 'year']).reset_index(drop=True))

config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
pa.to_csv(config.PA_METRICS, index=False)

flag_counts = points.loc[points['pointID'] == 0, 'in_other_pa'].value_counts().to_dict()
update_counts(config.PIPELINE_COUNTS, '04_ndvi', {
    'assets': len(asset_csvs),
    'csv_files': sum(len(f) for f in asset_csvs.values()),
    'completeness': completeness,
    'transects': {
        **transect_counts,
        'flagged_in_other_pa': int(flag_counts.get(1, 0)),
    },
    'per_year': {str(y): {'complete': per_year[y],
                          'dropped_missing_data': n_generated - per_year[y]} for y in years},
    'pa_years': len(pa),
    'pas': int(pa['WDPA_PID'].nunique()),
})

print("\n" + "=" * 80)
print("PROCESSING COMPLETE")
print("=" * 80)
print(f"PAs: {pa['WDPA_PID'].nunique():,} | PA-years: {len(pa):,}")
print(f"Transect-years: {config.TRANSECT_YEARS_DIR}/")
print(f"Final dataset: {config.PA_METRICS}")
print(f"Counts: {config.PIPELINE_COUNTS}")
print(f"Time: {(time.time() - t_start) / 60:.1f} min")
print("=" * 80)
