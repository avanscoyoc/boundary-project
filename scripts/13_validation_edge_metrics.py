"""
Script 13: Validation, part 3 - accept displacements and compute edge metrics (draft for review)

For each sampled PA (script 11), accepts the first candidate displacement whose points all lie
in the PA's biome (RESOLVE BIOME_NUM, grouped as in script 04) and all have a gradient value in
every year (land with valid data). Per biome, the first VALIDATION_N_PER_BIOME PAs in draw order
with an accepted displacement are the validation PAs (reserves replace PAs without one). Edge
metrics of their displaced transects are computed with script 04's functions.

Inputs:  data/11_validation/sample.csv, candidates.csv (script 11)
         data/12_gee_validation/*.csv (script 12, downloaded from Drive)
Outputs: results/validation_displaced_pa_years.csv  one row per validation PA-year of the
                                                    displaced copy (WDPA_PID = the original PA)
         results/validation_displacement_log.csv    one row per sampled PA: candidates, the
                                                    accepted one, distance, status

Usage:
    python scripts/13_validation_edge_metrics.py [--csv-dir DIR] [--sample-dir DIR] [--out DIR]
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from modules import config
from modules.analysis import read_asset_csvs
from modules.validation import (accept_candidates, biome_from_num, displaced_metrics, select_pas, split_key)


def arg(name, default, cast):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


CSV_DIR = arg('--csv-dir', config.VALIDATION_GEE_CSV_DIR, Path)
SAMPLE_DIR = arg('--sample-dir', config.VALIDATION_DIR, Path)
OUT = arg('--out', None, Path)
YEARS = list(range(config.START_YEAR, config.END_YEAR + 1))
YEAR_COLS = [str(y) for y in YEARS]

print("=" * 80)
print("SCRIPT 13: Validation - accept displacements, edge metrics")
print("=" * 80)

sample = pd.read_csv(SAMPLE_DIR / 'sample.csv', dtype={'WDPA_PID': str})
cands = pd.read_csv(SAMPLE_DIR / 'candidates.csv', dtype={'WDPA_PID': str})
files = sorted(CSV_DIR.glob('*.csv'))
if not files:
    raise FileNotFoundError(f"No CSVs in {CSV_DIR}: run script 12 and download its exports.")

# 1. Sampled points: split the key, biome group per point
parts = []
for f in files:
    x = pd.read_csv(f, dtype={'WDPA_PID': str}, usecols=['WDPA_PID', 'BIOME_NUM'])
    parts.append(read_asset_csvs([f], YEARS).assign(BIOME_NUM=x['BIOME_NUM'].to_numpy()))
pts = pd.concat(parts, ignore_index=True)
pts['WDPA_PID'], pts['candidate'] = split_key(pts['WDPA_PID'])
pts['biome'] = biome_from_num(pts['BIOME_NUM'])
print(f"\nSampled candidate points: {len(pts):,} ({pts['WDPA_PID'].nunique():,} PAs)")

# 2. Acceptance per PA and candidate; selection with reserves
acc = accept_candidates(pts, dict(zip(sample['WDPA_PID'], sample['BIOME_NAME'])), YEARS)
accepted = acc[acc['accepted']]
selected = select_pas(sample, set(accepted['WDPA_PID']), config.VALIDATION_N_PER_BIOME)

# 3. Edge metrics of the accepted displaced transects of the selected PAs
keep = pts.merge(accepted.loc[accepted['WDPA_PID'].isin(selected), ['WDPA_PID', 'candidate']],
                 on=['WDPA_PID', 'candidate'])
metrics = displaced_metrics(keep.drop(columns=['candidate', 'BIOME_NUM', 'biome']), YEARS)

# 4. Log: one row per sampled PA
best = (acc.sort_values(['WDPA_PID', 'passes', 'share_in_biome', 'share_complete'],
                        ascending=[True, False, False, False]).groupby('WDPA_PID').head(1))
log = sample.merge(best[['WDPA_PID', 'candidate', 'passes', 'share_in_biome', 'share_complete']]
                   .rename(columns={'candidate': 'best_candidate'}), on='WDPA_PID', how='left')
log = log.merge(accepted[['WDPA_PID', 'candidate']].rename(columns={'candidate': 'accepted_candidate'}),
                on='WDPA_PID', how='left')
log = log.merge(cands.rename(columns={'candidate': 'accepted_candidate'}), on=['WDPA_PID', 'accepted_candidate'],
                how='left')
log['status'] = np.select(
    [log['WDPA_PID'].isin(selected), log['accepted_candidate'].notna(), log['n_candidates'] == 0],
    ['selected', 'accepted, not needed (reserve)', 'failed: no displacement passed the local checks'],
    'failed: no candidate in the biome with valid data')

out_metrics = (OUT / config.VALIDATION_PA_METRICS.name) if OUT else config.VALIDATION_PA_METRICS
out_log = (OUT / config.VALIDATION_LOG.name) if OUT else config.VALIDATION_LOG
out_metrics.parent.mkdir(parents=True, exist_ok=True)
metrics.to_csv(out_metrics, index=False)
log.to_csv(out_log, index=False)

print("\nStatus by biome:")
print(log.groupby(['BIOME_NAME', 'status']).size().unstack(fill_value=0).to_string())
print(f"\nValidation PAs: {len(selected):,}; displaced PA-years: {len(metrics):,}")
print(f"Wrote {out_metrics} and {out_log}")
