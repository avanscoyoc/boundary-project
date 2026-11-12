"""
Script 13: Validation, part 3 - accept displacements and compute edge metrics

Each candidate copy (script 11) is measured with script 04's functions and must pass the
analysis-set rules of R script 05, as the analysed PAs did:
- biome: the PA polygon moved by the same vector has the PA's biome (largest-area RESOLVE
  ecoregion, as script 01; grouped as script 04; from script 12);
- sea: median over years of the % of transects complete >= VALIDATION_MIN_COMPLETE_PCT;
- lake: mean over years of the % of transects with surface water <= VALIDATION_MAX_WATER_PCT;
- years: >= VALIDATION_MIN_YEARS years of both edge metrics.
Per PA the first passing candidate in draw order is accepted. Per biome, the first
VALIDATION_N_PER_BIOME PAs in draw order with an accepted copy are the validation PAs
(reserves replace PAs without one).

Inputs:  data/11_validation/sample.csv, candidates.csv (script 11)
         data/12_gee_validation/*.csv (script 12, downloaded from Drive): the point parts and
         displaced_polygons_biome_NNN.csv
Outputs: results/validation_displaced_pa_years.csv  one row per validation PA-year of the
                                                    displaced copy (WDPA_PID = the original PA)
         results/validation_candidate_checks.csv    one row per candidate: the rule values and
                                                    results, passes, accepted
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
from modules.analysis import find_asset_csvs, read_asset_csvs, recategorize_biome
from modules.validation import (CHECK_RULES, biome_from_num, candidate_checks, displaced_metrics,
                                select_pas, split_key)


def arg(name, default, cast):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


CSV_DIR = arg('--csv-dir', config.VALIDATION_GEE_CSV_DIR, Path)
SAMPLE_DIR = arg('--sample-dir', config.VALIDATION_DIR, Path)
OUT = arg('--out', None, Path)
YEARS = list(range(config.START_YEAR, config.END_YEAR + 1))

print("=" * 80)
print("SCRIPT 13: Validation - accept displacements, edge metrics")
print("=" * 80)

sample = pd.read_csv(SAMPLE_DIR / 'sample.csv', dtype={'WDPA_PID': str})
cands = pd.read_csv(SAMPLE_DIR / 'candidates.csv', dtype={'WDPA_PID': str})
pa_biome = dict(zip(sample['WDPA_PID'], sample['BIOME_NAME']))

groups, unrecognized = find_asset_csvs(CSV_DIR)
biome_files = sorted(CSV_DIR.glob(f'{config.VALIDATION_BIOME_EXPORT}_[0-9][0-9][0-9].csv'))
other = [f for f in unrecognized if not f.startswith(config.VALIDATION_BIOME_EXPORT)]
if not groups:
    raise FileNotFoundError(f"No point CSVs in {CSV_DIR}: run script 12 and download its exports.")
if not biome_files:
    raise FileNotFoundError(f"No {config.VALIDATION_BIOME_EXPORT}_NNN.csv in {CSV_DIR}: download the "
                            "polygon biome exports of script 12.")
if other:
    raise ValueError(f"Unrecognised CSVs in {CSV_DIR} (Drive duplicates?): {other[:5]}")

# 1. Every candidate copy, asset by asset (all points of a PA are in one asset): script 04's
#    PA-year metrics, and the share of points in the PA's biome (a diagnostic only)
py_parts, pt_parts = [], []
for asset, files in groups.items():
    df = read_asset_csvs(files, YEARS)
    bnum = pd.concat([pd.read_csv(f, usecols=['BIOME_NUM']) for f in files], ignore_index=True)
    pid, cand = split_key(df['WDPA_PID'])
    in_biome = pd.Series(biome_from_num(bnum['BIOME_NUM'])) == pd.Series(pid).map(pa_biome)
    pt_parts.append(pd.DataFrame({'WDPA_PID': pid, 'candidate': cand, 'in_biome': in_biome.to_numpy()})
                    .groupby(['WDPA_PID', 'candidate']).agg(n_points=('in_biome', 'size'),
                                                            share_points_in_biome=('in_biome', 'mean'))
                    .reset_index())
    m = displaced_metrics(df, YEARS)                       # WDPA_PID is the "<PID>#<candidate>" key
    m['WDPA_PID'], m['candidate'] = split_key(m['WDPA_PID'])
    py_parts.append(m)
    print(f"  asset {asset}: {len(df):,} points, {m[['WDPA_PID', 'candidate']].drop_duplicates().shape[0]:,} "
          "candidates with data")
pa_years = pd.concat(py_parts, ignore_index=True)
pts_diag = pd.concat(pt_parts, ignore_index=True)
# every candidate point must have come back (as script 04 checks the PA points)
exp = cands[['WDPA_PID', 'candidate']].merge(sample[['WDPA_PID', 'n_transects']], on='WDPA_PID')
exp = exp.merge(pts_diag[['WDPA_PID', 'candidate', 'n_points']], on=['WDPA_PID', 'candidate'], how='left')
short = exp['n_points'].fillna(0) != 5 * exp['n_transects']
if short.any():
    raise ValueError(f"{int(short.sum())} candidates do not have all their points in the CSVs "
                     f"(e.g. {exp.loc[short, ['WDPA_PID', 'candidate']].head(3).values.tolist()}): "
                     "is an export part missing or downloaded twice?")

# 2. Biome of each displaced polygon (script 12: get_biome, as script 01), grouped as script 04
pb = pd.concat([pd.read_csv(f, dtype={'WDPA_PID': str, 'BIOME_NAME': str}, keep_default_na=False)
                for f in biome_files], ignore_index=True)
if pb['WDPA_PID'].duplicated().any():
    raise ValueError("A displaced polygon appears twice in the biome exports (downloaded twice?)")
pb['WDPA_PID'], pb['candidate'] = split_key(pb['WDPA_PID'])
pb['biome'] = [recategorize_biome(b) for b in pb['BIOME_NAME']]

# 3. The analysis-set rules per candidate; selection with reserves
chk = candidate_checks(cands, pa_years, dict(zip(sample['WDPA_PID'], sample['n_transects'])), pb, pa_biome,
                       config.VALIDATION_MIN_COMPLETE_PCT, config.VALIDATION_MAX_WATER_PCT,
                       config.VALIDATION_MIN_YEARS)
chk = chk.merge(pts_diag, on=['WDPA_PID', 'candidate'], how='left')
no_poly = ~chk.set_index(['WDPA_PID', 'candidate']).index.isin(pb.set_index(['WDPA_PID', 'candidate']).index)
if no_poly.any():
    raise ValueError(f"{int(no_poly.sum())} candidates have no polygon biome (a polygon export missing?)")
accepted = chk[chk['accepted']]
selected = select_pas(sample, set(accepted['WDPA_PID']), config.VALIDATION_N_PER_BIOME)

# 4. Edge metrics of the accepted copies of the selected PAs
metrics = pa_years.merge(accepted.loc[accepted['WDPA_PID'].isin(selected), ['WDPA_PID', 'candidate']],
                         on=['WDPA_PID', 'candidate']).drop(columns='candidate')
metrics = metrics.sort_values(['WDPA_PID', 'year'], kind='stable').reset_index(drop=True)

# 5. Log: one row per sampled PA (failing candidates counted per rule)
fails = (chk.assign(**{f'n_fail_{r[:-3]}': ~chk[r] for r in CHECK_RULES})
            .groupby('WDPA_PID')[[f'n_fail_{r[:-3]}' for r in CHECK_RULES]].sum().reset_index())
log = sample.merge(fails, on='WDPA_PID', how='left')
log = log.merge(accepted[['WDPA_PID', 'candidate', 'pct_complete', 'water_extent_pct', 'share_points_in_biome']]
                .rename(columns={'candidate': 'accepted_candidate'}), on='WDPA_PID', how='left')
log = log.merge(cands.rename(columns={'candidate': 'accepted_candidate'}), on=['WDPA_PID', 'accepted_candidate'],
                how='left')
log['status'] = np.select(
    [log['WDPA_PID'].isin(selected), log['accepted_candidate'].notna(), log['n_candidates'] == 0],
    ['selected', 'accepted, not needed (reserve)', 'failed: no displacement passed the local checks'],
    'failed: no candidate met the analysis-set rules')

out_metrics = (OUT / config.VALIDATION_PA_METRICS.name) if OUT else config.VALIDATION_PA_METRICS
out_checks = (OUT / config.VALIDATION_CHECKS.name) if OUT else config.VALIDATION_CHECKS
out_log = (OUT / config.VALIDATION_LOG.name) if OUT else config.VALIDATION_LOG
out_metrics.parent.mkdir(parents=True, exist_ok=True)
metrics.to_csv(out_metrics, index=False)
chk.to_csv(out_checks, index=False)
log.to_csv(out_log, index=False)

print("\nCandidates failing each rule (a candidate can fail several):")
print(chk.merge(sample[['WDPA_PID', 'BIOME_NAME']], on='WDPA_PID')
         .groupby('BIOME_NAME')[list(CHECK_RULES) + ['passes']].agg(lambda x: int((~x).sum()))
         .rename(columns={'passes': 'any rule'}).to_string())
print("\nStatus by biome:")
print(log.groupby(['BIOME_NAME', 'status']).size().unstack(fill_value=0).to_string())
print(f"\nValidation PAs: {len(selected):,}; displaced PA-years: {len(metrics):,}")
print(f"Wrote {out_metrics}, {out_checks.name} and {out_log.name}")
