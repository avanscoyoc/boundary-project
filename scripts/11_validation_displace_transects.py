"""
Script 11: Validation, part 1 - displaced transects (draft for review)

For a stratified random sample of analysis PAs (VALIDATION_N_PER_BIOME per biome, plus
VALIDATION_RESERVE reserves drawn in order), translates each PA's full transect set by random
vectors (no rotation) in a local azimuthal equidistant projection, and keeps up to
VALIDATION_N_CANDIDATES displacements that pass the checks needing no imagery:
- distance >= max(VALIDATION_MIN_DIST_KM, the PA's footprint), so the copy cannot overlap
  the PA; drawn uniformly in [minimum, minimum + VALIDATION_DIST_SPAN_KM];
- no displaced transect crosses the boundary of any PA of the 6,570-PA WDPA set
  (pas_selected + pas_removed), the PA itself included.
The imagery checks (every point in the PA's biome and with gradient data in every year) need
Earth Engine: script 12 samples the candidates, script 13 accepts per PA the first passing
candidate and replaces PAs without one by the biome's reserves.

Inputs:  data/05_analysis/pa_level.csv (the analysis PAs, R script 05),
         data/02_transects/transect_points.parquet, data/02_pas/pas_{selected,removed}.gpkg
Outputs: data/11_validation/sample.csv            sampled PAs: biome, draw order, transects,
                                                  attempts, candidates, minimum distance
         data/11_validation/candidates.csv        one row per candidate displacement
         data/11_validation/candidate_points.parquet  displaced points of every candidate
         data/11_validation/upload/displaced_points_{RUN_TAG}_NNN.zip  for Earth Engine
             (field WDPA_PID = "<WDPA_PID>#<candidate>", as the key of script 13)

Usage:
    python scripts/11_validation_displace_transects.py [--n-per-biome N] [--reserve R] [--out DIR]
(--out writes everything under DIR instead of data/11_validation, e.g. for a test.)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyogrio
import shapely

from modules import config
from modules.preprocessing import write_upload_zip
from modules.validation import (sample_pas, candidate_displacements, aeqd_transformers, footprint_m,
                                CANDIDATE_SEP)


def arg(name, default, cast):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


N_PER_BIOME = arg('--n-per-biome', config.VALIDATION_N_PER_BIOME, int)
RESERVE = arg('--reserve', config.VALIDATION_RESERVE, int)
OUT_DIR = arg('--out', config.VALIDATION_DIR, Path)
UPLOAD_DIR = OUT_DIR / 'upload'
POINTS_PER_ZIP = 1_000_000

print("=" * 80)
print("SCRIPT 11: Validation - displaced transects (local checks)")
print("=" * 80)

# 1. Sample of analysis PAs, stratified by biome
frame = pd.read_csv(config.VALIDATION_SAMPLE_FRAME, dtype={'WDPA_PID': str}, usecols=['WDPA_PID', 'BIOME_NAME'])
sample = sample_pas(frame, N_PER_BIOME, RESERVE, config.VALIDATION_SEED)
print(f"\nSample frame: {len(frame):,} analysis PAs; drawn {len(sample):,} "
      f"({N_PER_BIOME} per biome + {RESERVE} reserves)")

# 2. Their transect points
pts = pq.read_table(config.TRANSECT_POINTS, columns=['WDPA_PID', 'transectID', 'pointID', 'lon', 'lat'],
                    filters=[('WDPA_PID', 'in', sample['WDPA_PID'].tolist())]).to_pandas()
pts['WDPA_PID'] = pts['WDPA_PID'].astype(str)
by_pa = dict(tuple(pts.groupby('WDPA_PID', sort=False)))
missing = sorted(set(sample['WDPA_PID']) - set(by_pa))
if missing:
    raise ValueError(f"{len(missing)} sampled PAs have no transect points, e.g. {missing[:3]}")


def boundaries_near(lon_min, lat_min, lon_max, lat_max):
    """STRtree of the boundaries of all WDPA PAs (selected + removed) intersecting the box."""
    geoms = []
    for f in (config.PAS_SELECTED, config.PAS_REMOVED):
        g = pyogrio.read_dataframe(f, columns=[], bbox=(lon_min, lat_min, lon_max, lat_max))
        if len(g):
            if g.crs is not None and g.crs.to_epsg() != 4326:
                g = g.to_crs(4326)
            geoms.append(shapely.boundary(g.geometry.to_numpy()))
    return shapely.STRtree(np.concatenate(geoms) if geoms else np.array([], dtype=object))


# 3. Candidate displacements, PA by PA in sample order (one random stream)
rng = np.random.default_rng(config.VALIDATION_SEED + 1)
d_min, d_span = 1000 * config.VALIDATION_MIN_DIST_KM, 1000 * config.VALIDATION_DIST_SPAN_KM
rows, cand_rows, cand_points = [], [], []
for i, s in enumerate(sample.itertuples(index=False), start=1):
    p = by_pa[s.WDPA_PID]
    # search box: the PA's points plus the largest possible shift
    fwd, _ = aeqd_transformers(p['lon'].mean(), p['lat'].mean())
    reach = max(d_min, footprint_m(*fwd.transform(p['lon'].to_numpy(), p['lat'].to_numpy()))) + d_span
    dlat = reach / 111_000 + 0.1
    dlon = min(180.0, reach / (111_000 * max(np.cos(np.deg2rad(p['lat'].abs().max() + dlat)), 0.05)) + 0.1)
    tree = boundaries_near(p['lon'].min() - dlon, max(-90, p['lat'].min() - dlat),
                           p['lon'].max() + dlon, min(90, p['lat'].max() + dlat))
    cands, attempts, d_used = candidate_displacements(p, tree, rng, config.VALIDATION_N_CANDIDATES,
                                                      config.VALIDATION_MAX_ATTEMPTS, d_min, d_span)
    rows.append({'WDPA_PID': s.WDPA_PID, 'BIOME_NAME': s.BIOME_NAME, 'draw': s.draw,
                 'reserve': s.draw > N_PER_BIOME, 'n_transects': p['transectID'].nunique(),
                 'attempts': attempts, 'n_candidates': len(cands), 'min_dist_km': d_used / 1000})
    for c in cands:
        cand_rows.append({'WDPA_PID': s.WDPA_PID, **{k: c[k] for k in ('candidate', 'attempt', 'bearing')},
                          'dist_km': c['dist_m'] / 1000, 'dx_m': c['dx'], 'dy_m': c['dy']})
        cp = c['points'].assign(WDPA_PID=s.WDPA_PID, candidate=c['candidate'])
        cand_points.append(cp[['WDPA_PID', 'candidate', 'transectID', 'pointID', 'lon', 'lat']])
    if i % 50 == 0 or i == len(sample):
        print(f"  {i:,} of {len(sample):,} PAs")

sample_out = pd.DataFrame(rows)
cands = pd.DataFrame(cand_rows)
points = pd.concat(cand_points, ignore_index=True)

# 4. Write: tables, and upload zips of whole PAs (all candidates of a PA in the same zip)
OUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
sample_out.to_csv(OUT_DIR / 'sample.csv', index=False)
cands.to_csv(OUT_DIR / 'candidates.csv', index=False)
points.to_parquet(OUT_DIR / 'candidate_points.parquet', index=False)
for old in UPLOAD_DIR.glob(f'{config.VALIDATION_UPLOAD_PREFIX}*.zip'):
    old.unlink()   # a rerun replaces the previous upload files
upload = points.assign(WDPA_PID=points['WDPA_PID'] + CANDIDATE_SEP + points['candidate'].astype(str))
sizes = points.groupby('WDPA_PID', sort=False).size()
zip_of = (sizes.cumsum() - 1) // POINTS_PER_ZIP
n_zip = int(zip_of.max()) + 1 if len(zip_of) else 0
for z in range(n_zip):
    pids = set(zip_of.index[zip_of.to_numpy() == z])
    write_upload_zip(upload[points['WDPA_PID'].isin(pids)],
                     UPLOAD_DIR / f'{config.VALIDATION_UPLOAD_PREFIX}{z:03d}.zip')

# 5. Summary
summ = sample_out.groupby('BIOME_NAME').agg(drawn=('WDPA_PID', 'size'),
                                            with_candidate=('n_candidates', lambda x: int((x > 0).sum())),
                                            median_attempts=('attempts', 'median'),
                                            candidates=('n_candidates', 'sum'))
print("\nBy biome:")
print(summ.to_string())
print(f"\nCandidate points: {len(points):,} in {n_zip} upload file(s) in {UPLOAD_DIR}")
print(f"Wrote {OUT_DIR / 'sample.csv'}, candidates.csv, candidate_points.parquet")
print("\nNEXT (manual): upload each zip as an Earth Engine table asset named "
      f"{config.VALIDATION_ASSET_PREFIX}NNN, then run script 12.")
