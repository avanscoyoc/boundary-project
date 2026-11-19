"""
Script 11: Validation, part 1 - displaced transects.

Draws a biome-stratified random sample of analysis PAs (VALIDATION_N_PER_BIOME per biome, plus
VALIDATION_RESERVE reserves drawn in order). Each PA's full transect set is shifted by random
vectors (no rotation) in a local azimuthal equidistant projection. Up to VALIDATION_N_CANDIDATES
displacements are kept that pass the checks needing no imagery:
- distance drawn uniformly in [d, VALIDATION_DIST_MULT × d], with
  d = max(VALIDATION_MIN_DIST_KM, PA footprint), so the copy cannot overlap the PA;
- at most VALIDATION_MAX_BOUNDARY_SHARE of the displaced transects have a PA boundary
  (pas_selected + pas_removed, the PA itself included) in their centre zone (points -1 to 1,
  within 2.5 km of the centre point). The share is recorded.
The imagery checks (the analysis-set rules of R script 05: biome of the moved PA polygon, sea,
lake, years) follow in scripts 12 and 13. Script 13 accepts each PA's first passing candidate
and replaces PAs without one by the biome's reserves.

Run: python scripts/11_validation_displace_transects.py

Inputs:  data/05_analysis/pa_level.csv (analysis PAs, R script 05),
         data/02_transects/transect_points.parquet, data/02_pas/pas_{selected,removed}.gpkg
Outputs (data/11_validation/):
  sample.csv: sampled PAs (biome, draw order, transects, attempts, candidates, minimum distance)
  candidates.csv: one row per candidate, with its share of transects near a PA boundary
  candidate_points.parquet: displaced points of every candidate
  upload/displaced_points_NNN.zip, upload/displaced_polygons_NNN.zip: the displaced points and
    the PA polygon moved by each candidate's vector, keyed WDPA_PID = "<WDPA_PID>#<candidate>"

After running: upload each zip as an Earth Engine table asset of the same name, then run
script 12.
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
from modules.transects import write_upload_zip
from modules.validation import (sample_pas, candidate_displacements, aeqd_transformers, footprint_m,
                                displacement_centre, displace_polygon, write_polygon_zip, CANDIDATE_SEP)


N_PER_BIOME = config.VALIDATION_N_PER_BIOME
RESERVE = config.VALIDATION_RESERVE
OUT_DIR = config.VALIDATION_DIR
UPLOAD_DIR = config.VALIDATION_UPLOAD_DIR
POINTS_PER_ZIP = 1_000_000
POLY_VERTICES_PER_ZIP = 2_500_000

print("=" * 80)
print("SCRIPT 11: Validation - displaced transects (local checks)")
print("=" * 80)

# 1. Sample of analysis PAs, stratified by biome
frame = pd.read_csv(config.VALIDATION_SAMPLE_FRAME, dtype={'WDPA_PID': str}, usecols=['WDPA_PID', 'BIOME_NAME'])
sample = sample_pas(frame, N_PER_BIOME, RESERVE, config.VALIDATION_SEED)
print(f"\nSample frame: {len(frame):,} analysis PAs; drawn {len(sample):,} "
      f"({N_PER_BIOME} per biome + {RESERVE} reserves)")

# 2. Transect points of the sampled PAs
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


# PA polygons from script 02 (the outlines the transects were built on), moved with each candidate
polys = pyogrio.read_dataframe(config.PAS_SELECTED, columns=['WDPA_PID'])
polys['WDPA_PID'] = polys['WDPA_PID'].astype(str)
if polys.crs is not None and polys.crs.to_epsg() != 4326:
    polys = polys.to_crs(4326)
poly_of = dict(zip(polys['WDPA_PID'], polys.geometry))
missing = sorted(set(sample['WDPA_PID']) - set(poly_of))
if missing:
    raise ValueError(f"{len(missing)} sampled PAs have no polygon in pas_selected, e.g. {missing[:3]}")

# 3. Candidate displacements, PA by PA in sample order (one random stream)
rng = np.random.default_rng(config.VALIDATION_SEED + 1)
d_min, mult = 1000 * config.VALIDATION_MIN_DIST_KM, config.VALIDATION_DIST_MULT
rows, cand_rows, cand_points, cand_polys = [], [], [], []
for i, s in enumerate(sample.itertuples(index=False), start=1):
    p = by_pa[s.WDPA_PID]
    # search box: the PA's points plus the largest possible shift
    fwd, _ = aeqd_transformers(p['lon'].mean(), p['lat'].mean())
    reach = mult * max(d_min, footprint_m(*fwd.transform(p['lon'].to_numpy(), p['lat'].to_numpy())))
    dlat = reach / 111_000 + 0.1
    dlon = min(180.0, reach / (111_000 * max(np.cos(np.deg2rad(p['lat'].abs().max() + dlat)), 0.05)) + 0.1)
    tree = boundaries_near(p['lon'].min() - dlon, max(-90, p['lat'].min() - dlat),
                           p['lon'].max() + dlon, min(90, p['lat'].max() + dlat))
    cands, attempts, d_used = candidate_displacements(p, tree, rng, config.VALIDATION_N_CANDIDATES,
                                                      config.VALIDATION_MAX_ATTEMPTS, d_min, mult,
                                                      config.VALIDATION_MAX_BOUNDARY_SHARE)
    rows.append({'WDPA_PID': s.WDPA_PID, 'BIOME_NAME': s.BIOME_NAME, 'draw': s.draw,
                 'reserve': s.draw > N_PER_BIOME, 'n_transects': p['transectID'].nunique(),
                 'attempts': attempts, 'n_candidates': len(cands), 'min_dist_km': d_used / 1000})
    for c in cands:
        cand_rows.append({'WDPA_PID': s.WDPA_PID, **{k: c[k] for k in ('candidate', 'attempt', 'bearing')},
                          'dist_km': c['dist_m'] / 1000, 'dx_m': c['dx'], 'dy_m': c['dy'],
                          'boundary_share': c['boundary_share']})
        cp = c['points'].assign(WDPA_PID=s.WDPA_PID, candidate=c['candidate'])
        cand_points.append(cp[['WDPA_PID', 'candidate', 'transectID', 'pointID', 'lon', 'lat']])
        lon0, lat0 = displacement_centre(p)
        cand_polys.append((s.WDPA_PID, c['candidate'],
                           displace_polygon(poly_of[s.WDPA_PID], lon0, lat0, c['dx'], c['dy'])))
    if i % 50 == 0 or i == len(sample):
        print(f"  {i:,} of {len(sample):,} PAs")

sample_out = pd.DataFrame(rows)
cands = pd.DataFrame(cand_rows)
points = pd.concat(cand_points, ignore_index=True)

# 4. Write tables and upload zips of whole PAs (all candidates of a PA in one zip)
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

# Displaced PA polygons (for their biome in script 12), in zips of whole PAs.
# Outlines made invalid by the move are repaired.
pg = pd.DataFrame(cand_polys, columns=['WDPA_PID', 'candidate', 'geometry'])
geoms = pg['geometry'].to_numpy()
bad = ~shapely.is_valid(geoms)
if bad.any():
    geoms[bad] = shapely.make_valid(geoms[bad])
pg['geometry'] = geoms
for old in UPLOAD_DIR.glob(f'{config.VALIDATION_POLY_UPLOAD_PREFIX}*.zip'):
    old.unlink()
nv = pd.Series(shapely.get_num_coordinates(geoms), index=pg.index).groupby(pg['WDPA_PID'], sort=False).sum()
poly_zip_of = (nv.cumsum() - 1) // POLY_VERTICES_PER_ZIP
n_poly_zip = int(poly_zip_of.max()) + 1 if len(poly_zip_of) else 0
for z in range(n_poly_zip):
    sel = pg[pg['WDPA_PID'].isin(set(poly_zip_of.index[poly_zip_of.to_numpy() == z]))]
    write_polygon_zip(sel['WDPA_PID'] + CANDIDATE_SEP + sel['candidate'].astype(str), sel['geometry'],
                      UPLOAD_DIR / f'{config.VALIDATION_POLY_UPLOAD_PREFIX}{z:03d}.zip')

# 5. Summary
summ = sample_out.groupby('BIOME_NAME').agg(drawn=('WDPA_PID', 'size'),
                                            with_candidate=('n_candidates', lambda x: int((x > 0).sum())),
                                            median_attempts=('attempts', 'median'),
                                            candidates=('n_candidates', 'sum'))
print("\nBy biome:")
print(summ.to_string())
print(f"\nCandidate points: {len(points):,} in {n_zip} upload file(s) in {UPLOAD_DIR}")
print(f"Displaced PA polygons: {len(pg):,} ({int(nv.sum()):,} vertices; {int(bad.sum())} repaired) "
      f"in {n_poly_zip} upload file(s)")
print(f"Wrote {OUT_DIR / 'sample.csv'}, candidates.csv, candidate_points.parquet")
print("\nNEXT (manual): upload each zip as an Earth Engine table asset of the same name "
      f"({config.VALIDATION_ASSET_PREFIX}NNN and {config.VALIDATION_POLY_ASSET_PREFIX}NNN), then run script 12.")
