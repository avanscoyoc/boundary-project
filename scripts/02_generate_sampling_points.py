"""
Script 02: Select Protected Areas and Generate Transect Sampling Points

Part 1 - Choose PAs (equal-area Mollweide projection):
1. Fill holes smaller than 2.25 km²
2. Remove duplicates (overlap > 90% of both PAs; oldest kept) and nested PAs
   (overlap > 90% of the smaller PA only; larger kept)
3. Merge PAs with the same ORIG_NAME + ISO3 that touch or overlap (oldest
   non-zero establishment year kept; merged PIDs listed in MERGED_PID)
4. Add geodesic area, perimeter, and shape ratio (information only; no shape filter)

Part 2 - Create transects (each PA in its own azimuthal equidistant projection):
1. Remove PAs with no interior at least 5.5 km from the edge
2. Boundary points every 500 m; perpendicular transects of 5 points 2.5 km apart
   (2 inner, 1 boundary, 2 outer)
3. Keep a transect only if its inner half lies within the PA and its outer
   half lies outside it (crosses the boundary once)
4. Remove PAs whose longest unsampled stretch of boundary (MAX_GAP_PCT) is
   more than 25% of the boundary
5. Flag transects whose outer points fall inside another PA (in_other_pa)
6. Write upload zips of 400 whole PAs each, and all points to Parquet

Input:  data/01_wdpa/WDPA_polygons.geojson
Output: data/02_pas/pas_selected.gpkg, data/02_pas/pas_removed.gpkg
        data/02_transects/transect_points.parquet
        data/02_transects/upload/transect_points_{RUN_TAG}_NNN.zip
        results/pipeline_counts.json (sections '02_pas', '02_transects')

MANUAL STEP REQUIRED AFTER RUNNING:
Upload each zip in data/02_transects/upload/ to Google Earth Engine as a table asset.

Then run: python scripts/03_sample_satellite_imagery.py
"""

import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from modules import config
from modules.counts import update_counts
from modules.preprocessing import (
    fill_holes, resolve_overlaps, merge_same_name, add_shape_metrics, generate_transects
)

# Attribute columns kept in pas_removed.gpkg (plus step, reason, replaced_by)
REMOVED_COLS = ['WDPA_PID', 'WDPAID', 'NAME', 'ORIG_NAME', 'ISO3', 'DESIG_ENG', 'IUCN_CAT',
                'STATUS', 'STATUS_YR', 'GIS_AREA', 'BIOME_NAME', 'PCT_KEPT', 'MAX_GAP_PCT']

t_start = time.time()
print("=" * 80)
print("SCRIPT 02: Select Protected Areas and Generate Transect Sampling Points")
print("=" * 80)

# =========================================================================
# PART 1: CHOOSE PROTECTED AREAS
# =========================================================================

print("\n" + "=" * 80)
print("PART 1: Choose Protected Areas")
print("=" * 80)

if not config.WDPA_POLYGONS.exists():
    raise FileNotFoundError(
        f"Input file not found: {config.WDPA_POLYGONS}\n"
        f"Download WDPA_polygons.geojson from Google Drive first "
        f"(see output from script 01_collect_protected_areas.py)."
    )

print(f"\nLoading {config.WDPA_POLYGONS}...")
wdpa_ll = gpd.read_file(config.WDPA_POLYGONS, engine='pyogrio', use_arrow=True)
wdpa_ll['WDPA_PID'] = wdpa_ll['WDPA_PID'].astype(str)
wdpa_ll = wdpa_ll.drop(columns=['geometry_type'], errors='ignore')
n_input = len(wdpa_ll)
print(f"Loaded {n_input:,} PAs (already filtered by script 01).")
pas = wdpa_ll.to_crs(config.PROCESSING_CRS)

steps = []
removed_frames = []


def record_removed(removed, step):
    """Keep removed PAs (with step, reason, replaced_by) for pas_removed.gpkg."""
    if len(removed):
        removed = removed.copy()
        removed['step'] = step
        removed_frames.append(removed)


# Step 1: Fill small holes
print(f"\nStep 1: Filling holes < {config.MAX_HOLE_AREA / 1_000_000:.2f} km²...")
pas, n_filled = fill_holes(pas, max_hole_area=config.MAX_HOLE_AREA)
steps.append({'step': '1_fill_holes', 'n_in': n_input, 'pas_with_holes_filled': n_filled, 'n_out': len(pas)})
print(f"  Filled small holes in {n_filled:,} PAs")

# Step 2: Duplicates and nested PAs
print(f"\nStep 2: Removing duplicate and nested PAs (> {config.OVERLAP_THRESHOLD}% overlap)...")
pas, removed, c = resolve_overlaps(pas, overlap_threshold=config.OVERLAP_THRESHOLD)
record_removed(removed, '2_overlap')
steps.append({'step': '2_overlap', **c})
print(f"  Removed {c['removed_duplicate']:,} duplicates and {c['removed_nested']:,} nested PAs "
      f"({c['n_invalid_fixed']} invalid geometries repaired) -> {c['n_out']:,} PAs")

# Step 3: Merge same-name PAs that touch or overlap
print("\nStep 3: Merging same-name PAs (ORIG_NAME + ISO3) that touch or overlap...")
pas, removed, c = merge_same_name(pas, max_hole_area=config.MAX_HOLE_AREA)
record_removed(removed, '3_same_name')
steps.append({'step': '3_same_name', **c})
print(f"  Merged {c['pas_merged_away']:,} PAs into {c['groups_merged']:,} PAs "
      f"({c['groups_not_single_polygon']} groups left separate: not a single polygon) -> {c['n_out']:,} PAs")

# Step 4: Area, perimeter, and shape ratio (information only; no shape filter)
print("\nStep 4: Computing geodesic area, perimeter, and shape ratio...")
pas = add_shape_metrics(pas)

print(f"\nPart 1 done ({(time.time() - t_start) / 60:.1f} min)")

# =========================================================================
# PART 2: CREATE TRANSECTS
# =========================================================================

print("\n" + "=" * 80)
print("PART 2: Create Transects")
print("=" * 80)

# Reference PAs for the other-PA flag: all script-01 polygons (EPSG:4326)
tree = shapely.STRtree(np.asarray(wdpa_ll.geometry.values, dtype=object))
tree_pids = wdpa_ll['WDPA_PID'].to_numpy()

# PIDs that count as the same PA: itself, plus PAs merged into it or removed as
# its duplicate or nested PA (following chains, e.g. nested -> duplicate -> kept)
replaced = {}
for frame in removed_frames:
    for pid, rep in zip(frame['WDPA_PID'], frame['replaced_by']):
        if rep:
            replaced[pid] = rep


def resolve(pid):
    seen = set()
    while pid in replaced and pid not in seen:
        seen.add(pid)
        pid = replaced[pid]
    return pid


own_pids = {pid: {pid} for pid in pas['WDPA_PID']}
for pid in replaced:
    root = resolve(pid)
    if root in own_pids:
        own_pids[root].add(pid)

print(f"\nGenerating transects for {len(pas):,} PAs...")
print(f"  Boundary point spacing: {config.BOUNDARY_SPACING} m")
print(f"  Transect point spacing: {config.TRANSECT_SPACING} m")
print(f"  Points per transect: {2 * config.POINTS_PER_SIDE + 1}")
print(f"  Inner buffer: {config.BUFFER_DIST} m")
print(f"  PAs per upload file: {config.CHUNK_SIZE}")
print(f"  Coverage limit: longest unsampled stretch <= {config.MAX_GAP_PCT_LIMIT}% of the boundary")

res = generate_transects(
    pas, src_crs=config.PROCESSING_CRS,
    sample_dist=config.BOUNDARY_SPACING, transect_unit=config.TRANSECT_SPACING,
    npts=config.POINTS_PER_SIDE, buffer_dist=config.BUFFER_DIST,
    chunk_size=config.CHUNK_SIZE, upload_dir=config.UPLOAD_DIR,
    upload_prefix=config.UPLOAD_PREFIX, points_path=config.TRANSECT_POINTS,
    tree=tree, tree_pids=tree_pids, own_pids=own_pids,
    max_gap_pct_limit=config.MAX_GAP_PCT_LIMIT,
)

print(f"  generated {res['n_generated']} transects")

# Transects per PA and how evenly the kept ones cover the boundary (also kept
# for PAs removed for a coverage gap, so pas_removed.gpkg shows why)
pas['N_TRNST'] = pas['WDPA_PID'].map(res['n_transects']).astype(int)
pas['N_TRNST_GEN'] = pas['WDPA_PID'].map(res['n_generated']).astype(int)
pas['PCT_KEPT'] = 100 * pas['N_TRNST'] / pas['N_TRNST_GEN'].replace(0, np.nan)
pas['MAX_GAP_PCT'] = pas['WDPA_PID'].map(res['max_gap_pct'])

status = pas['WDPA_PID'].map(res['status'])
n_in = len(pas)
for st, step in [('empty_inner_buffer', '4_inner_buffer'),
                 ('no_boundary_points', '5_no_transects'),
                 ('no_valid_transects', '5_no_transects'),
                 ('coverage_gap', '6_coverage_gap')]:
    removed = pas[status == st].copy()
    removed['reason'] = st
    removed['replaced_by'] = None
    record_removed(removed, step)
n_empty = int((status == 'empty_inner_buffer').sum())
n_none = int(status.isin(['no_boundary_points', 'no_valid_transects']).sum())
steps.append({'step': '4_inner_buffer', 'n_in': n_in, 'removed': n_empty, 'n_out': n_in - n_empty})
steps.append({'step': '5_no_transects', 'n_in': n_in - n_empty, 'removed': n_none,
              'n_out': n_in - n_empty - n_none})
n_gap = int((status == 'coverage_gap').sum())
steps.append({'step': '6_coverage_gap', 'n_in': n_in - n_empty - n_none,
              'max_gap_pct_limit': config.MAX_GAP_PCT_LIMIT, 'removed': n_gap,
              'n_out': n_in - n_empty - n_none - n_gap})

pas = pas[status == 'ok'].copy()

# =========================================================================
# OUTPUTS
# =========================================================================

config.PAS_DIR.mkdir(parents=True, exist_ok=True)
for path in [config.PAS_SELECTED, config.PAS_REMOVED]:
    path.unlink(missing_ok=True)

pas.to_crs(config.STORAGE_CRS).to_file(config.PAS_SELECTED, driver='GPKG', engine='pyogrio')

removed_all = gpd.GeoDataFrame(pd.concat(removed_frames, ignore_index=True), crs=config.PROCESSING_CRS)
removed_cols = [c for c in REMOVED_COLS if c in removed_all.columns] + ['step', 'reason', 'replaced_by', 'geometry']
removed_all[removed_cols].to_crs(config.STORAGE_CRS).to_file(config.PAS_REMOVED, driver='GPKG', engine='pyogrio')

totals = res['totals']
update_counts(config.PIPELINE_COUNTS, '02_pas', {
    'input_from_script_01': n_input,
    'steps': steps,
    'final_pas': len(pas),
})
update_counts(config.PIPELINE_COUNTS, '02_transects', {
    **totals,
    'mean_transects_per_pa': totals['kept'] / max(len(pas), 1),
    'upload_files': len(res['zip_files']),
})

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
for s in steps:
    print(f"  {s['step']:<16} {s['n_in']:>6,} -> {s['n_out']:>6,}")
print(f"\nFinal PAs: {len(pas):,}")
print(f"Transects generated: {totals['generated']:,}")
print(f"  removed, inner half leaves the PA:  {totals['removed_inner_half_leaves_pa']:,}")
print(f"  removed, outer half enters the PA:  {totals['removed_outer_half_enters_pa']:,}")
print(f"  removed with their PA (coverage gap): {totals['in_pas_removed_for_coverage_gap']:,}")
print(f"  kept: {totals['kept']:,} ({totals['flagged_in_other_pa']:,} flagged in_other_pa)")
print(f"Points: {totals['points']:,}")
print(f"\nWrote {config.PAS_SELECTED}")
print(f"Wrote {config.PAS_REMOVED} ({len(removed_all):,} removed PAs)")
print(f"Wrote {config.TRANSECT_POINTS}")
print(f"Wrote counts to {config.PIPELINE_COUNTS}")
print(f"Time: {(time.time() - t_start) / 60:.1f} min")

print("\n" + "=" * 80)
print("MANUAL STEP REQUIRED:")
print("=" * 80)
print(f"Upload each file in {config.UPLOAD_DIR}/ to Google Earth Engine as a table asset:")
for zip_file in res['zip_files']:
    name = Path(zip_file).stem
    print(f"  {Path(zip_file).name}  ->  {config.GEE_ASSET_PREFIX}{name[len(config.UPLOAD_PREFIX):]}")
print("\nThen run: python scripts/03_sample_satellite_imagery.py")
print("=" * 80)
