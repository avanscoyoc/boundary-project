"""
Script 05b: Export the analysis set for visual inspection

Writes the analysis-set PAs (R script 05) for the Earth Engine inspection app
(scripts/05b_inspection_app.js), plus a checklist for the inspection notes.
Run once, before the inspection. PAs marked `exclude` in the checklist are removed
by script 05 as a counted step; scripts 02-04 do not rerun (each PA's transects
and metrics are computed on their own).

Review order: biome, country (ISO3), name (case-insensitive), then WDPA_PID.
The app's PA number is the checklist's `order` column.

Polygon fields (shapefile, names of at most 10 characters):
  ORDER      review number, 1..n
  WDPA_PID, NAME, ISO3
  BIOME      grouped biome (script 04)
  AREA_KM2   area (script 02)
  N_TRNST    transects kept by script 02 (shown as points in the app)
  PCT_KEPT   % of generated transects kept by script 02
  MISS_PCT   % of the PA's transect-years with a missing value (as 05's missing-data count)
  COMPL_PCT  median % of complete transects (05's sea rule: >= 75 kept)
  WATER_PCT  % of transects with water (05's lake rule: <= 75 kept)
  ASSET      NNN of the point asset transect_points_NNN that holds the PA

INPUT:
- data/05_analysis/pa_level.csv                         (analysis set, R script 05)
- data/02_pas/pas_selected.gpkg                         (polygons, transect counts; script 02)
- data/02_transects/upload/transect_points_NNN.zip      (the point assets; script 02)
- data/04_transect_years/missing_transects.csv          (script 04)
- results/analysis_counts.json                          (05's totals, cross-checked)

OUTPUT:
- data/inspection/pas_final.zip      (upload to Earth Engine as table asset pas_final)
- data/inspection/checklist.csv      (order, WDPA_PID, NAME, ISO3, BIOME_NAME, decision, note;
                                      UTF-8 with BOM so spreadsheet programs show names correctly)

A checklist that already has entries is never overwritten: the script stops
before writing anything.
"""

import sys
import json
import time
import zipfile
import tempfile
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import geopandas as gpd
import pyogrio
from modules import config

t_start = time.time()
print("=" * 80)
print("SCRIPT 05b: Export the analysis set for visual inspection")
print("=" * 80)

n_years = config.END_YEAR - config.START_YEAR + 1
counts_path = config.RESULTS_DIR / "analysis_counts.json"
for path in [config.INSPECTION_INPUT, config.PAS_SELECTED, config.MISSING_TRANSECTS, counts_path]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts 02, 04 and 05 first.")

# --- Never overwrite the inspection notes ---------------------------------------
checklist = config.INSPECTION_CHECKLIST
if checklist.exists():
    try:
        old = pd.read_csv(checklist, dtype=str, keep_default_na=False, encoding='utf-8-sig')
        started = (old[['decision', 'note']].apply(lambda s: s.str.strip()) != '').any().any()
    except (KeyError, ValueError, pd.errors.ParserError):
        started = True  # unreadable or edited layout: treat as started
    if started:
        sys.exit(f"STOP: {checklist} already has entries; nothing written. "
                 "Move it away first to write a new checklist and zip.")

# --- Analysis set and review order ----------------------------------------------
pa = pd.read_csv(config.INSPECTION_INPUT, dtype={'WDPA_PID': str}, encoding='utf-8')
assert pa['WDPA_PID'].is_unique
pa['_name'] = pa['NAME'].str.casefold()
pa = pa.sort_values(['BIOME_NAME', 'ISO3', '_name', 'WDPA_PID'], kind='stable').reset_index(drop=True)
pa['ORDER'] = range(1, len(pa) + 1)
print(f"\nAnalysis set: {len(pa):,} PAs (from {config.INSPECTION_INPUT.relative_to(project_root)})")

# --- Polygons and transect counts (script 02) -----------------------------------
polys = gpd.read_file(config.PAS_SELECTED, engine='pyogrio',
                      columns=['WDPA_PID', 'N_TRNST', 'PCT_KEPT'])
polys = polys[polys['WDPA_PID'].isin(pa['WDPA_PID'])]
assert polys['WDPA_PID'].is_unique and len(polys) == len(pa), "analysis PAs missing from pas_selected.gpkg"
polys = polys.set_index('WDPA_PID').loc[pa['WDPA_PID']]

# --- Which point asset holds each PA, and its transects there ------------------
zips = sorted(config.UPLOAD_DIR.glob(f'{config.UPLOAD_PREFIX}*.zip'))
assert zips, f"no {config.UPLOAD_PREFIX}*.zip in {config.UPLOAD_DIR}"
asset_of, n_in_asset = {}, {}
for z in zips:
    nnn = int(z.stem[len(config.UPLOAD_PREFIX):])
    pts = pyogrio.read_dataframe(f'/vsizip/{z}/{z.stem}.shp', columns=['WDPA_PID', 'transectID'],
                                 read_geometry=False)
    for pid, n in pts.groupby('WDPA_PID')['transectID'].nunique().items():
        assert pid not in asset_of, f"PA {pid} is in two point assets"
        asset_of[pid], n_in_asset[pid] = nnn, n
print(f"Point assets: {len(zips)} ({zips[0].name} … {zips[-1].name}), {len(asset_of):,} PAs")
missing = sorted(set(pa['WDPA_PID']) - set(asset_of))
assert not missing, f"{len(missing)} analysis PAs are in no point asset, e.g. {missing[:5]}"
n_pts = pa['WDPA_PID'].map(n_in_asset)
assert (n_pts.to_numpy() == polys['N_TRNST'].to_numpy()).all(), "point assets and N_TRNST disagree"

# --- Missing data (as 05's missing_data_summary) --------------------------------
mt = pd.read_csv(config.MISSING_TRANSECTS, dtype={'WDPA_PID': str}, usecols=['WDPA_PID', 'n_missing_years'])
miss = mt.groupby('WDPA_PID')['n_missing_years'].sum().reindex(pa['WDPA_PID'], fill_value=0).to_numpy()
n_trnst = polys['N_TRNST'].to_numpy()
md = json.loads(counts_path.read_text())['missing_data']
assert int(n_trnst.sum()) == md['transects'], "transect total differs from analysis_counts.json"
assert int(miss.sum()) == md['missing_transect_years'], "missing transect-years differ from analysis_counts.json"
assert int((miss > 0).sum()) == md['pas_affected'], "PAs with missing data differ from analysis_counts.json"
print(f"Missing data: {int(miss.sum()):,} transect-years in {int((miss > 0).sum())} PAs "
      f"(matches analysis_counts.json)")

# --- Polygon table --------------------------------------------------------------
out = gpd.GeoDataFrame({
    'ORDER': pa['ORDER'].astype('int32').to_numpy(),
    'WDPA_PID': pa['WDPA_PID'].to_numpy(),
    'NAME': pa['NAME'].to_numpy(),
    'ISO3': pa['ISO3'].to_numpy(),
    'BIOME': pa['BIOME_NAME'].to_numpy(),
    'AREA_KM2': pa['AREA_KM2'].round(1).to_numpy(),
    'N_TRNST': n_trnst.astype('int32'),
    'PCT_KEPT': polys['PCT_KEPT'].round(2).to_numpy(),
    'MISS_PCT': (100 * miss / (n_trnst * n_years)).round(2),
    'COMPL_PCT': pa['pct_complete'].round(2).to_numpy(),
    'WATER_PCT': pa['water_extent_pct'].round(2).to_numpy(),
    'ASSET': pa['WDPA_PID'].map(asset_of).astype('int32').to_numpy(),
}, geometry=polys.geometry.values, crs=polys.crs)
assert out.crs.to_epsg() == 4326

config.INSPECTION_DIR.mkdir(parents=True, exist_ok=True)
zip_path = config.INSPECTION_ZIP
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    # RESIZE=YES shrinks the text field widths in the .dbf to the longest value
    out.to_file(tmp / f'{zip_path.stem}.shp', engine='pyogrio', layer_options={'RESIZE': 'YES'})
    back = pyogrio.read_dataframe(tmp / f'{zip_path.stem}.shp', read_geometry=False)
    assert back['NAME'].tolist() == out['NAME'].tolist(), "names changed in the shapefile"
    assert back['ORDER'].tolist() == out['ORDER'].tolist()
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(tmp.iterdir()):
            z.write(f, arcname=f.name)
print(f"\nWrote {zip_path.relative_to(project_root)} ({zip_path.stat().st_size / 1e6:.0f} MB, {len(out):,} PAs)")

# --- Checklist ------------------------------------------------------------------
cl = pd.DataFrame({'order': pa['ORDER'], 'WDPA_PID': pa['WDPA_PID'], 'NAME': pa['NAME'],
                   'ISO3': pa['ISO3'], 'BIOME_NAME': pa['BIOME_NAME'], 'decision': '', 'note': ''})
cl.to_csv(checklist, index=False, encoding='utf-8-sig')
print(f"Wrote {checklist.relative_to(project_root)} ({len(cl):,} rows; decision: blank = keep, "
      f"{', '.join(repr(d) for d in config.INSPECTION_DECISIONS if d)})")

print("\nReview order (biome, country, name):")
for biome, g in pa.groupby('BIOME_NAME', sort=False):
    print(f"  {g['ORDER'].min():>5}–{g['ORDER'].max():<5} {biome} ({len(g):,})")

asset_id = f"projects/{config.GEE_PROJECT}/assets/{zip_path.stem}"
print("\nMANUAL STEP: upload the zip to Earth Engine as a table asset:")
print(f"  {zip_path.relative_to(project_root)}  ->  {asset_id}")
print("  Then paste scripts/05b_inspection_app.js into the Code Editor and run it.")
print(f"\nDone in {time.time() - t_start:.0f} s")
