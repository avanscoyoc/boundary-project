"""
Script 12: Validation, part 2 - sample the displaced transects.

Samples the same layers as script 03 (NDVI gradient magnitude 2001-2025 on the native MODIS
grid, JRC max_extent, gHM, elevation, slope), plus RESOLVE Ecoregions 2017 BIOME_NUM as a
diagnostic, at every candidate point from script 11. Each candidate's moved PA polygon gets
its biome as in script 01 (get_biome: the ecoregion with the largest area of the polygon).

Run:
    python scripts/12_validation_sample_imagery.py           # full run
    python scripts/12_validation_sample_imagery.py --test    # test exports only
--test exports part 0 of point asset 000 and the biome of the first 20 polygons of polygon
asset 000 (as displaced_polygons_biome_000_test, which script 13 ignores).

Before running: run script 11, then upload every data/11_validation/upload/
displaced_points_NNN.zip and displaced_polygons_NNN.zip as an Earth Engine table asset
of the same name in your project.

After running: download the Google Drive folder '12_gee' and put its CSVs (point parts and
displaced_polygons_biome_NNN.csv) in data/12_gee/, then run script 13.
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import ee
from modules import config
from modules.remotesensing import (sampling_image, get_ndvi_projection, batch_sample_assets,
                                   get_biome, _state)

TEST_RUN = '--test' in sys.argv

print("=" * 80)
print("SCRIPT 12: Validation - sample the displaced transects" + ("  [TEST: asset 000, part 0]" if TEST_RUN else ""))
print("=" * 80)

config.require_gee_project()
ee.Authenticate(auth_mode='notebook')  # prompts only if not signed in yet
ee.Initialize(project=config.GEE_PROJECT)

folder_name = config.VALIDATION_DRIVE_FOLDER
export_prefix = config.VALIDATION_EXPORT_PREFIX
ndvi_proj = get_ndvi_projection()
ndvi_scale = ndvi_proj.nominalScale().getInfo()

# The same image as script 03, plus the biome number
image, selectors = sampling_image(with_biome=True)

n_assets = len(list(config.VALIDATION_UPLOAD_DIR.glob(f'{config.VALIDATION_UPLOAD_PREFIX}*.zip')))
n_poly = len(list(config.VALIDATION_UPLOAD_DIR.glob(f'{config.VALIDATION_POLY_UPLOAD_PREFIX}*.zip')))
if n_assets == 0 or n_poly == 0:
    raise FileNotFoundError(f"No point or polygon upload files in {config.VALIDATION_UPLOAD_DIR}; run script 11 first.")
print(f"Export folder: {folder_name}; point assets: {n_assets}; polygon assets: {n_poly}; "
      f"MODIS grid {ndvi_scale:.1f} m")

# Biome of each displaced PA polygon, as in script 01 (small; started first so it runs alongside)
poly_tasks = []
for idx in ([0] if TEST_RUN else range(n_poly)):
    fc = ee.FeatureCollection(f'{config.VALIDATION_POLY_ASSET_PREFIX}{idx:03d}')
    if TEST_RUN:
        fc = fc.limit(20)
    name = f'{config.VALIDATION_BIOME_EXPORT}_{idx:03d}' + ('_test' if TEST_RUN else '')
    task = ee.batch.Export.table.toDrive(collection=fc.map(get_biome), description=name, fileFormat='CSV',
                                         selectors=['WDPA_PID', 'BIOME_NAME'], folder=folder_name)
    task.start()
    poly_tasks.append((name, task))
    print(f"Started polygon biome export {name}")

# Every candidate point, as script 03
for idx in ([0] if TEST_RUN else range(n_assets)):
    asset = f'{config.VALIDATION_ASSET_PREFIX}{idx:03d}'
    print(f"\nAsset {idx + 1} of {n_assets}: {asset}")
    batch_sample_assets(asset, image, selectors, folder_name, export_prefix,
                        chunk_size=config.GEE_SUBCHUNK_SIZE, batch_size=config.GEE_BATCH_SIZE,
                        chunks_to_run=[0] if TEST_RUN else None, scale=ndvi_scale, crs=ndvi_proj)
while True:
    states = {name: _state(t) for name, t in poly_tasks}
    if all(v in ('COMPLETED', 'FAILED', 'CANCELLED') for v in states.values()):
        break
    time.sleep(30)
print("\nPolygon biome exports: " + ", ".join(f"{k} {v}" for k, v in states.items()))
if any(v != 'COMPLETED' for v in states.values()):
    print("  WARNING: rerun the failed polygon export(s) before script 13.")
print("\nDone. Download the CSVs to data/12_gee/ and run script 13.")
