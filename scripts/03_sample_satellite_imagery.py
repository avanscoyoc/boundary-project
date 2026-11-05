"""
Script 03: Sample Satellite Imagery

Samples NDVI gradient magnitude (2003-2025) and static layers at every transect
point using Google Earth Engine.

Workflow:
1. Load the transect point assets uploaded after script 02
2. Sample static layers (water, human modification, elevation and slope from
   the Copernicus GLO-30 DEM, 2024_1 release)
3. Compute the NDVI gradient magnitude of each annual median composite, on the
   native MODIS grid (sinusoidal, ~463 m), and sample at the same projection
4. Export results to Google Drive in parts of 50,000 points

BEFORE RUNNING:
1. Set GEE_PROJECT in modules/config.py
2. Upload every data/02_transects/upload/transect_points_{RUN_TAG}_NNN.zip as a
   table asset (see output from script 02)

TEST FIRST:
    python scripts/03_sample_satellite_imagery.py --test
exports only part 0 of asset 000, to check it completes before the full run.

RESUME:
    python scripts/03_sample_satellite_imagery.py --start 5
starts at asset 005 (e.g. if a run stopped after asset 004 finished), so assets
that were already exported are not exported again.

MANUAL STEP REQUIRED AFTER RUNNING:
Download all CSV files from Google Drive folder '{INDEX_NAME}_gradient_{RUN_TAG}'
to data/03_gee/

Then run: python scripts/04_compute_edge_metrics.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import ee
from modules import config
from modules.remotesensing import (
    get_annual_gradient_magnitude, get_ndvi_projection, batch_sample_assets
)

TEST_RUN = '--test' in sys.argv
START_ASSET = int(sys.argv[sys.argv.index('--start') + 1]) if '--start' in sys.argv else 0

print("=" * 80)
print("SCRIPT 03: Sample Satellite Imagery" + ("  [TEST: asset 000, part 0 only]" if TEST_RUN else ""))
print("=" * 80)

print(f"\nIndex: NDVI gradient magnitude ({config.NDVI_COLLECTION})")
print(f"Years: {config.START_YEAR}-{config.END_YEAR}")

# Initialize Earth Engine
print("\nInitializing Google Earth Engine...")
config.require_gee_project()
ee.Authenticate()
ee.Initialize(project=config.GEE_PROJECT)

# Export folder and names
folder_name = config.GEE_DRIVE_FOLDER
export_prefix = f"{config.INDEX_NAME}_gradient_{config.RUN_TAG}_"
print(f"Export folder: {folder_name}")

# Sampling projection: native MOD13A1 grid
ndvi_proj = get_ndvi_projection()
ndvi_scale = ndvi_proj.nominalScale().getInfo()
print(f"Sampling on the native MODIS grid: {ndvi_scale:.1f} m")

# Static layers
gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
hm = ee.ImageCollection('CSP/HM/GlobalHumanModification').mean()
# Copernicus DEM: tag the mosaic with the tiles' native projection. Without it,
# ee.Terrain.slope works on the mosaic's default 1-degree grid and returns
# near-zero or missing slopes (Earth Engine catalog: COPERNICUS/DEM/GLO30_2024_1).
dem = ee.ImageCollection('COPERNICUS/DEM/GLO30_2024_1').select('DEM')
elevation = dem.mosaic().setDefaultProjection(dem.first().projection()).rename('elevation')
slope = ee.Terrain.slope(elevation)

staticImage = ee.Image.cat([
    gsw.select('max_extent'),
    hm.rename('gHM'),
    elevation,
    slope
])

# Year configuration
years = ee.List.sequence(config.START_YEAR, config.END_YEAR)
gradBandNames = [str(y) for y in range(config.START_YEAR, config.END_YEAR + 1)]
selectors = ['WDPA_PID', 'transectID', 'pointID', 'max_extent', 'gHM', 'elevation', 'slope'] + gradBandNames

# Annual gradient band layers
gradientBands = ee.ImageCollection.fromImages(
    years.map(get_annual_gradient_magnitude)
).toBands()
gradientBands = gradientBands.rename(gradBandNames)
image = staticImage.addBands(gradientBands)

# Process assets sequentially (~5 hours for the full run)
total_assets = config.get_num_chunks()
if total_assets == 0:
    raise FileNotFoundError(
        f"No {config.UPLOAD_PREFIX}*.zip files found in {config.UPLOAD_DIR}. "
        "The number of assets to sample is counted from these files; run script 02 first."
    )
if not 0 <= START_ASSET < total_assets:
    raise ValueError(f"--start {START_ASSET} is outside 0-{total_assets - 1}")
asset_indices = [0] if TEST_RUN else range(START_ASSET, total_assets)
print(f"Assets to process: {len(asset_indices)} of {total_assets}"
      + (f" (starting at {START_ASSET:03d})" if START_ASSET and not TEST_RUN else ""))
print("Monitor at: https://code.earthengine.google.com/tasks")

for idx in asset_indices:
    asset = f'{config.GEE_ASSET_PREFIX}{idx:03d}'
    print(f"\nProcessing asset {idx + 1} of {total_assets}: {asset}")
    batch_sample_assets(
        asset, image, selectors, folder_name, export_prefix,
        chunk_size=config.GEE_SUBCHUNK_SIZE, batch_size=config.GEE_BATCH_SIZE,
        chunks_to_run=[0] if TEST_RUN else None,
        scale=ndvi_scale, crs=ndvi_proj,
    )
    print(f"Asset {idx + 1} of {total_assets} complete")

print("\n" + "=" * 80)
if TEST_RUN:
    print("TEST COMPLETE: check the task finished in the Tasks tab, then run without --test.")
    print(f"(Delete the test CSV {export_prefix}000_part_0.csv from Drive or keep it; the")
    print(" full run re-exports it, so download only one copy.)")
else:
    print("MANUAL STEP REQUIRED:")
    print("=" * 80)
    print(f"1. Download all CSV files from Google Drive folder: {folder_name}")
    print(f"2. Place them in: {config.GEE_CSV_DIR}/")
    print("3. Then run: python scripts/04_compute_edge_metrics.py")
print("=" * 80)
