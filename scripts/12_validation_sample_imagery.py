"""
Script 12: Validation, part 2 - sample the displaced transects (draft for review)

Samples exactly what script 03 samples (NDVI gradient magnitude 2003-2025 on the native
MODIS grid, JRC max_extent, gHM, elevation, terrain slope), plus the RESOLVE Ecoregions 2017
BIOME_NUM, at every candidate point written by script 11. Script 13 uses the biome and the
gradient values to accept one displacement per PA.

BEFORE RUNNING:
1. Run script 11.
2. Upload every data/11_validation/upload/displaced_points_{RUN_TAG}_NNN.zip as an Earth
   Engine table asset named displaced_points_{RUN_TAG}_NNN in your project.

TEST FIRST:
    python scripts/12_validation_sample_imagery.py --test
exports only part 0 of asset 000.

MANUAL STEP AFTER RUNNING:
Download all CSVs from the Google Drive folder '{INDEX_NAME}_gradient_displaced_{RUN_TAG}' to
data/12_gee_validation/, then run script 13.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import ee
from modules import config
from modules.remotesensing import get_annual_gradient_magnitude, get_ndvi_projection, batch_sample_assets

TEST_RUN = '--test' in sys.argv

print("=" * 80)
print("SCRIPT 12: Validation - sample the displaced transects" + ("  [TEST: asset 000, part 0]" if TEST_RUN else ""))
print("=" * 80)

config.require_gee_project()
ee.Authenticate(auth_mode='notebook')  # prompts only if not signed in yet
ee.Initialize(project=config.GEE_PROJECT)

folder_name = config.VALIDATION_DRIVE_FOLDER
export_prefix = f"{config.INDEX_NAME}_gradient_displaced_{config.RUN_TAG}_"
ndvi_proj = get_ndvi_projection()
ndvi_scale = ndvi_proj.nominalScale().getInfo()

# The same static layers and gradient bands as script 03, plus the biome number
gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
hm = ee.ImageCollection('CSP/HM/GlobalHumanModification').mean()
dem = ee.ImageCollection('COPERNICUS/DEM/GLO30_2024_1').select('DEM')
elevation = dem.mosaic().setDefaultProjection(dem.first().projection()).rename('elevation')
slope = ee.Terrain.slope(elevation).rename('terrain_slope')
biome = (ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")
         .reduceToImage(['BIOME_NUM'], ee.Reducer.first()).rename('BIOME_NUM'))
static_image = ee.Image.cat([gsw.select('max_extent'), hm.rename('gHM'), elevation, slope, biome])

years = ee.List.sequence(config.START_YEAR, config.END_YEAR)
grad_names = [str(y) for y in range(config.START_YEAR, config.END_YEAR + 1)]
gradient = ee.ImageCollection.fromImages(years.map(get_annual_gradient_magnitude)).toBands().rename(grad_names)
image = static_image.addBands(gradient)
selectors = ['WDPA_PID', 'transectID', 'pointID', 'max_extent', 'gHM', 'elevation', 'terrain_slope',
             'BIOME_NUM'] + grad_names

n_assets = len(list(config.VALIDATION_UPLOAD_DIR.glob(f'{config.VALIDATION_UPLOAD_PREFIX}*.zip')))
if n_assets == 0:
    raise FileNotFoundError(f"No upload files in {config.VALIDATION_UPLOAD_DIR}; run script 11 first.")
print(f"Export folder: {folder_name}; assets: {n_assets}; MODIS grid {ndvi_scale:.1f} m")
for idx in ([0] if TEST_RUN else range(n_assets)):
    asset = f'{config.VALIDATION_ASSET_PREFIX}{idx:03d}'
    print(f"\nAsset {idx + 1} of {n_assets}: {asset}")
    batch_sample_assets(asset, image, selectors, folder_name, export_prefix,
                        chunk_size=config.GEE_SUBCHUNK_SIZE, batch_size=config.GEE_BATCH_SIZE,
                        chunks_to_run=[0] if TEST_RUN else None, scale=ndvi_scale, crs=ndvi_proj)
print("\nDone. Download the CSVs to data/12_gee_validation/ and run script 13.")
