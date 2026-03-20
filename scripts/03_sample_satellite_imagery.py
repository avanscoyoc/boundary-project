"""
Script 03: Sample Satellite Imagery

Extracts satellite imagery and environmental variables at transect points using
Google Earth Engine.

Workflow:
1. Load transect chunks from GEE assets
2. Sample static layers (water, human modification, elevation, slope)
3. Compute gradient magnitude for selected index across all years
4. Export results to Google Drive in batches

BEFORE RUNNING:
1. Edit modules/config.py to set INDEX_NAME ('ndvi', 'lai', 'fpar', or 'ndbi')
2. Upload chunk shapefiles to GEE as assets (see output from script 02)

MANUAL STEP REQUIRED AFTER RUNNING:
Download CSV files from Google Drive folder '{INDEX_NAME}_raw' to:
results/raw/{INDEX_NAME}_raw/

Then run: python scripts/04_compute_edge_metrics.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import ee
from modules import config
from modules.remotesensing import get_annual_gradient_magnitude, batch_sample_assets

print("="*80)
print("SCRIPT 03: Sample Satellite Imagery")
print("="*80)

# Validate index selection
print(f"\nCurrent index: {config.INDEX_NAME.upper()}")
print(f"Description: {config.INDEX_CONFIGS[config.INDEX_NAME]['description']}")
print(f"Years: {config.START_YEAR}-{config.END_YEAR}")

# Initialize Earth Engine
print("\nInitializing Google Earth Engine...")
ee.Authenticate()
ee.Initialize(project=config.GEE_PROJECT)

# Configure export folder
folder_name = f"{config.INDEX_NAME}_raw"
print(f"Export folder: {folder_name}")

# Static layers 
gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
hm = ee.ImageCollection('CSP/HM/GlobalHumanModification').mean()
elevation = ee.Image('USGS/SRTMGL1_003').select('elevation')
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

# Process all assets sequentially #305 minutes
print("1. Wait for all export tasks to complete (~5 hours)")
print("Monitor at: https://code.earthengine.google.com/tasks")
total_assets = 11
for idx in range(total_assets):
    asset = f'projects/dse-staff/assets/chunk_{idx:03d}'
    print(f"\nProcessing asset {idx + 1} of {total_assets}: {asset}")
    batch_sample_assets(asset, image, selectors, folder_name, config.INDEX_NAME)
    print(f"Asset {idx + 1} of {total_assets} complete")


print("\n" + "="*80)
print("MANUAL STEP REQUIRED:")
print("="*80)
print(f"1. Download all CSV files from Google Drive folder: {folder_name}")
print(f"2. Place them in: {config.RESULTS_RAW / f'{config.INDEX_NAME}_raw'}/")
print("3. If you wish to calculate a different index (e.g., ndvi, ndbi, lai, fpar)")
print("   change the 'INDEX_NAME' in modules/config.py and rerun this script.")
print("4. Then run: python scripts/04_compute_edge_metrics.py")
print("="*80)
