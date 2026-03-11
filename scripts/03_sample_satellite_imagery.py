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
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import ee
from modules import config
from modules.remotesensing import make_gradient

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
folder_name = f"{config.INDEX_NAME}_raw_TEST"
print(f"Export folder: {folder_name}")

# Static layers (unchanged)
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
years = ee.List.sequence(2003, 2025)
gradBandNames = [str(y) for y in range(2003, 2026)]
selectors = ['WDPA_PID', 'transectID', 'pointID', 'max_extent', 'gHM', 'elevation', 'slope'] + gradBandNames

# Build gradient function for selected index
def make_current_gradient(y):
    return make_gradient(config.INDEX_NAME, y)


# Build image with gradient bands (your existing logic)
gradientBands = ee.ImageCollection.fromImages(
    years.map(make_current_gradient)
).toBands()
gradientBands = gradientBands.rename(gradBandNames)
image = staticImage.addBands(gradientBands)


def process_samples(asset_path, chunk_size=50_000, batch_size=10, chunks_to_run=None):
    """
    Process samples from Earth Engine asset and export to Google Drive.
    
    Parameters
    ----------
    asset_path : str
        Path to Earth Engine FeatureCollection asset
    chunk_size : int
        Number of samples per chunk
    batch_size : int
        Number of chunks to process simultaneously
    chunks_to_run : list, optional
        Specific chunk indices to process (for rerunning failures)
    """
    samples = ee.FeatureCollection(asset_path)
    size = samples.size().getInfo()
    nChunks = int((size + chunk_size - 1) // chunk_size)
    tasks = []
    
    # If chunks_to_run is None, run all chunks
    if chunks_to_run is None:
        chunks_to_run = list(range(nChunks))
    
    # Extract asset number from path (e.g., "chunk_003" -> "003")
    asset_num = asset_path.split("_")[-1]
    
    # Create tasks only for specified chunks
    for i in chunks_to_run:
        fcChunk = ee.FeatureCollection(samples.toList(chunk_size, i * chunk_size))
        sampled = image.sampleRegions(
            collection=fcChunk,
            properties=['WDPA_PID', 'transectID', 'pointID'],
            scale=500,
            tileScale=4
        )
        task = ee.batch.Export.table.toDrive(
            collection=sampled,
            description=f'{config.INDEX_NAME}_raw_grad_{asset_num}_chunk_{i}',
            fileFormat='CSV',
            selectors=selectors,
            folder=folder_name
        )
        tasks.append((i, task))

    # Process in batches
    for j in range(0, len(tasks), batch_size):
        batch = tasks[j:j + batch_size]
        for idx, t in batch:
            t.start()
        
        chunk_nums = [idx for idx, _ in batch]
        print(f"  Processing chunks {chunk_nums}...")
        
        while True:
            statuses = [t.status()['state'] for _, t in batch]
            if all(s in ['COMPLETED', 'FAILED', 'CANCELLED'] for s in statuses):
                print(f"  Completed chunks {chunk_nums}")
                break
            time.sleep(30)

# Process all assets sequentially #305 minutes
total_assets = 10
for idx in range(total_assets):
    asset = f'projects/dse-staff/assets/chunk_{idx:03d}'
    print(f"\nProcessing asset {idx + 1} of {total_assets}: {asset}")
    process_samples(asset)
    print(f"Asset {idx + 1} of {total_assets} complete")


print("\n" + "="*80)
print(f"SAMPLING COMPLETE for {config.INDEX_NAME.upper()}")
print("="*80)

print("\n" + "="*80)
print("MANUAL STEP REQUIRED:")
print("="*80)
print("1. Wait for all export tasks to complete")
print("   Monitor at: https://code.earthengine.google.com/tasks")
print(f"2. Download all CSV files from Google Drive folder: {folder_name}")
print(f"3. Place them in: {config.RESULTS_RAW / f'{config.INDEX_NAME}_raw'}/")
print("4. Then run: python scripts/04_compute_edge_metrics.py")
print("="*80)
