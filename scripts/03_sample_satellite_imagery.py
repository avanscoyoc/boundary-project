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

import ee
import time
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
folder_name = f"{config.INDEX_NAME}_raw"
print(f"Export folder: {folder_name}")

# Load static layers
print("\nLoading static environmental layers...")
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

# Build gradient image
print(f"\nBuilding {config.INDEX_NAME.upper()} gradient image...")
years = ee.List.sequence(config.START_YEAR, config.END_YEAR)
gradBandNames = [str(y) for y in range(config.START_YEAR, config.END_YEAR + 1)]

def make_current_gradient(y):
    return make_gradient(config.INDEX_NAME, y)

gradStack = ee.ImageCollection(years.map(make_current_gradient)).toBands()
gradStack = gradStack.select(
    gradStack.bandNames(),
    ee.List(gradBandNames)
)

# Combine static and gradient layers
sampleImage = staticImage.addBands(gradStack)
selectors = ['max_extent', 'gHM', 'elevation', 'slope'] + gradBandNames

print(f"Sample bands: {len(selectors)}")

# Sampling function
def process_samples(asset_path, chunk_size=config.GEE_SUBCHUNK_SIZE, 
                   batch_size=config.GEE_BATCH_SIZE):
    """Process samples from a GEE asset with batching."""
    samples = ee.FeatureCollection(asset_path)
    size = samples.size().getInfo()
    nChunks = int((size + chunk_size - 1) // chunk_size)
    
    # Extract asset number from path
    asset_num = asset_path.split('_')[-1]
    
    print(f"  Total points: {size:,}")
    print(f"  Sub-chunks: {nChunks} (batch size: {batch_size})")
    
    tasks = []
    for i in range(nChunks):
        fcChunk = ee.FeatureCollection(samples.toList(chunk_size, i * chunk_size))
        
        sampledChunk = sampleImage.sampleRegions(
            collection=fcChunk,
            properties=['WDPA_PID', 'transectID', 'pointID'],
            scale=500,
            tileScale=4
        )
        
        task = ee.batch.Export.table.toDrive(
            collection=sampledChunk,
            description=f'{config.INDEX_NAME}_raw_grad_{asset_num}_chunk_{i}',
            folder=folder_name,
            fileFormat='CSV',
            selectors=selectors
        )
        tasks.append((task, i))
    
    # Start tasks in batches
    for batch_start in range(0, len(tasks), batch_size):
        batch = tasks[batch_start:batch_start + batch_size]
        chunk_nums = [idx for idx, _ in batch]
        print(f"  Processing chunks {chunk_nums}...")
        
        for task, _ in batch:
            task.start()
        
        # Wait for batch to complete
        time.sleep(10)
        all_complete = False
        while not all_complete:
            time.sleep(30)
            statuses = [t.status()['state'] for t, _ in batch]
            if all(s in ['COMPLETED', 'FAILED', 'CANCELLED'] for s in statuses):
                all_complete = True
                print(f"  Completed chunks {chunk_nums}")
    
    return len(tasks)

# Process all chunk assets
print(f"\n{'='*80}")
print(f"Processing {config.NUM_CHUNKS} asset chunks...")
print(f"{'='*80}")

total_exports = 0
for idx in range(config.NUM_CHUNKS):
    asset = f"{config.GEE_ASSET_PREFIX}{idx:03d}"
    print(f"\nProcessing asset {idx+1} of {config.NUM_CHUNKS}: {asset}")
    
    try:
        n_exports = process_samples(asset)
        total_exports += n_exports
        print(f"  Created {n_exports} export tasks")
    except Exception as e:
        print(f"  Error processing asset {asset}: {e}")

print("\n" + "="*80)
print("SAMPLING COMPLETE")
print("="*80)
print(f"Total export tasks: {total_exports}")
print(f"Google Drive folder: {folder_name}")

print("\n" + "="*80)
print("MANUAL STEP REQUIRED:")
print("="*80)
print("1. Wait for all export tasks to complete")
print("   Monitor at: https://code.earthengine.google.com/tasks")
print(f"2. Download all CSV files from Google Drive folder: {folder_name}")
print(f"3. Place them in: {config.RESULTS_RAW / f'{config.INDEX_NAME}_raw'}/")
print("4. Then run: python scripts/04_compute_edge_metrics.py")
print("="*80)
