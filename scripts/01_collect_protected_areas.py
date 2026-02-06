"""
Script 01: Collect Protected Areas

Downloads and filters protected area polygons from Google Earth Engine.

Workflow:
1. Filter WDPA dataset (≥200 km², non-marine, excludes UNESCO reserves)
2. Add biome information from RESOLVE Ecoregions
3. Export to Google Drive

MANUAL STEP REQUIRED AFTER RUNNING:
Download WDPA_polygons.geojson from Google Drive to data/raw/

Then run: python scripts/02_generate_sampling_points.py
"""

import ee
from modules import config
from modules.remotesensing import get_pa_filter, set_geometry_type, get_biome

print("="*80)
print("SCRIPT 01: Collect Protected Areas")
print("="*80)

# Initialize Earth Engine
print("\nInitializing Google Earth Engine...")
ee.Authenticate()
ee.Initialize(project=config.GEE_PROJECT)

# Load WDPA dataset
print("Loading WDPA dataset...")
wdpa = ee.FeatureCollection("WCMC/WDPA/current/polygons")

# Add geometry type property
print("Adding geometry type property...")
wdpa = wdpa.map(set_geometry_type)

# Apply filters
print("Applying filters...")
print(f"  - Minimum area: {config.MIN_AREA_KM2} km²")
print(f"  - Valid status: {', '.join(config.VALID_STATUS)}")
print(f"  - Excluding {len(config.EXCLUDED_PIDS)} problematic PIDs")

pa_filter = get_pa_filter(type="Polygon")
wdpa_filtered = wdpa.filter(pa_filter)

# Add biome information
print("Adding biome information from RESOLVE Ecoregions...")
wdpa_with_biome = wdpa_filtered.map(get_biome)

# Count filtered polygons
count = wdpa_with_biome.size().getInfo()
print(f"\nFiltered to {count:,} protected area polygons")

# Export to Google Drive
print("\nExporting to Google Drive...")
print("  Folder: WDPA_Export")
print("  File: WDPA_polygons")

task = ee.batch.Export.table.toDrive(
    collection=wdpa_with_biome,
    description='WDPA_polygons',
    folder='WDPA_Export',
    fileFormat='GeoJSON'
)

task.start()
print(f"\nExport task started: {task.id}")
print("Monitor progress at: https://code.earthengine.google.com/tasks")

print("\n" + "="*80)
print("MANUAL STEP REQUIRED:")
print("="*80)
print("1. Wait for the export task to complete (check GEE Tasks)")
print("2. Download 'WDPA_polygons.geojson' from Google Drive")
print("3. Place it in: data/raw/WDPA_polygons.geojson")
print("4. Then run: python scripts/02_generate_sampling_points.py")
print("="*80)
