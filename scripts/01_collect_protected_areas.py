"""
Script 01: Collect Protected Areas

Downloads and filters protected area polygons from Google Earth Engine.

Workflow:
1. Filter the WDPA release (modules/remotesensing.py, pa_filter_steps): terrestrial
   (MARINE == "0"), not a Marine Protected Area or a UNESCO-MAB Biosphere Reserve, status
   Designated / Established / Inscribed, GIS_AREA >= 200 km², WDPA_PID not in
   EXCLUDED_PIDS, single-part polygon; the PAs removed by each filter are counted in
   results/pipeline_counts.json (section '01_wdpa')
2. Add the RESOLVE Ecoregions 2017 biome with the largest overlap
3. Export to Google Drive

MANUAL STEP REQUIRED AFTER RUNNING:
Download WDPA_polygons.geojson from Google Drive to data/01_wdpa/

Then run: python scripts/02_generate_sampling_points.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import ee
from modules import config
from modules.counts import update_counts
from modules.remotesensing import get_pa_filter, pa_filter_steps, set_geometry_type, get_biome

print("="*80)
print("SCRIPT 01: Collect Protected Areas")
print("="*80)

# Initialize Earth Engine
print("\nInitializing Google Earth Engine...")
config.require_gee_project()
ee.Authenticate(auth_mode='notebook')  # prompts only if not signed in yet
ee.Initialize(project=config.GEE_PROJECT)

# Load WDPA dataset (a pinned monthly release, not WCMC/WDPA/current)
release = config.WDPA_COLLECTION.split("/")[-2]
print(f"Loading WDPA release {release} ({config.WDPA_COLLECTION})...")
try:
    ee.data.getAsset(config.WDPA_COLLECTION)
except ee.EEException as e:
    raise SystemExit(
        f"WDPA release {release} is not available in Earth Engine ({e}).\n"
        f"If you have this release's WDPA_polygons.geojson from an earlier run of this script, "
        f"put it in {config.WDPA_DIR} and continue with script 02, which applies "
        f"EXCLUDED_PIDS again."
    )
wdpa = ee.FeatureCollection(config.WDPA_COLLECTION)

# Add geometry type property
wdpa = wdpa.map(set_geometry_type)

# Apply filters
print("Applying filters...")
print(f"  - Minimum area: {config.MIN_AREA_KM2} km²")
print(f"  - Status: {', '.join(config.VALID_STATUS)}")
print(f"  - Excluding {len(config.EXCLUDED_PIDS)} problematic PIDs")

pa_filter = get_pa_filter(type="Polygon")
wdpa_filtered = wdpa.filter(pa_filter)

# PAs left after each filter, in order (for the methods' counts; about 2 min). The geometry
# type is computed only for the PAs left by the attribute filters (single_part is last).
print("Counting the PAs removed by each filter...")
try:
    fc = ee.FeatureCollection(config.WDPA_COLLECTION)
    before = fc.size().getInfo()
    counts = {'collection': config.WDPA_COLLECTION, 'features_in_release': before, 'steps': []}
    print(f"  {'release':<28}{before:>9,}")
    for name, f in pa_filter_steps(type="Polygon"):
        if name == "single_part":
            fc = fc.map(set_geometry_type)
        fc = fc.filter(f)
        n = fc.size().getInfo()
        counts['steps'].append({'step': name, 'removed': before - n, 'n_out': n})
        print(f"  {name:<28}{n:>9,}  (-{before - n:,})")
        before = n
    counts['exported'] = before
    update_counts(config.PIPELINE_COUNTS, '01_wdpa', counts)
except ee.EEException as e:
    print(f"  WARNING: counts not computed ({e}); the export below is unaffected.")

# Add biome information
print("Adding biome information from RESOLVE Ecoregions 2017...")
wdpa_with_biome = wdpa_filtered.map(get_biome)

# Count filtered polygons
count = wdpa_with_biome.size().getInfo()
print(f"\nFiltered to {count:,} protected area polygons")

# Export to Google Drive
print("\nExporting to Google Drive...")
print("  File: WDPA_polygons.geojson")

task = ee.batch.Export.table.toDrive(
    collection=wdpa_with_biome,
    description='WDPA_polygons',
    fileFormat='GeoJSON'
)

task.start()
print(f"\nExport task started: {task.id}")
print("Monitor progress at: https://code.earthengine.google.com/tasks")

config.WDPA_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("MANUAL STEP REQUIRED:")
print("="*80)
print("1. Wait for the export task to complete (check GEE Tasks) ~2 hours")
print("2. Download 'WDPA_polygons.geojson' from Google Drive")
print(f"3. Place it in: {config.WDPA_POLYGONS}")
print("4. Then run: python scripts/02_generate_sampling_points.py")
print("="*80)
