"""
Script 01: Collect protected areas.

Filters WDPA polygons in Earth Engine, adds the RESOLVE Ecoregions 2017 biome with the
largest overlap, and exports the result to Google Drive. Filters (modules/remotesensing.py):
terrestrial, not a Marine Protected Area or UNESCO-MAB Biosphere Reserve, status
Designated/Established/Inscribed, GIS_AREA >= 200 km², not in EXCLUDED_PIDS, single-part
polygon. PAs removed by each filter are counted in results/pipeline_counts.json ('01_wdpa').

Run: python scripts/01_collect_protected_areas.py

After running: download WDPA_polygons.geojson from Google Drive to data/01_wdpa/,
then run script 02.
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

print("\nInitializing Google Earth Engine...")
config.require_gee_project()
ee.Authenticate(auth_mode='notebook')  # prompts only if not signed in yet
ee.Initialize(project=config.GEE_PROJECT)

# A pinned monthly release, not WCMC/WDPA/current
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

wdpa = wdpa.map(set_geometry_type)

print("Applying filters...")
print(f"  - Minimum area: {config.MIN_AREA_KM2} km²")
print(f"  - Status: {', '.join(config.VALID_STATUS)}")
print(f"  - Excluding {len(config.EXCLUDED_PIDS)} problematic PIDs")

pa_filter = get_pa_filter(type="Polygon")
wdpa_filtered = wdpa.filter(pa_filter)

# PAs left after each filter, for the methods counts (~2 min). Geometry type is computed
# only for PAs that pass the attribute filters (single_part is last).
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

print("Adding biome information from RESOLVE Ecoregions 2017...")
wdpa_with_biome = wdpa_filtered.map(get_biome)

count = wdpa_with_biome.size().getInfo()
print(f"\nFiltered to {count:,} protected area polygons")

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
