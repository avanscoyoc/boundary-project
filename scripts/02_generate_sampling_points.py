"""
Script 02: Generate Sampling Points

This script processes downloaded WDPA data and creates transect sampling points:

Part 1 - Filter WDPA Geometries:
1. Fill small holes in polygons
2. Remove overlapping geometries (>90% overlap, keeps oldest)
3. Remove duplicate names (dissolves by ORIG_NAME)
4. Filter narrow polygons (upper 25% perimeter-to-area ratio)

Part 2 - Create Transects:
1. Create inner buffers (5500m) for each PA
2. Generate boundary points every 500m
3. Create perpendicular transects with 5 points (2 inner, 1 boundary, 2 outer)
4. Filter bad transects (inner points in buffer or outside PA)
5. Write chunks of 400 PAs each to shapefiles
6. Combine chunks into transects_final.csv and attributes_final.csv

Input:  data/raw/WDPA_polygons.geojson
Output: data/intermediate/transect_chunks/chunk_*.shp
        data/processed/attributes_final.csv

MANUAL STEP REQUIRED AFTER RUNNING:
Upload chunk_000.shp through chunk_009.shp to Google Earth Engine as assets.

Then run: python scripts/03_sample_satellite_imagery.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import geopandas as gpd
import pandas as pd
from modules import config
from modules.preprocessing import (
    fill_holes, find_overlap_groups, get_min_year_from_group, 
    filter_and_save_removed, remove_pa_transects_in_chunks, 
    combine_chunks_to_files
)
import ee
ee.Authenticate()
ee.Initialize(project=config.GEE_PROJECT)

print("="*80)
print("SCRIPT 02: Generate Sampling Points")
print("="*80)

# =========================================================================
# PART 1: FILTER WDPA GEOMETRIES
# =========================================================================

print("\n" + "="*80)
print("PART 1: Filter WDPA Geometries")
print("="*80)

# Check input file exists
input_path = config.DATA_RAW / "WDPA_polygons.geojson"
if not input_path.exists():
    raise FileNotFoundError(
        f"Input file not found: {input_path}\n"
        f"Please download WDPA_polygons.geojson from Google Drive first.\n"
        f"See output from script 01_collect_protected_areas.py for instructions."
    )

# Load data
print(f"\nLoading WDPA data from {input_path}...")
wdpa = gpd.read_file(input_path).to_crs(config.PROCESSING_CRS)
print(f"Loaded {len(wdpa):,} polygons and reprojected to {config.PROCESSING_CRS} for processing.")

# Create folder for removed geometries from filtering steps
removed_dir = config.DATA_INTERMEDIATE / "removed"
removed_dir.mkdir(parents=True, exist_ok=True)


# Step 1: Fill small holes
print(f"\nStep 1: Filling holes < {config.MAX_HOLE_AREA / 1_000_000:.2f} km²...")
filled = fill_holes(wdpa, max_hole_area=config.MAX_HOLE_AREA)


# Step 2: Remove overlapping geometries
print(f"\nStep 2: Removing overlaps > {config.OVERLAP_THRESHOLD}%... ~15 minutes")
overlap_groups = find_overlap_groups(filled, overlap_threshold=config.OVERLAP_THRESHOLD)
selected_rows = [
    filled.loc[group_indices[0]] if len(group_indices) == 1
    else get_min_year_from_group(filled.loc[group_indices])
    for group_indices in overlap_groups
]
deduped_overlaps = gpd.GeoDataFrame(selected_rows, crs=filled.crs)
deduped_overlaps, n_removed = filter_and_save_removed(
    filled, filled.index.isin(deduped_overlaps.index),
    removed_dir / "removed_overlapped_pas.shp",
    "overlapping geometries"
)


# Step 3: Remove duplicate names
print("\nStep 3: Dissolving duplicate names...")
deduped_names = deduped_overlaps.groupby('ORIG_NAME').apply(
    lambda x: get_min_year_from_group(x)
).reset_index(drop=True)

# Save removed duplicate-name PAs, but do NOT overwrite dissolved
_, _ = filter_and_save_removed(
    deduped_overlaps,
    deduped_overlaps['WDPA_PID'].isin(deduped_names['WDPA_PID']),
    removed_dir / "removed_duplicate_names.shp",
    "duplicate names"
)
# Keep dissolved geometry for Step 4
dissolved = deduped_names.dissolve(by='ORIG_NAME', as_index=False)


# Step 4: Filter narrow polygons
print("\nStep 4: Removing narrow polygons...")
dissolved["AREA_DISSO"] = dissolved.geometry.area
dissolved["PERIMETER"] = dissolved.geometry.length
dissolved["PA_RATIO"] = dissolved["PERIMETER"] / dissolved["AREA_DISSO"]
quantile = dissolved["PA_RATIO"].quantile(config.QUANTILE_THRESHOLD)
print(f"{int(config.QUANTILE_THRESHOLD * 100)} percentile of PA_RATIO: {quantile}")
wdpa_filtered, n_removed = filter_and_save_removed(
    dissolved, dissolved["PA_RATIO"] < quantile,
    removed_dir / "removed_narrow_pas.shp",
    "narrow PAs"
)
wdpa_filtered = wdpa_filtered.set_crs(config.PROCESSING_CRS)

# Save filtered dataset
output_dir = config.DATA_INTERMEDIATE / "wdpa_filtered"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "wdpa_filtered.shp"

print(f"\nSaving filtered wdpa shapefile to {output_path}...")
wdpa_filtered.to_file(output_path)
del wdpa_filtered

# =========================================================================
# PART 2: CREATE TRANSECTS
# =========================================================================

print("\n" + "="*80)
print("PART 2: Create Transects")
print("="*80)

wdpa_filtered = gpd.read_file(config.DATA_INTERMEDIATE / "wdpa_filtered" / "wdpa_filtered.shp")
print(f"Number of filtered protected areas: {len(wdpa_filtered):,}")

# Create all interior buffers at once (takes ~2min)
print("\nCreating inner buffers for all protected areas...")
wdpa_buffers = wdpa_filtered[['WDPA_PID', 'geometry']].copy()
wdpa_buffers['geometry'] = wdpa_buffers.geometry.buffer(-config.BUFFER_DIST)
wdpa_buffer_dict = dict(zip(wdpa_buffers['WDPA_PID'], wdpa_buffers['geometry']))
del wdpa_buffers  # Free memory - only need the dictionary

# Generate transects with streaming chunk output
transect_output_dir = config.DATA_INTERMEDIATE / "transect_chunks"
print(f"\nGenerating transects...")
print(f"  Boundary point spacing: {config.BOUNDARY_SPACING}m")
print(f"  Transect point spacing: {config.TRANSECT_SPACING}m")
print(f"  Points per transect: {2 * config.POINTS_PER_SIDE + 1}")
print(f"  Chunk size: {config.CHUNK_SIZE} PAs per chunk")
print(f"  Output CRS: {config.STORAGE_CRS} (for GEE compatibility)")

stats = remove_pa_transects_in_chunks(
    wdpa_gdf=wdpa_filtered,
    wdpa_buffer_dict=wdpa_buffer_dict,
    sample_dist=config.BOUNDARY_SPACING,
    transect_unit=config.TRANSECT_SPACING,
    transect_pts=config.POINTS_PER_SIDE,
    output_dir=str(transect_output_dir),
    chunk_size=config.CHUNK_SIZE,
    output_crs=config.STORAGE_CRS
)

# Removed PAs without a valid inner buffer
_, n_empty_removed = filter_and_save_removed(
    wdpa_filtered,
    ~wdpa_filtered["WDPA_PID"].astype(str).isin(stats['empty_buffer_pids']),  # keep non-empty buffers
    removed_dir / "removed_empty_buffer_pas.shp",
    "empty-buffer PAs"
)

# Print statistics
print("\n" + "="*80)
print("TRANSECT STATISTICS")
print("="*80)
print(f"Total PAs processed: {stats['total_pas']:,}")
print(f"PAs with valid transects: {stats['pas_processed']:,}")
print(f"Total transects: {stats['total_transects']:,}")
print(f"Total points: {stats['total_points']:,}")
print(f"Chunk files created: {len(stats['chunk_files'])}")
print(f"\nFiltering diagnostics:")
print(f"  Empty buffers: {stats['empty_buffer']}")
print(f"  All transects filtered: {stats['all_filtered']}")
print(f"  Transects removed (in buffer): {stats['bad_inside_buffer']}")
print(f"  Transects removed (outside PA): {stats['bad_outside_pa']}")

# OPTIONAL: Combine chunks into CSV files
print("\n" + "="*80)
print("Combining chunks into CSV files...")
print("="*80)

transect_csv = str(config.DATA_PROCESSED / "transects_final.csv")
attributes_csv = str(config.DATA_PROCESSED / "attributes_final.csv")

combine_chunks_to_files(
    chunk_pattern=str(transect_output_dir / "chunk_*.shp"),
    transect_output=transect_csv,
    attributes_output=attributes_csv
)

print("COMPLETE: Transect generation finished!")

print("\n" + "="*80)
print("MANUAL STEP REQUIRED:")
print("="*80)
print("Upload the following chunk files to Google Earth Engine:")
print(f"  Location: {transect_output_dir}/")
print(f"  Files: chunk_000.shp through chunk_{len(stats['chunk_files'])-1:03d}.shp")
print(f"\nUpload as assets with names:")
for i in range(len(stats['chunk_files'])):
    asset_path = f"{config.GEE_ASSET_PREFIX}{i:03d}"
    print(f"  {asset_path}")
print(f"\nThen run: python scripts/03_sample_satellite_imagery.py")
print("="*80)