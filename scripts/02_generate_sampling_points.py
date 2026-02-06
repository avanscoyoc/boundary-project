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

import geopandas as gpd
import pandas as pd
from pathlib import Path
from modules import config
from modules.preprocessing import (
    fill_holes, find_overlap_groups, get_min_year_from_group,
    remove_pa_transects_in_chunks, combine_chunks_to_files
)

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
wdpa = gpd.read_file(input_path)
print(f"Loaded {len(wdpa):,} polygons")

# Reproject to equal-area for accurate area calculations
print(f"\nReprojecting to {config.PROCESSING_CRS}...")
wdpa = wdpa.to_crs(config.PROCESSING_CRS)

# Step 1: Fill small holes
print(f"\nStep 1: Filling holes < {config.MAX_HOLE_AREA / 1_000_000:.2f} km²...")
wdpa = fill_holes(wdpa, max_hole_area=config.MAX_HOLE_AREA)

# Step 2: Remove overlapping geometries
print(f"\nStep 2: Removing overlaps > {config.OVERLAP_THRESHOLD}%...")
overlap_groups = find_overlap_groups(wdpa, overlap_threshold=config.OVERLAP_THRESHOLD)

# Keep only oldest from each overlap group
keep_indices = []
for group in overlap_groups:
    if len(group) > 1:
        # Multiple polygons overlap - keep oldest
        group_df = wdpa.loc[group]
        oldest = get_min_year_from_group(group_df)
        keep_indices.append(oldest.name)
    else:
        # No overlap - keep
        keep_indices.append(group[0])

wdpa = wdpa.loc[keep_indices].copy()
print(f"Removed overlaps, kept {len(wdpa):,} polygons")

# Step 3: Remove duplicate names (dissolve by ORIG_NAME)
print("\nStep 3: Dissolving duplicate names...")
initial_count = len(wdpa)
wdpa = wdpa.dissolve(by='ORIG_NAME', aggfunc='first').reset_index()
print(f"Dissolved {initial_count - len(wdpa)} duplicates, {len(wdpa):,} polygons remaining")

# Step 4: Filter narrow polygons (upper 25% perimeter-to-area ratio)
print("\nStep 4: Filtering narrow polygons...")
wdpa['pa_ratio'] = wdpa.geometry.length / wdpa.geometry.area
threshold = wdpa['pa_ratio'].quantile(0.75)
wdpa = wdpa[wdpa['pa_ratio'] <= threshold].copy()
wdpa = wdpa.drop(columns=['pa_ratio'])
print(f"Filtered narrow polygons, {len(wdpa):,} polygons remaining")

# Save filtered dataset
output_dir = config.DATA_INTERMEDIATE / "wdpa_filtered"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "wdpa_filtered.shp"

print(f"\nSaving filtered dataset to {output_path}...")
wdpa.to_file(output_path)

print("\nFiltering complete: {len(wdpa):,} protected areas")

# =========================================================================
# PART 2: CREATE TRANSECTS
# =========================================================================

print("\n" + "="*80)
print("PART 2: Create Transects")
print("="*80)

# Create inner buffers
print(f"\nCreating inner buffers ({config.INNER_BUFFER}m)...")
wdpa['buffer'] = wdpa.geometry.buffer(-config.INNER_BUFFER)
wdpa_buffer_dict = dict(zip(wdpa['WDPA_PID'], wdpa['buffer']))
print(f"Created {len(wdpa_buffer_dict)} buffers")

# Generate transects with streaming chunk output
transect_output_dir = config.DATA_INTERMEDIATE / "transect_chunks"
print(f"\nGenerating transects...")
print(f"  Boundary spacing: {config.BOUNDARY_SPACING}m")
print(f"  Transect spacing: {config.TRANSECT_SPACING}m")
print(f"  Points per transect: {2 * config.POINTS_PER_TRANSECT + 1}")
print(f"  Chunk size: {config.CHUNK_SIZE} PAs per chunk")
print(f"  Output CRS: {config.STORAGE_CRS} (for GEE compatibility)")

stats = remove_pa_transects_in_chunks(
    wdpa_gdf=wdpa,
    wdpa_buffer_dict=wdpa_buffer_dict,
    sample_dist=config.BOUNDARY_SPACING,
    transect_unit=config.TRANSECT_SPACING,
    transect_pts=config.POINTS_PER_TRANSECT,
    output_dir=str(transect_output_dir),
    chunk_size=config.CHUNK_SIZE,
    output_crs=config.STORAGE_CRS
)

# Print statistics
print("\n" + "="*80)
print("TRANSECT GENERATION STATISTICS")
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

# Combine chunks into CSV files
print("\n" + "="*80)
print("Combining chunks into CSV files...")
print("="*80)

chunk_pattern = str(transect_output_dir / "chunk_*.shp")
transect_csv = "data/processed/transects_final.csv"  # Not used downstream, informational
attributes_output = str(config.DATA_PROCESSED / "attributes_final.csv")

combine_chunks_to_files(
    chunk_pattern=chunk_pattern,
    transect_output=transect_csv,
    attributes_output=attributes_output
)

print("\n" + "="*80)
print("SAMPLING POINTS GENERATED")
print("="*80)
print(f"Chunk shapefiles: {transect_output_dir}/")
print(f"Attributes file: {attributes_output}")

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
