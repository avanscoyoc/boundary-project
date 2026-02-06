"""
Configuration file for Protected Area Edge Analysis workflow.

This module centralizes all configurable parameters including:
- Index selection (NDVI, LAI, FPAR, NDBI)
- Time period
- Transect geometry parameters
- File paths
- Google Earth Engine settings

To run the workflow for different indices, edit INDEX_NAME below and re-run
scripts 03-05.
"""

import os
from pathlib import Path

# =====================================================================
# Project Paths
# =====================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTERMEDIATE = DATA_DIR / "intermediate"
DATA_PROCESSED = DATA_DIR / "processed"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_RAW = RESULTS_DIR / "raw"
RESULTS_FIGURES = RESULTS_DIR / "figures"

# =====================================================================
# Index Selection - EDIT THIS TO CHANGE WHICH INDEX TO PROCESS
# =====================================================================

INDEX_NAME = 'ndvi'  # Options: 'ndvi', 'lai', 'fpar', 'ndbi'

# =====================================================================
# Temporal Parameters
# =====================================================================

# Year range for analysis
START_YEAR = 2003
END_YEAR = 2025
YEARS = list(range(START_YEAR, END_YEAR + 1))

# =====================================================================
# Protected Area Filtering Parameters
# =====================================================================

# Minimum protected area size (km²)
MIN_AREA_KM2 = 200

# Protected area status to include
VALID_STATUS = ["Designated", "Established", "Inscribed"]

# Problematic WDPA_PIDs to exclude
EXCLUDED_PIDS = [
    "555655917", "555656005", "555656013", "555665477", "555656021",
    "555665485", "555556142", "187", "555703455", "555563456", "15894"
]

# =====================================================================
# Transect Geometry Parameters
# =====================================================================

# Distance between sample points along protected area boundary (meters)
BOUNDARY_SPACING = 500

# Distance between points along each transect (meters)
TRANSECT_SPACING = 2500

# Number of points on each side of the boundary (not including boundary point)
# Total points per transect = 2 * POINTS_PER_TRANSECT + 1
POINTS_PER_TRANSECT = 2

# Inner buffer distance for filtering bad transects (meters)
# Points inside this buffer are considered problematic
INNER_BUFFER = 5500

# Maximum hole area to fill in polygons (square meters)
# Holes smaller than this will be filled, larger ones preserved
MAX_HOLE_AREA = 2_250_000  # 1500m × 1500m

# Overlap threshold for deduplication (percent)
# Polygons with >90% overlap are considered duplicates
OVERLAP_THRESHOLD = 90

# =====================================================================
# Chunking Parameters
# =====================================================================

# Number of protected areas per chunk file
# Controls memory usage and file sizes
CHUNK_SIZE = 400  # ~1 million points per chunk

# Number of points per sub-chunk for GEE processing
GEE_SUBCHUNK_SIZE = 50_000

# Number of chunks to process in parallel in GEE
GEE_BATCH_SIZE = 10

# =====================================================================
# Google Earth Engine Settings
# =====================================================================

# GEE project ID - UPDATE THIS to your GEE project ID
GEE_PROJECT = 'dse-staff'

# GEE asset paths for transect chunks
GEE_ASSET_PREFIX = f'projects/{GEE_PROJECT}/assets/chunk_'

# Number of chunk files (0-9 = 10 chunks)
NUM_CHUNKS = 10

# =====================================================================
# Coordinate Reference Systems
# =====================================================================

# Processing CRS (equal-area projection for accurate distances)
PROCESSING_CRS = 'ESRI:54009'  # Mollweide

# Storage/GEE CRS (geographic coordinates)
STORAGE_CRS = 'EPSG:4326'  # WGS84

# =====================================================================
# Remote Sensing Index Configurations
# =====================================================================

INDEX_CONFIGS = {
    'ndvi': {
        'collection': 'MODIS/061/MOD13A1',
        'band': 'NDVI',
        'scale_factor': 0.0001,
        'description': 'Normalized Difference Vegetation Index'
    },
    'ndbi': {
        'collection': 'MODIS/061/MOD09GA',
        'scale_factor': 0.0001,  # Applied to bands before index calculation
        'description': 'Normalized Difference Built-up Index'
    },
    'lai': {
        'collection': 'MODIS/061/MCD15A3H',
        'band': 'Lai',
        'scale_factor': 0.1,
        'description': 'Leaf Area Index'
    },
    'fpar': {
        'collection': 'MODIS/061/MCD15A3H',
        'band': 'Fpar',
        'scale_factor': 0.01,
        'description': 'Fraction of Photosynthetically Active Radiation'
    }
}

# =====================================================================
# Validation
# =====================================================================

def validate_config():
    """Validate configuration settings."""
    if INDEX_NAME not in INDEX_CONFIGS:
        raise ValueError(
            f"Invalid INDEX_NAME: {INDEX_NAME}. "
            f"Must be one of: {list(INDEX_CONFIGS.keys())}"
        )
    
    if START_YEAR > END_YEAR:
        raise ValueError(f"START_YEAR ({START_YEAR}) must be <= END_YEAR ({END_YEAR})")
    
    if POINTS_PER_TRANSECT < 1:
        raise ValueError(f"POINTS_PER_TRANSECT must be >= 1")
    
    if CHUNK_SIZE < 1:
        raise ValueError(f"CHUNK_SIZE must be >= 1")

# Run validation on import
validate_config()
