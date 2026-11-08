"""
Configuration file for the Protected Area Edge Analysis workflow (NDVI).

This module centralizes all configurable parameters:
- Time period and NDVI source
- Protected area filtering and transect geometry parameters
- Folder layout and file paths
- Google Earth Engine settings
"""

import json
import os
from pathlib import Path

# =====================================================================
# Project Paths
# =====================================================================
# Each data folder is named after the script that fills it.

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"
WDPA_DIR = DATA_DIR / "01_wdpa"                     # script 01 export (download from Drive)
PAS_DIR = DATA_DIR / "02_pas"                       # script 02 Part 1: selected / removed PAs
TRANSECTS_DIR = DATA_DIR / "02_transects"           # script 02 Part 2: transect points
UPLOAD_DIR = TRANSECTS_DIR / "upload"               # zipped shapefiles to upload to Earth Engine
GEE_CSV_DIR = DATA_DIR / "03_gee"                   # script 03 CSVs (download from Drive)
TRANSECT_YEARS_DIR = DATA_DIR / "04_transect_years" # script 04: one row per transect-year

RESULTS_DIR = PROJECT_ROOT / "results"              # final outputs only

WDPA_POLYGONS = WDPA_DIR / "WDPA_polygons.geojson"
PAS_SELECTED = PAS_DIR / "pas_selected.gpkg"
PAS_REMOVED = PAS_DIR / "pas_removed.gpkg"
TRANSECT_POINTS = TRANSECTS_DIR / "transect_points.parquet"
PA_METRICS = RESULTS_DIR / "pa_edge_metrics_ndvi.csv"    # FINAL: one row per PA-year
PIPELINE_COUNTS = RESULTS_DIR / "pipeline_counts.json"

# =====================================================================
# Run tag
# =====================================================================
# Used only where old and new outputs could mix: Earth Engine asset names and the
# Google Drive export folder. Change it for a new full re-run.
RUN_TAG = 'v2'

# =====================================================================
# Google Earth Engine Project - REQUIRED for scripts 01 and 03
# =====================================================================

# Each person sets their project once, outside the repo, so this file never needs
# editing. In order of precedence:
#   1. the GEE_PROJECT environment variable (one-off override)
#   2. the default saved by `earthengine set_project MY-PROJECT`, which is stored
#      next to the Earth Engine credentials and survives container rebuilds
# Find your project ID at: https://console.cloud.google.com/
EE_CREDENTIALS = Path.home() / ".config" / "earthengine" / "credentials"


def _saved_gee_project():
    """Project saved by `earthengine set_project`, or None."""
    try:
        return json.loads(EE_CREDENTIALS.read_text()).get("project")
    except (OSError, ValueError):
        return None


GEE_PROJECT = os.environ.get("GEE_PROJECT") or _saved_gee_project() or 'YOUR-GEE-PROJECT'

# =====================================================================
# Dataset Parameters
# =====================================================================

# World Database on Protected Areas collection year - month
WDPA_COLLECTION = "WCMC/WDPA/202508/polygons"
START_YEAR = 2003
END_YEAR = 2025

# NDVI source (MODIS Terra Vegetation Indices 16-Day Global 500 m)
INDEX_NAME = 'ndvi'  # used in file and export names
NDVI_COLLECTION = 'MODIS/061/MOD13A1'
NDVI_BAND = 'NDVI'
NDVI_SCALE_FACTOR = 0.0001

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
# Coordinate Reference Systems
# =====================================================================

# Equal-area projection (Mollweide) for area and overlap calculations in script 02
# Part 1. Transects are built in a per-PA azimuthal equidistant projection.
PROCESSING_CRS = 'ESRI:54009'

# Storage/GEE CRS (geographic coordinates)
STORAGE_CRS = 'EPSG:4326'  # WGS84

# =====================================================================
# Transect Geometry Parameters
# =====================================================================

# Distance between sample points along protected area boundary (meters)
BOUNDARY_SPACING = 500

# Distance between points along each transect (meters)
TRANSECT_SPACING = 2500

# Number of points on each side of the boundary (not including boundary point)
# Total points per transect = 2 * POINTS_PER_SIDE + 1
POINTS_PER_SIDE = 2

# Inner buffer distance (meters). PAs with no interior at least this far from
# their edge (i.e., narrower than ~11 km everywhere) are removed.
BUFFER_DIST = TRANSECT_SPACING * POINTS_PER_SIDE + 500  # 500m beyond last point

# Maximum hole area to fill in polygons (square meters)
# Holes smaller than this will be filled, larger ones preserved
MAX_HOLE_AREA = 2_250_000  # 1500m × 1500m

# Coverage limit (percent of the boundary). PAs whose longest run of dropped
# transects (MAX_GAP_PCT) exceeds this are removed, so no PA has a large part of
# its boundary unsampled.
MAX_GAP_PCT_LIMIT = 25

# Overlap threshold (percent). Two PAs overlapping by more than this share of
# BOTH polygons are duplicates (oldest kept); by more than this share of the
# smaller polygon only, the smaller is nested in the larger (larger kept).
OVERLAP_THRESHOLD = 90

# =====================================================================
# Chunking Parameters
# =====================================================================

# Protected areas per Earth Engine upload file (whole PAs only)
CHUNK_SIZE = 400  # ~1 million points per file

# Points per exported CSV part in script 03
GEE_SUBCHUNK_SIZE = 50_000

# Number of export tasks run at once in script 03
GEE_BATCH_SIZE = 10

# =====================================================================
# Google Earth Engine Settings
# =====================================================================

# Upload files written by script 02 ({UPLOAD_PREFIX}000.zip, ...) and the Earth
# Engine table assets they are uploaded as ({GEE_ASSET_PREFIX}000, ...)
UPLOAD_PREFIX = f'transect_points_{RUN_TAG}_'
GEE_ASSET_PREFIX = f'projects/{GEE_PROJECT}/assets/{UPLOAD_PREFIX}'

# Google Drive folder that script 03 exports CSVs to
GEE_DRIVE_FOLDER = f'{INDEX_NAME}_gradient_{RUN_TAG}'

def get_num_chunks():
    """
    Count the upload zips written by script 02 (one Earth Engine asset each).

    Counted when called (not on import), so the result reflects the files
    on disk at that moment rather than when config was first imported.
    """
    return len(list(UPLOAD_DIR.glob(f'{UPLOAD_PREFIX}*.zip')))

# =====================================================================
# Validation
# =====================================================================

def validate_config():
    """Validate configuration settings (runs on import; no Earth Engine checks)."""
    if START_YEAR > END_YEAR:
        raise ValueError(f"START_YEAR ({START_YEAR}) must be <= END_YEAR ({END_YEAR})")

    if POINTS_PER_SIDE < 1:
        raise ValueError("POINTS_PER_SIDE must be >= 1")

    if CHUNK_SIZE < 1:
        raise ValueError("CHUNK_SIZE must be >= 1")


def require_gee_project():
    """Raise if GEE_PROJECT is still the placeholder. Called by scripts 01 and 03."""
    if GEE_PROJECT == 'YOUR-GEE-PROJECT':
        raise ValueError(
            "GEE_PROJECT has not been set. Run once in the container terminal:\n"
            "    earthengine authenticate --auth_mode=notebook\n"
            "    earthengine set_project YOUR-PROJECT-ID\n"
            "or set the GEE_PROJECT environment variable."
        )

# Run validation on import
validate_config()
