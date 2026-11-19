"""Settings and paths of the Python scripts (01-04, 11-13, the inspection tool)."""

import json
import os
from pathlib import Path

# --- Paths: each data folder is named after the script that fills it -------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
WDPA_DIR = DATA_DIR / "01_wdpa"                      # 01 export (downloaded from Drive)
PAS_DIR = DATA_DIR / "02_pas"                        # 02 part 1: selected / removed PAs
TRANSECTS_DIR = DATA_DIR / "02_transects"            # 02 part 2: transect points
UPLOAD_DIR = TRANSECTS_DIR / "upload"                # zips to upload to Earth Engine
GEE_CSV_DIR = DATA_DIR / "03_gee"                    # 03 CSVs (downloaded from Drive)
TRANSECT_YEARS_DIR = DATA_DIR / "04_transect_years"  # 04: one row per transect-year
MISSING_TRANSECTS = TRANSECT_YEARS_DIR / "missing_transects.csv"
RESULTS_DIR = PROJECT_ROOT / "results"               # final outputs only
BIOME_GROUPS_CSV = PROJECT_ROOT / "modules" / "biome_groups.csv"  # RESOLVE biome -> group (Python and R)

WDPA_POLYGONS = WDPA_DIR / "WDPA_polygons.geojson"
PAS_SELECTED = PAS_DIR / "pas_selected.gpkg"
PAS_REMOVED = PAS_DIR / "pas_removed.gpkg"
TRANSECT_POINTS = TRANSECTS_DIR / "transect_points.parquet"
PA_METRICS = RESULTS_DIR / "pa_edge_metrics_ndvi.csv"  # final: one row per PA-year
PIPELINE_COUNTS = RESULTS_DIR / "pipeline_counts.json"

# --- Earth Engine project (scripts 01, 03, 12) ----------------------------------------------
# Set once outside the repo: the GEE_PROJECT environment variable, else the project saved by
# `earthengine set_project YOUR-PROJECT-ID` (stored with the credentials).
EE_CREDENTIALS = Path.home() / ".config" / "earthengine" / "credentials"


def _saved_gee_project():
    """Project saved by `earthengine set_project`, or None."""
    try:
        return json.loads(EE_CREDENTIALS.read_text()).get("project")
    except (OSError, ValueError):
        return None


GEE_PROJECT = os.environ.get("GEE_PROJECT") or _saved_gee_project() or 'YOUR-PROJECT-ID'

# --- Data -----------------------------------------------------------------------------------
WDPA_COLLECTION = "WCMC/WDPA/202508/polygons"   # pinned WDPA release (August 2025)
START_YEAR = 2003
END_YEAR = 2025
INDEX_NAME = 'ndvi'                             # used in file and export names
NDVI_COLLECTION = 'MODIS/061/MOD13A1'           # MODIS Terra vegetation indices, 16-day, 500 m
NDVI_BAND = 'NDVI'
NDVI_SCALE_FACTOR = 0.0001

# --- PA selection (01, 02) ------------------------------------------------------------------
MIN_AREA_KM2 = 200
VALID_STATUS = ["Designated", "Established", "Inscribed"]
EXCLUDED_PIDS = [
    "555655917", "555656005", "555656013", "555665477", "555656021",
    "555665485", "555556142", "187", "555703455", "555563456", "15894",
    # Circular polygons (problematic geometries), found in the visual inspection
    "6690", "6742", "6738", "6745", "6733", "6744", "6807", "6740", "6736",
    "555682777", "555682773", "555543053", "555512070", "6794", "6737",
    "555592664", "555592665", "555621717", "95418",
    # City PAs (the Haram of Al-Madinah and the Haram of Makkah), found in the visual inspection
    "101853", "101854",
]
OVERLAP_THRESHOLD = 90         # % overlap: of both PAs = duplicates (oldest kept); of the smaller only = nested (larger kept)
MAX_HOLE_AREA = 2_250_000      # m² (1,500 m x 1,500 m): smaller holes are filled
MAX_GAP_PCT_LIMIT = 25         # PAs whose longest unsampled stretch exceeds this % of the boundary are removed

PROCESSING_CRS = 'ESRI:54009'  # Mollweide (equal area): areas and overlaps in 02 part 1 (transects: per-PA AEQD)
STORAGE_CRS = 'EPSG:4326'      # geographic coordinates: stored files and Earth Engine

# --- Transects (02) -------------------------------------------------------------------------
BOUNDARY_SPACING = 500         # m between boundary points
TRANSECT_SPACING = 2500        # m between points along a transect
POINTS_PER_SIDE = 2            # points on each side of the boundary point (5 per transect)
BUFFER_DIST = TRANSECT_SPACING * POINTS_PER_SIDE + 500  # PAs with no interior this far from the edge are removed

# --- Earth Engine uploads and exports (02, 03) ----------------------------------------------
CHUNK_SIZE = 400               # whole PAs per upload file (~1 million points)
GEE_SUBCHUNK_SIZE = 50_000     # points per exported CSV part
GEE_BATCH_SIZE = 10            # export tasks run at once
UPLOAD_PREFIX = 'transect_points_'                               # {UPLOAD_PREFIX}NNN.zip -> asset
GEE_ASSET_PREFIX = f'projects/{GEE_PROJECT}/assets/{UPLOAD_PREFIX}'
GEE_DRIVE_FOLDER = GEE_CSV_DIR.name                               # Drive folder of 03's CSVs: 03_gee, as the local folder
GEE_EXPORT_PREFIX = f'{INDEX_NAME}_gradient_'                     # {GEE_EXPORT_PREFIX}NNN_part_N.csv

# --- Visual inspection (tools/, optional, after the analysis) -------------------------------
# PAs excluded after the inspection are added to EXCLUDED_PIDS and the pipeline is rerun.
INSPECTION_INPUT = DATA_DIR / "05_analysis" / "pa_level.csv"     # the analysed PAs (R script 05)
INSPECTION_DIR = DATA_DIR / "inspection"
INSPECTION_ZIP = INSPECTION_DIR / "pas_final.zip"                # upload as the table asset pas_final
INSPECTION_CHECKLIST = INSPECTION_DIR / "checklist.csv"
INSPECTION_DECISIONS = ('', 'exclude', 'check')                  # blank = keep

# --- Validation: displaced transects (11-13) ------------------------------------------------
VALIDATION_DIR = DATA_DIR / "11_validation"                      # 11: sample, candidates
VALIDATION_UPLOAD_DIR = VALIDATION_DIR / "upload"
VALIDATION_GEE_CSV_DIR = DATA_DIR / "12_gee"                     # 12 CSVs (downloaded from Drive)
VALIDATION_SAMPLE_FRAME = DATA_DIR / "05_analysis" / "pa_level.csv"
VALIDATION_OUT_DIR = DATA_DIR / "13_validation"                  # 13 (read by R script 14)
VALIDATION_PA_METRICS = VALIDATION_OUT_DIR / "validation_displaced_pa_years.csv"
VALIDATION_LOG = VALIDATION_OUT_DIR / "validation_displacement_log.csv"
VALIDATION_CHECKS = VALIDATION_OUT_DIR / "validation_candidate_checks.csv"

VALIDATION_N_PER_BIOME = 50        # PAs per biome group
VALIDATION_RESERVE = 25            # extra PAs per biome, used in draw order for PAs without a copy
VALIDATION_N_CANDIDATES = 5        # copies per PA sent to Earth Engine; the first that passes is used
VALIDATION_MAX_ATTEMPTS = 200      # vectors drawn per PA for the local checks
VALIDATION_MIN_DIST_KM = 50        # d = the larger of this and the PA's footprint (no overlap with it)
VALIDATION_DIST_MULT = 2           # distance uniform in [d, mult x d]
VALIDATION_MAX_BOUNDARY_SHARE = 0.10   # max share of copied transects with a PA boundary within 2.5 km of their centre
VALIDATION_SEED = 20261108
# The copy passes the analysis-set rules of R script 05 (same values as modules/r/config.R;
# tests/test_validation.py checks that they agree), and keeps the PA's biome (as script 01).
VALIDATION_MIN_COMPLETE_PCT = 75   # sea rule: median % of complete transects
VALIDATION_MAX_WATER_PCT = 75      # lake rule: mean % of transects with water
VALIDATION_MIN_YEARS = 5           # years with both edge metrics

VALIDATION_UPLOAD_PREFIX = 'displaced_points_'
VALIDATION_ASSET_PREFIX = f'projects/{GEE_PROJECT}/assets/{VALIDATION_UPLOAD_PREFIX}'
VALIDATION_POLY_UPLOAD_PREFIX = 'displaced_polygons_'
VALIDATION_POLY_ASSET_PREFIX = f'projects/{GEE_PROJECT}/assets/{VALIDATION_POLY_UPLOAD_PREFIX}'
VALIDATION_DRIVE_FOLDER = VALIDATION_GEE_CSV_DIR.name            # Drive folder of 12's CSVs: 12_gee, as the local folder
VALIDATION_EXPORT_PREFIX = f'{INDEX_NAME}_gradient_displaced_'   # 12's CSV names
VALIDATION_BIOME_EXPORT = 'displaced_polygons_biome'             # 12's polygon biomes, same Drive folder


def get_num_chunks():
    """Number of upload zips written by script 02 (one Earth Engine asset each), counted on call."""
    return len(list(UPLOAD_DIR.glob(f'{UPLOAD_PREFIX}*.zip')))


def require_gee_project():
    """Raise if GEE_PROJECT is still the placeholder. Called by scripts 01, 03 and 12."""
    if GEE_PROJECT == 'YOUR-PROJECT-ID':
        raise ValueError(
            "GEE_PROJECT has not been set. Run once in the container terminal:\n"
            "    earthengine authenticate --auth_mode=notebook\n"
            "    earthengine set_project YOUR-PROJECT-ID\n"
            "or set the GEE_PROJECT environment variable."
        )
