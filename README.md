# Protected Area Edge Analysis

## Overview
This project analyzes change in landscape gradients at protected area boundaries using Earth Engine and geospatial methods. The workflow processes ~4,000 protected areas globally, generating perpendicular transects at boundaries to sample environmental variables including vegetation indices (NDVI, LAI, FPAR), built-up index (NDBI), water presence, human modification, and topography from satellite imagery (2003-2025). The analysis produces ~8.9 million transect points with edge detection metrics to quantify annual change in boundary sharpness from 2003 - 2025.

## Quick Start

### Workflow Overview
```
Phase 1: Data Preparation (Run Once)
├── 01_collect_protected_areas.py     → Export to GEE → Manual download
└── 02_generate_sampling_points.py    → Filter + transects → Manual upload to GEE

Phase 2: Remote Sensing (Run Per Index)
└── 03_sample_satellite_imagery.py    → Sample in GEE → Manual download

Phase 3: Analysis (Run Per Index)
└── 04_compute_edge_metrics.py        → Edge metrics + per-PA trends → final CSV
```

### First Time Setup
```bash
# 1. Clone the repository
git clone https://github.com/your-org/boundary-project.git
cd boundary-project

# 2. Open in VS Code and reopen in container
code .
# → VS Code will prompt "Reopen in Container" — click it
# → Wait ~5 minutes for the container to build and install all dependencies

# 3. Set your GEE project ID (required before running any script)
nano modules/config.py  # Change GEE_PROJECT = 'YOUR-GEE-PROJECT' to your project ID

# 4. Authenticate with Earth Engine (one-time, run in terminal inside the container)
python3 -c "import ee; ee.Authenticate()"
# → Follow the browser link, sign in with your Google account,
#   copy the token back into the terminal when prompted

# 5. Run Phase 1 (Data Preparation)
python scripts/01_collect_protected_areas.py
# → Download WDPA_polygons.geojson from Google Drive to data/raw/

python scripts/02_generate_sampling_points.py
# → Upload chunk_*.shp to GEE assets (see script output for paths)
```

> **Reproducing the exact published analysis:** Script 01 reads a date-stamped WDPA release
> (`WCMC/WDPA/202508/polygons`, August 2025). Earth Engine keeps historical monthly releases,
> so re-running script 01 reproduces the input used here. `WDPA_polygons.geojson` is not
> included in this repository because the WDPA terms do not allow redistribution (see [License](#license)).

### For Each Index (ndvi, lai, fpar, ndbi)
```bash
# 1. Set index in config
nano modules/config.py  # Change INDEX_NAME = 'ndvi'

# 2. Sample satellite imagery
python scripts/03_sample_satellite_imagery.py
# → Download CSVs from Google Drive to results/raw/{index}_raw/

# 3. Compute edge metrics and trends
python scripts/04_compute_edge_metrics.py
# → results/wdpa_df_{index}_trends.csv (final PA-year dataset with trends)
```

## Project Structure
```
boundary-project/
├── scripts/                  # Main workflow (4 scripts)
│   ├── 01_collect_protected_areas.py
│   ├── 02_generate_sampling_points.py
│   ├── 03_sample_satellite_imagery.py
│   └── 04_compute_edge_metrics.py
├── modules/                  # Reusable domain logic
│   ├── config.py             # All parameters and paths
│   ├── remotesensing.py      # Google Earth Engine functions
│   ├── preprocessing.py      # Geometry and transect generation
│   └── analysis.py           # Edge detection, PA metrics, trend classification
├── data/
│   ├── raw/                  # WDPA_polygons.geojson
│   ├── intermediate/         # wdpa_filtered/, transect_chunks/
│   └── processed/            # attributes_final.csv
└── results/
    ├── raw/                  # ndvi_raw/, lai_raw/, fpar_raw/, ndbi_raw/ (GEE CSVs + transect_chunks/)
    ├── wdpa_df_{index}.parquet
    └── wdpa_df_{index}_trends.csv   # Final PA-year dataset with trends
```

## Setup

### Prerequisites
- [VS Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) running
- A [Google Earth Engine account](https://earthengine.google.com/signup/) with API access enabled

### Initial Setup
1. Clone the repository and open in VS Code:
   ```bash
   git clone https://github.com/your-org/boundary-project.git
   code boundary-project
   ```
2. When prompted, click **"Reopen in Container"** — VS Code will build the Docker container and install all pinned dependencies from `requirements.txt` (~5 minutes).
3. Open `modules/config.py` and set your GEE project ID:
   ```python
   GEE_PROJECT = 'your-project-id'  # at the top of the file
   ```
4. Authenticate with Earth Engine in the VS Code terminal:
   ```bash
   python3 -c "import ee; ee.Authenticate()"
   ```
   Follow the browser link, sign in with your Google account, and paste the token back into the terminal. Credentials are stored inside the container and persist for the session.

## Detailed Workflow

### Phase 1: Data Preparation (Run Once)

#### Script 01: Collect Protected Areas
```bash
python scripts/01_collect_protected_areas.py
```
Filters WDPA polygons (≥200 km², non-marine), adds biome data, exports to Google Drive.

**⚠️ MANUAL STEP**: Download `WDPA_polygons.geojson` from Google Drive → `data/raw/`

> **Note on reproducibility:** The WDPA collection is date-stamped (`WDPA_COLLECTION = "WCMC/WDPA/202508/polygons"`
> in `modules/config.py`). Earth Engine keeps historical monthly releases back to July 2017, so re-running this
> script reproduces the published input. Newer releases (e.g., `WCMC/WDPA/current/polygons`) contain updated
> protected area boundaries and may alter results.

#### Script 02: Generate Sampling Points
```bash
python scripts/02_generate_sampling_points.py
```
Cleans geometries, removes overlaps, generates perpendicular transects (5 points per transect: 2 inner, 1 boundary, 2 outer), writes chunks of `CHUNK_SIZE` (400) PAs each (currently 11 files, chunk_000–chunk_010; ~8.9M points total). Old chunk files are deleted before each run.

**⚠️ MANUAL STEP**: Upload every chunk_*.shp file (chunk_000.shp onward) to GEE as assets `projects/{your-project}/assets/chunk_XXX`. Script 03 samples one asset per local chunk file.

---

### Phase 2: Remote Sensing (Run Per Index)

#### Script 03: Sample Satellite Imagery
```bash
# Edit modules/config.py: INDEX_NAME = 'ndvi'  # or 'lai', 'fpar', 'ndbi'
python scripts/03_sample_satellite_imagery.py
```
Samples MODIS imagery (2003-2025) at transect points, exports ~180 CSV files to Google Drive.

**⚠️ MANUAL STEP**: Download all CSVs from Google Drive folder `{index}_raw` → `results/raw/{index}_raw/`

---

### Phase 3: Analysis (Run Per Index)

#### Script 04: Compute Edge Metrics
```bash
python scripts/04_compute_edge_metrics.py
```
Combines CSVs, flags edge transects, aggregates to PA-year edge extent and edge intensity (Cohen's d), then fits an OLS trend of each metric on year for every PA.
- Output: `results/raw/{index}_raw/transect_chunks/{index}_transect_chunk_*.parquet`, `results/wdpa_df_{index}.parquet`, `results/wdpa_df_{index}_trends.csv` (final dataset)

## Configuration

Edit [modules/config.py](modules/config.py) for:

**Required settings**:
- `GEE_PROJECT`: Your Google Earth Engine project ID
- `INDEX_NAME`: Current index ('ndvi', 'lai', 'fpar', 'ndbi')

**Optional settings** (defaults work for most cases):
- `START_YEAR`, `END_YEAR`: Analysis time period (default: 2003-2025)
- `BOUNDARY_SPACING`: Distance between boundary points (default: 500m)
- `TRANSECT_SPACING`: Distance between transects (default: 2500m)
- `POINTS_PER_TRANSECT`: Points on each side of boundary (default: 2)

**Indices available**:
- NDVI: Normalized Difference Vegetation Index (MOD13A1)
- LAI: Leaf Area Index (MCD15A3H)
- FPAR: Fraction Photosynthetically Active Radiation (MCD15A3H)
- NDBI: Normalized Difference Built-up Index (MOD09GA)

## Troubleshooting

**"Input file not found"**
- Ensure previous scripts completed successfully
- Check that manual download/upload steps were completed

**"No CSV files found"**
- Download GEE results to correct folder: `results/raw/{index}_raw/`
- Verify all CSV files from Google Drive are present

**"Invalid INDEX_NAME"**
- Edit `modules/config.py`: set INDEX_NAME to 'ndvi', 'lai', 'fpar', or 'ndbi'

**GEE authentication errors**
```bash
python3 -c "import ee; ee.Authenticate()"
```
Follow the browser link, sign in, and paste the token into the terminal. Then re-run the script.

**"GEE_PROJECT has not been set"**
- Open `modules/config.py` and change `GEE_PROJECT = 'YOUR-GEE-PROJECT'` to your actual project ID
- Your project ID is visible at [console.cloud.google.com](https://console.cloud.google.com/)

## Legacy Workflow

The original workflow in `src/` remains available for reference.

## License
The code in this project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Data:** Protected area boundaries and attributes come from the World Database on Protected Areas (WDPA) and are not covered by the MIT License. They are subject to the [Protected Planet Terms and Conditions](https://www.protectedplanet.net/en/legal): the data may not be redistributed, sub-licensed (including within derivative works), or put to commercial use (including derivative works) without prior written permission from UNEP-WCMC. Obtain the data from [Protected Planet](https://www.protectedplanet.net) or Earth Engine (`WCMC/WDPA/202508/polygons`), and cite it as:

> UNEP-WCMC and IUCN (2025), Protected Planet: The World Database on Protected Areas (WDPA) [On-line], August 2025, Cambridge, UK: UNEP-WCMC and IUCN. Available at: www.protectedplanet.net.
