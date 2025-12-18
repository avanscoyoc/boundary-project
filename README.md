# boundary-project

## Overview
This project analyzes habitat continuity and landscape gradients across protected area boundaries using Earth Engine and geospatial methods. The workflow processes ~4,000 protected areas globally, generating perpendicular transects at boundaries to sample environmental variables including water presence (Global Surface Water), human modification, and NDVI gradients from MODIS imagery. The analysis produces ~8.6 million transect points with chunked processing to handle large-scale spatial data efficiently.

## Project Structure
```
boundary-project/
├── .devcontainer/            # Docker container configuration
├── src/
│   ├── 1_get_wdpas.ipynb     # Download and filter WDPA polygons from Earth Engine
│   ├── 2_filter_wdpas.ipynb  # Remove overlaps, duplicate names, and narrow polygons
│   ├── 3_create_points.ipynb # Generate transect points and remove bad transects with chunked processing
│   ├── 4_gee_tasks.ipynb     # Sample all points in Earth Engine (water, human modification, NDVI)
│   └── utils.py              # Reusable functions for all notebooks
├── data/
│   ├── transect_chunks/      # Intermediate chunk files (generated in Step 3)
│   ├── wdpa_filtered/        # Filtered WDPA shapefile (generated in Step 2)
│   ├── attributes_final.csv  # Protected area metadata (generated in Step 3)
│   ├── transects_final.csv   # Final transect points in EPSG:4326 (generated in Step 3)
│   └── WDPA_polygons.geojson # Spatial WDPA polygon file (manually added in Step 1)
└── results/                  # Earth Engine export outputs (manually added in Step 3)
```

## Setup Instructions

### Prerequisites
- VS Code installed on your machine
- Docker installed and running
- Google Earth Engine account with project access

### Initial Setup
1. Clone the repository
2. Open the project folder in VS Code
3. When prompted, select **"Reopen in Container"** (or use Command Palette: `Dev Containers: Reopen in Container`)
4. Wait for the Docker container to build and start (~2-3 minutes first time)
5. Authenticate with Google Earth Engine (see section below)

### Google Earth Engine Authentication
On first run, authenticate with Earth Engine:
```python
import ee
ee.Authenticate()  # Follow the browser prompts to authorize
ee.Initialize(project='your-project-id') #update this line for each script
```

## Running the Workflow

### Complete Pipeline
Run notebooks sequentially in the `src/` directory:

1. **`1_get_wdpas.ipynb`**: Download WDPA data from Earth Engine, filter by criteria, add biome information
   - Output: `data/WDPA_polygons.geojson` (~6,358 polygons)

2. **`2_filter_wdpas.ipynb`**: Clean geometries, remove overlaps and duplicates
   - Output: `data/wdpa_filtered/` shapefile (~4,176 polygons)

3. **`3_create_points.ipynb`**: Generate transect points with chunked processing
   - Parameters: 500m boundary spacing, 2500m transect spacing, 2 inner/outer points
   - Output: `data/transects_final.csv` (~8.6M points for 3,939 polygons), `data/attributes_final.csv` (metadata)

4. **`4_gee_tasks.ipynb`**: Process transects in Earth Engine with task queue management
   - Samples: Global Surface Water, Human Modification, MODIS NDVI gradients
   - Output: Results exported to Google Drive (folder: `boundary-project_results_YYYYMMDD/`)

### Key Parameters
- **CRS**: ESRI:54009 (Mollweide) for geoprocessing, EPSG:4326 (WGS84) for Earth Engine
- **Chunking**: 500 PAs per chunk (~1M points) to manage memory
- **Buffer**: 5500m inner buffer for transect filtering

## Dependencies
All dependencies are automatically installed when building the Docker container:
- `earthengine-api` - Google Earth Engine Python API
- `geopandas` - Geospatial data processing
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `shapely` - Geometric operations

## Notes
- Processing all ~4,000 protected areas takes several hours
- Earth Engine exports are asynchronous; monitor task status in the Earth Engine Code Editor
- Final transect data can be rejoined with attributes using `WDPA_PID` column

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.