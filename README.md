# Protected Area Edge Analysis

This project analyzes change in vegetation contrast at protected area (PA) boundaries using Google Earth Engine and geospatial methods. The workflow selects more than 4,000 terrestrial PAs from the WDPA, generates transects perpendicular to each boundary, samples the gradient magnitude of annual median NDVI (MODIS MOD13A1, 2001–2025) plus water, human modification, and topography at every point, and computes yearly edge metrics for each PA. The analysis models the mean edge extent and intensity and change over time. 

Run the scripts to generate the data and results used in this analysis. 

## Setup

You need [VS Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), [Docker Desktop](https://www.docker.com/products/docker-desktop/), and a Google account with a Google Cloud project [registered for Earth Engine](https://code.earthengine.google.com/register) (free for noncommercial use).

1. Clone the repository and open it in VS Code:
   ```bash
   git clone https://github.com/avanscoyoc/boundary-project.git
   code boundary-project
   ```
2. Click **Reopen in Container**. VS Code builds the container with all pinned dependencies (Python from `requirements.txt` + `constraints.txt`, R 4.5.3 from `.devcontainer/pixi.lock`; about 5 minutes).
3. Sign in to Earth Engine and set your project, once, in the VS Code terminal:
   ```bash
   earthengine authenticate --auth_mode=notebook
   earthengine set_project YOUR-PROJECT-ID
   ```
   Open the printed link, sign in, and paste the code back. Sign-in and project are saved in a Docker volume (`boundary-project-earthengine`), not in the repo, so they survive **Rebuild Container**. 


## Run the pipeline

Run from the repository root, in order. *Note: Lines starting with `→` are manual steps*. The scripts gather all data used for this project from the public Earth Engine catalog. Any files that need to be uploaded as Earth Engine assets (scripts 02, 11) are saved locally in `data/`; Any Earth engine exports (scripts 01, 03, 12) are saved to your Google Drive and should be downloaded to `data/`.

```bash
# 01 Protected areas (Earth Engine, export can take ~12 h)
python scripts/01_collect_protected_areas.py
#  → download WDPA_polygons.geojson from Google Drive to data/01_wdpa/

# 02 Choose PAs and build transects (local)
python scripts/02_generate_sampling_points.py
#  → upload each data/02_transects/upload/transect_points_NNN.zip as the table asset transect_points_NNN
#    (one asset per file; script 02 prints the list)

# 03 Sample NDVI gradients (Earth Engine, ~5 h)
python scripts/03_sample_satellite_imagery.py
#  → download the Drive folder 03_gee and put its CSVs in data/03_gee/ (Drive may split the
#    download into several zips: move the CSVs of each into the one folder)

# 04 Edge metrics (local) → results/pa_edge_metrics_ndvi.csv
python scripts/04_compute_edge_metrics.py

# 05–10 Statistical analysis (R; 06, 08 and 09 take ~2hrs for spatial fits and can require ~2GB of memory: run them one at a time)
Rscript scripts/05_prepare_analysis_data.R
Rscript scripts/06_models_mean.R
Rscript scripts/07_trends.R
Rscript scripts/08_models_change.R
Rscript scripts/09_sensitivity.R
Rscript scripts/10_figures.R

# 11–14 Validation 
python scripts/11_validation_displace_transects.py
#  → upload each zip in data/11_validation/upload/ as a table asset of the same name
python scripts/12_validation_sample_imagery.py
#  → download the Drive folder 12_gee and put its CSVs in data/12_gee/ (as for 03)
python scripts/13_validation_edge_metrics.py
Rscript scripts/14_validation_compare.R
```

Scripts 06, 08 and 09 take `--cores K` (fits run in parallel, default 2). At full size each spatial fit takes about an hour and 1.5 GB of memory (8 GB machine: one script at a time). Fits are cached in `data/05_analysis/cache/` by content (about 300 MB each), so a rerun only refits what changed; 09's quick fixed-range fits are not cached.

## What each script does

### Data pipeline

#### 01 Collect PAs · Earth Engine
- Gets geometry data from a pinned WDPA release (August 2025)
- Keeps terrestrial PAs of at least 200 km² that are designated, established or inscribed and single-part.
- Excludes Marine Protected Areas, UNESCO-MAB Biosphere Reserves, and 32 PAs found to have problematic geometries upon visual inspection (i.e. points, circles).
- Counts the PAs that each filter removes (`pipeline_counts.json`).
- Assigns each PA the RESOLVE 2017 biome with the largest overlap.
- Exports `WDPA_polygons.geojson` to Drive.

#### 02 Clean PA data and build transects · Python
- Fills holes smaller than 2.25 km².
- Removes duplicate PAs (> 90% overlap of both PAs; oldest kept) 
- Removes nested PAs (> 90% of the smaller; larger kept)
- Merges touching PAs with the same name and country (ID and year of the oldest member, other attributes of the largest).
- Builds transects in each PA's own equidistant projection: 5 points 2.5 km apart (2 inside, 1 on the boundary, 2 outside), every 500 m of boundary.
- Keeps a transect only if its inner half lies inside the PA and its outer half outside.
- Removes PAs with no interior 5.5 km from the edge, no valid transect, or a boundary gap that is >25% of the perimeter. (All removed PAs and the reason are stored in `data/02_pas/pas_removed.gpkg`)
- Flags transects whose outer points fall in another of 01's PAs (`in_other_pa`).
- Writes zips containing PA transect points to upload to Earth Engine `data/02_transects/upload`.

#### 03 Sample imagery · Earth Engine
- Reads the transect points uploaded as assets in script 02 (`transect_points_NNN`).
- Takes the annual median of each year's MOD13A1 NDVI composites, 2001–2025.
- Computes the NDVI gradient magnitude on MODIS's native ~463 m grid.
- Samples the NDVI gradient magnitude of every point in every year. Records missing values as -9999.
- Adds covariates by sampling the maximum water extent (JRC GSW 1.4 `max_extent`), human modification (gHM 2016), and elevation and slope (Copernicus GLO-30, 2024) at all points, resampling all layers to the MODIS grid.
- Exports one CSV per 50,000 points to a Drive folder `03_gee`; user must download them to `data/03_gee/`.

#### 04 Edge metrics · Python
- Computes 
   - `edge`: a transect-year is an edge when the boundary gradient magnitude value is higher than at least one point on each side
   - `edge_extent`: the proportion of transects that are edges for a PA
   - `edge_intensity`: the mean of four Cohen's d values (boundary point vs each other position, pooled SD) for edge transects of a PA
- Groups the RESOLVE biomes into 8 (`modules/biome_groups.csv`): Tropical Grassland, Temperate & Montane Grassland, Tundra, Desert, Mediterranean, Tropical Forest, Temperate Forest, Boreal Forest.
- Writes dataset to `results/pa_edge_metrics_ndvi.csv`.

### Statistics

#### 05 Analysis set · R
- Removes Mangrove and Rock & Ice PAs.
- Removes PAs dominated by sea (< 75% of transects complete in a typical year) or open water (> 75% of transects with water).
- Removes PAs missing a covariate or with fewer than 5 years of either metric (removals stored in `data/05_analysis/excluded_pas.csv`).
- Imputes missing establishment years (median of 1,000 draws from the country's known years).
- Z-scores the covariates.

#### 06 Mean models · R
- Fits spatial GLS models of each PA's mean edge extent and intensity (exponential correlation over great-circle distance, REML), with non-strict IUCN and Boreal Forest as reference levels.
- Creates **Table SX** (main effects).
- Creates **Table SY** (HMI interactions with IUCN category and biome, Wald tests, HMI slopes per group, and predictions).

#### 07 Trends · R
- Fits an OLS trend per PA and metric classified as increase, no trend, or decrease (Benjamini–Hochberg); Yue–Pilon trends as a check.
- Converts each trend to standardised change (slope × 24 years / SD of the PA means).
- Creates **Table SV** (change per biome and for all PAs: bootstrap, area-weighted and Sen-slope means; correlation of extent and intensity trends).
- Creates the Fig. 4 tables: bootstrap means of change per biome (4A; descriptive, PAs treated as independent) and the case-study PAs' series (4B).
- Checks the autocorrelation of the trend residuals (diagnostic).

#### 08 Change models · R
- Fits spatial GLS models of standardised change on the mean models' predictors, with 06's correlation structure and reference levels.
- Creates **Table SZ** (unweighted, plus precision-weighted as a check).
- Creates **Table SW** (whether biomes differ in change; overall Wald test, pairwise contrasts with Holm adjustment, compact letters).

#### 09 Sensitivity · R
- Reruns the PA means, trends, and models (06–08) on transect and PA subests to check that the results hold for PAs:
   - without water transects
   - without transects whose outer points fall in another PA
   - without both
   - established by 2001
   - with a reported establishment year
   - with more than 50% of their transects
- The subsets use only PAs that keep ≥ 60 complete transects in at least 5 years, and reuses the spatial ranges estimated in 06 and 08 rather than re-estimating. 

#### 10 Figures · R
- Draws figures from the tables of scripts 06–08.
   - **Fig. 2**: writes map layers that are styled outside the repository (GeoPackage and KML in `data/10_map_layers/`). 
   - **Fig. 3**: mean-model coefficients in response SDs, plus one panel for each human-modification interaction whose Wald test is significant (p < 0.05).
   - **Fig. 4**: standardised change by biome and the case study time series. 
   - **Supplementary Figure**: per-PA change in edge extent vs edge intensity.
   - **Diagnostics Figures**: Moran's I correlogram of the mean-model residuals and residual plots of the spatial models.

### Validation
Tests whether the edge signal at the boundary exceeds what the same transects record in the surrounding landscape.

#### 11 Validation Displace transects · Python
- Draws 50 analysed PAs per biome group.
- Moves each PA's whole transect set by a random vector (no rotation), d to 2d away (d = the larger of 50 km and the PA's footprint).
- Keeps the displaced PA copy if at most 10% of its transects have a boundary of one of 01's PAs within 2.5 km of their center.

#### 12 Validation Sample imagery · Earth Engine
- Samples the displaced PAs with 03's image, and designates its biome based on largest area.

#### 13 Validation Edge metrics · Python
- Measures edge metrics for each displaced PA with 04's functions.
- Keeps the displaced PA if it passes 05's rules (same biome; sea, lake, 5 years). 

#### 14 Validtation Compare · R
- Compares each PA with its displaced copy in the 2001–2025 mean of each metric, per biome and for all PAs.
- Reports the mean with a bootstrap CI, the median, the share of PAs above their displaced copy, and a Wilcoxon test (Holm across biomes).

#### Visual inspection (optional)
- You can visually inspect every PA and transect point used in this analysis by running `tools/inspection_export.py`
- Upload the zip `data/inspection/analysis_pas.zip` to Earth Engine assets. Set `ASSET_ROOT` in `tools/inspection_app.js`, and run the .js script in the Earth Engine Code Editor.


## Configuration

Python settings are in [modules/config.py](modules/config.py) (the defaults reproduce this analysis): `START_YEAR`/`END_YEAR` (2001–2025), `BOUNDARY_SPACING` (500 m), `TRANSECT_SPACING` (2,500 m), `POINTS_PER_SIDE` (2), `CHUNK_SIZE` (400 PAs per upload file), `EXCLUDED_PIDS`, `VALIDATION_*`. R settings (analysis set, reference levels, seeds, spatial model, sensitivity, validation) are in [modules/r/config.R](modules/r/config.R).

## Troubleshooting

- **"Input file not found":** a previous script or a manual download/upload step is missing.
- **Script stops on missing, unexpected or duplicate points, or on unrecognised CSVs:** remove Drive duplicates such as `name (1).csv` in `data/03_gee` or `data/12_gee` outputs.
- **Earth Engine sign-in errors:** try `earthengine authenticate --auth_mode=notebook --force` 
- **"GEE_PROJECT has not been set":** run `earthengine set_project YOUR-PROJECT-ID`, or set the `GEE_PROJECT` environment variable (project IDs are listed at [console.cloud.google.com](https://console.cloud.google.com/)).

## License
The code in this project is licensed under the MIT License [LICENSE](LICENSE). 

The `modules/r/corHaversine.R` is adapted from a Stack Overflow answer by Nate Pope (2013) and is shared under CC BY-SA 3.0.

Protected area boundaries and attributes come from the World Database on Protected Areas (WDPA) and are subject to the [Protected Planet Terms and Conditions](https://www.protectedplanet.net/en/legal). Obtain the data from [Protected Planet](https://www.protectedplanet.net) or Earth Engine (`WCMC/WDPA/202508/polygons`), and cite it as:

> UNEP-WCMC and IUCN (2025), Protected Planet: The World Database on Protected Areas (WDPA) [On-line], August 2025, Cambridge, UK: UNEP-WCMC and IUCN. Available at: www.protectedplanet.net.
