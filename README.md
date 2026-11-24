# Protected Area Edge Analysis

This project analyzes change in vegetation contrast at protected area (PA) boundaries using Google Earth Engine and geospatial methods. The workflow selects more than 4,000 terrestrial PAs from the WDPA, generates transects perpendicular to each boundary, samples the gradient magnitude of annual median NDVI (MODIS MOD13A1, 2001–2025) plus water, human modification, and topography at every point, and computes yearly edge metrics for each PA. The statistical analysis (R) models mean edge extent and intensity and their change, and a validation compares every sampled PA with a displaced copy of its transects in the surrounding landscape.

The data the scripts produce (`data/`, `results/`) are not part of the repository: run the scripts to regenerate them. WDPA boundaries may not be redistributed (see [License](#license)).

## Setup

You need [VS Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), [Docker Desktop](https://www.docker.com/products/docker-desktop/), and a Google account with a Google Cloud project [registered for Earth Engine](https://code.earthengine.google.com/register) (free for noncommercial use; use your own project, no access to the authors' project is needed).

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
   Open the printed link, sign in, and paste the code back. Sign-in and project are saved in a Docker volume (`boundary-project-earthengine`), not in the repo, so they survive **Rebuild Container**. To use another project for one run, put `GEE_PROJECT=other-project` before the command. To share your project and its uploaded assets, give the other account the roles *Service Usage Consumer* and *Earth Engine Resource Writer* in Cloud Console → IAM.

Every dataset the scripts read is in the public Earth Engine catalog. Uploads (scripts 02, 11 and the optional inspection export) go into your project; exports (scripts 01, 03, 12) go to your Google Drive.

## Run the pipeline

Run from the repository root, in order. Lines starting with `→` are manual steps.

```bash
# 01 Protected areas (Earth Engine, export can take ~12 h)
python scripts/01_collect_protected_areas.py
#  → download WDPA_polygons.geojson from Google Drive to data/01_wdpa/

# 02 Choose PAs and build transects (local)
python scripts/02_generate_sampling_points.py
#  → upload each data/02_transects/upload/transect_points_NNN.zip as the table asset transect_points_NNN
#    (one asset per file; script 02 prints the list)

# 03 Sample NDVI gradients (Earth Engine, ~5 h); --test runs one part of asset 000 first
python scripts/03_sample_satellite_imagery.py --test
python scripts/03_sample_satellite_imagery.py
#  → download the Drive folder 03_gee and put its CSVs in data/03_gee/ (Drive may split the
#    download into several zips: move the CSVs of each into the one folder)

# 04 Edge metrics (local) → results/pa_edge_metrics_ndvi.csv
python scripts/04_compute_edge_metrics.py

# 05–10 Statistical analysis (R; 06, 08 and 09 take hours for spatial fits: run them in the background, one at a time)
Rscript scripts/05_prepare_analysis_data.R
Rscript scripts/06_models_mean.R
Rscript scripts/07_trends.R
Rscript scripts/08_models_change.R
Rscript scripts/09_sensitivity.R
Rscript scripts/10_figures.R

# 11–14 Validation (after 05)
python scripts/11_validation_displace_transects.py
#  → upload each zip in data/11_validation/upload/ as a table asset of the same name
python scripts/12_validation_sample_imagery.py --test
python scripts/12_validation_sample_imagery.py
#  → download the Drive folder 12_gee and put its CSVs in data/12_gee/ (as for 03)
python scripts/13_validation_edge_metrics.py
Rscript scripts/14_validation_compare.R
```

Scripts 06, 08 and 09 take `--cores K` (fits run in parallel, default 2). At full size each spatial fit takes about an hour and 1.5 GB of memory (8 GB machine: one script at a time). Fits are cached in `data/05_analysis/cache/` by content (about 300 MB each), so a rerun only refits what changed; 09's quick fixed-range fits are not cached. **Tests:** `python -m pytest tests/` and `Rscript -e 'testthat::test_dir("tests/r")'`.

## What each script does

### Data pipeline

#### 01 Collect PAs · Earth Engine
- Keeps terrestrial PAs of at least 200 km² that are designated, established or inscribed and single-part.
- Excludes Marine Protected Areas, UNESCO-MAB Biosphere Reserves and the 32 PIDs in `EXCLUDED_PIDS` (21 from the visual inspection: 19 circular polygons, 2 city PAs).
- Counts the PAs each filter removes (`pipeline_counts.json`).
- Assigns each PA the RESOLVE 2017 biome with the largest overlap.
- Exports `WDPA_polygons.geojson` to Drive.
- Uses a pinned WDPA release (August 2025). 

#### 02 Choose PAs and build transects · Python
- Fills holes smaller than 2.25 km².
- Removes duplicates (> 90% overlap of both PAs; oldest kept) and nested PAs (> 90% of the smaller; larger kept).
- Merges touching PAs with the same name and country (ID and year of the oldest member, other attributes of the largest).
- Builds transects in each PA's own equidistant projection: 5 points 2.5 km apart (2 inside, 1 on the boundary, 2 outside), every 500 m of boundary.
- Keeps a transect only if its inner half lies inside the PA and its outer half outside.
- Removes PAs with no interior 5.5 km from the edge, no valid transect, or a boundary gap over 25% (the gap rule lets a few PAs keep few transects; see 09).
- Flags transects whose outer points fall in another of 01's PAs (`in_other_pa`).
- Writes upload zips of 400 PAs; removed PAs and why are in `pas_removed.gpkg`.

#### 03 Sample imagery · Earth Engine
- Samples every transect point in every year, 2001–2025 (2001 is MOD13A1's first full year).
<<<<<<< HEAD
- NDVI: gradient magnitude of the annual median MOD13A1 NDVI, on its native ~463 m grid, with no quality or snow masking.
- Covariates: water (JRC GSW 1.4) max_extent at all points, human modification (gHM 2016) at 2 outside points, elevation and slope (Copernicus GLO-30, 2024) at boundary points.
=======
- NDVI: gradient magnitude of the annual median MOD13A1 NDVI, on its native ~463 m grid.
- Covariates: water (JRC GSW 1.4), human modification (gHM 2016), elevation and slope (Copernicus GLO-30, 2024).
>>>>>>> b840fd1e01bfe5be540388b7e395d9d4a003dc95
- Exports CSVs of 50,000 points each to Drive.

#### 04 Edge metrics · Python
- Checks that every exported point arrived exactly once.
- Keeps transect-years with gradient values at all 5 points.
- Computes 
   - `edge`: a transect-year is an edge when the boundary value is higher than at least one point on each side
   - `edge_extent`: the share of complete transects that are edges
   - `edge_intensity`: the mean of four Cohen's d values (boundary point vs each other position, pooled SD), computed on edge transects only

- Calculates PA-year edge extent and intensity for sensitivity tests with four transect versions: all, without water, without other-PA, without either.
- Groups the RESOLVE biomes into 8 (`modules/biome_groups.csv`): Tropical Grassland, Temperate & Montane Grassland, Tundra, Desert, Mediterranean, Tropical Forest, Temperate Forest, Boreal Forest.
- Writes `results/pa_edge_metrics_ndvi.csv`.

### Statistics

Settings for 05–10 and 14 are in [modules/r/config.R](modules/r/config.R).

#### 05 Analysis set · R
- Removes Mangrove and Rock & Ice PAs.
- Removes PAs dominated by sea (< 75% of transects complete in a typical year) or open water (> 75% of transects with water).
- Removes PAs missing a covariate or with fewer than 5 years of either metric (all removals in `excluded_pas.csv`).
- Imputes missing establishment years (median of 1,000 draws from the country's known years).
- Z-scores the covariates.

#### 06 Mean models · R
- Spatial GLS of mean edge extent and intensity (exponential correlation over great-circle distance, REML).
- Main effects and HMI interactions (Tables SX, SY).
- Reference levels in every model: non-strict IUCN and Boreal Forest.

#### 07 Trends · R
- Per-PA OLS trends with Benjamini–Hochberg classes; Yue–Pilon as a check.
- Standardised change and its bootstrap means per biome (Fig. 4A; descriptive, PAs treated as independent).
- Autocorrelation of the trend residuals (diagnostic).
- Series of the Fig. 4B case-study PAs.

#### 08 Change models · R
- Spatial GLS of standardised change, unweighted and precision-weighted (Table SZ).
- Biome comparison of change, with the same spatial correlation (Table SW).

#### 09 Sensitivity · R
- Repeats the headline results without water or other-PA transects.
- Repeats them for PAs established by 2001, with a reported establishment year, or keeping at least 50% of their transects.
- Fixes the spatial ranges at the main estimates.

#### 10 Figures · R
- Fig. 3 (interaction panels only where the Wald test is significant) and Fig. 4.
- Supplementary and diagnostic figures.
- Fig. 2 map layers (GeoPackage, KML), styled outside the repository.

### Validation

Tests whether the edge signal at the boundary exceeds what the same transects record in the surrounding landscape.

#### 11 Displace transects · Python
- Draws 50 analysed PAs per biome group, plus 25 reserves.
- Moves each PA's whole transect set by a random vector (no rotation), d to 2d away (d = the larger of 50 km and the PA's footprint).
- Keeps a copy if at most 10% of its transects have a boundary of one of 01's PAs within 2.5 km of their centre.
- Makes up to 5 copies per PA; the PA polygon moves with them.

#### 12 Sample copies · Earth Engine
- Samples the copies with 03's image.
- Gives each moved PA polygon its biome, based on largest area.

#### 13 Accept copies · Python
- Measures each copy with 04's functions.
- Keeps, per PA, the first copy that passes 05's rules (same biome; sea, lake, 5 years); reserves replace PAs without one.

#### 14 Compare · R
- Compares PA minus copy in the 2001–2025 mean of each metric, per biome and for all PAs.
- Reports the mean with a bootstrap CI, the median, the share of PAs above their copy, and a Wilcoxon test (Holm across biomes).
- Weights biomes equally in "All PAs"; the table adds a PA-weighted row.

### Visual inspection (optional)
- `tools/inspection_export.py` writes the analysed PAs (`data/inspection/analysis_pas.zip`) and a checklist.
- Upload the zip as the asset `analysis_pas`, set `ASSET_ROOT` in `tools/inspection_app.js`, and run the app in the Code Editor.
- Add PAs to exclude to `EXCLUDED_PIDS` and rerun from 01.

## Outputs

**`results/pa_edge_metrics_ndvi.csv`** (04) — one row per PA-year:
- WDPA attributes (name, country, IUCN category, establishment year, biome, …), `MERGED_PID`, `AREA_KM2`, `SHAPE_RATIO`, and transect coverage from 02: `N_TRNST_GEN` (generated, one per 500 m of boundary), `N_TRNST` (kept), `PCT_KEPT`, `MAX_GAP_PCT`.
- Four versions of every metric, by suffix: none (all transects), `_nowater`, `_nootherpa`, `_noflags`:
  - `n_trnst*`: complete transects that year; `pct_complete`: as a % of `N_TRNST`.
  - `edge_extent*`: share of transects that are edges (`pt_0 > min(pt_1, pt_2)` and `pt_0 > min(pt_m1, pt_m2)`). With no boundary effect (five exchangeable values) the expected share is 8/15 ≈ 0.533, not 0.
  - `D02*`, `D01*`, `D0m1*`, `D0m2*`: Cohen's d of the boundary point vs each other position, on edge transects; `edge_intensity*`: their mean. Computed on edge transects only, so positive even with no boundary effect.
  - The validation measures both metrics on displaced copies of each PA's transects: the baseline of the surrounding landscape.
- Covariates (PA means over the year's complete transects): `gHM_mean` (human modification at the two outer points), `elevation_mean` and `terrain_slope_mean` (boundary point), `water_extent_pct` (% of transects with any point in the JRC maximum water extent), `in_other_pa_pct`.
- Per-PA trends are computed by 07 on the analysis set (`data/05_analysis/pa_trends.csv`).

**Other outputs:** `results/pipeline_counts.json` (counts of 01, 02 and 04), `results/analysis_counts.json` (05), `results/tables/` (supplementary tables and figure source tables), `results/figures/`, `results/diagnostics/` (spatial ranges, Moran's I, residuals, …), `results/R_sessionInfo.txt`. Intermediate data are in `data/`, one folder per script (`01_wdpa`, `02_pas`, `02_transects`, `03_gee`, `04_transect_years`, `05_analysis`, `10_map_layers`, `11_validation`, `12_gee`, `13_validation`, `inspection`); coordinates and WDPA polygons stay in `data/`.

## Configuration

Python settings are in [modules/config.py](modules/config.py) (the defaults reproduce the analysis): `START_YEAR`/`END_YEAR` (2001–2025), `BOUNDARY_SPACING` (500 m), `TRANSECT_SPACING` (2,500 m), `POINTS_PER_SIDE` (2), `CHUNK_SIZE` (400 PAs per upload file), `EXCLUDED_PIDS`, `VALIDATION_*`. R settings (analysis set, reference levels, seeds, spatial model, sensitivity, validation) are in [modules/r/config.R](modules/r/config.R).

## Troubleshooting

- **"Input file not found":** a previous script or a manual download/upload step is missing.
- **Script 04 stops on missing, unexpected or duplicate points, or on unrecognised CSVs:** `data/03_gee/` must hold exactly one copy of every part of this run (remove Drive duplicates such as `name (1).csv` and the second copy of the `--test` part). Script 13 checks `data/12_gee/` the same way.
- **Earth Engine sign-in errors:** `earthengine authenticate --auth_mode=notebook --force` (the default `gcloud` mode saves no credentials for the Drive scope).
- **"GEE_PROJECT has not been set":** run `earthengine set_project YOUR-PROJECT-ID`, or set the `GEE_PROJECT` environment variable (project IDs are listed at [console.cloud.google.com](https://console.cloud.google.com/)).

## License
The code in this project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details, except `modules/r/corHaversine.R`, which is adapted from a Stack Overflow answer by Nate Pope (2013) and is shared under CC BY-SA 3.0 (see its header).

**Data:** Protected area boundaries and attributes come from the World Database on Protected Areas (WDPA) and are not covered by the MIT License. They are subject to the [Protected Planet Terms and Conditions](https://www.protectedplanet.net/en/legal): the data may not be redistributed, sub-licensed (including within derivative works), or put to commercial use (including derivative works) without prior written permission from UNEP-WCMC. Obtain the data from [Protected Planet](https://www.protectedplanet.net) or Earth Engine (`WCMC/WDPA/202508/polygons`), and cite it as:

> UNEP-WCMC and IUCN (2025), Protected Planet: The World Database on Protected Areas (WDPA) [On-line], August 2025, Cambridge, UK: UNEP-WCMC and IUCN. Available at: www.protectedplanet.net.
