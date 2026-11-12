# Protected Area Edge Analysis

## Overview
This project analyzes change in vegetation contrast at protected area (PA) boundaries using Google Earth Engine and geospatial methods. The workflow selects more than 4,000 terrestrial PAs from the WDPA, generates transects perpendicular to each boundary, samples the gradient magnitude of annual median NDVI (MODIS MOD13A1, 2003–2025) plus water, human modification, and topography at every point, and computes yearly edge metrics and their trends for each PA. The statistical analysis (R) models mean edge extent and intensity and their change, and a validation compares every sampled PA with a displaced copy of its transects in the surrounding landscape.

The data the scripts produce (`data/`, `results/`) are not part of the repository: run the scripts to regenerate them. WDPA boundaries may not be redistributed (see [License](#license)).

## Quick Start

### Workflow Overview
```
01_collect_protected_areas.py      → filter WDPA in Earth Engine → download GeoJSON from Drive
02_generate_sampling_points.py     → choose PAs + build transects → upload zips to Earth Engine
03_sample_satellite_imagery.py     → sample NDVI gradients in Earth Engine → download CSVs from Drive
04_compute_edge_metrics.py         → transect-years, PA-year metrics, trends → final CSV
05_prepare_analysis_data.R         → analysis set, imputed establishment years, z-scores (R)
05b_inspection_export.py + .js     → visual inspection of the analysis set in an Earth Engine app (optional)
06_models_mean.R                   → spatial GLS of mean edge extent and intensity → Tables SX, SY
07_trends.R                        → per-PA trends, standardised change, biome comparison, case studies → Tables SV, SW
08_models_change.R                 → spatial GLS of change → Table SZ
09_sensitivity.R                   → transect subsets and PA subsets → sensitivity tables
10_figures.R                       → Fig. 3, Fig. 4, supplementary and diagnostic figures, Fig. 2 map layers
11_validation_displace_transects.py → displaced copies of sampled PAs' transects → upload zips to Earth Engine
12_validation_sample_imagery.py    → sample the copies in Earth Engine → download CSVs from Drive
13_validation_edge_metrics.py      → accept one copy per PA, its edge metrics
14_validation_compare.R            → each PA against its copy → validation table and figure
```

### Run the pipeline
```bash
# 1. Protected areas (Earth Engine, ~2 h) — or skip and reuse an existing download
python scripts/01_collect_protected_areas.py
# → Download WDPA_polygons.geojson from Google Drive to data/01_wdpa/

# 2. Choose PAs and build transects (local)
python scripts/02_generate_sampling_points.py
# → Upload each data/02_transects/upload/transect_points_NNN.zip as an Earth Engine table asset

# 3. Sample NDVI gradients (Earth Engine, ~5 h) — test one part first
python scripts/03_sample_satellite_imagery.py --test
python scripts/03_sample_satellite_imagery.py
# → Download all CSVs from Drive folder ndvi_gradient to data/03_gee/

# 4. Edge metrics and trends (local)
python scripts/04_compute_edge_metrics.py
# → results/pa_edge_metrics_ndvi.csv

# 5–10. Statistical analysis (R, local; 06, 08 and 09 take hours: run them in the background)
Rscript scripts/05_prepare_analysis_data.R
Rscript scripts/06_models_mean.R
Rscript scripts/07_trends.R
Rscript scripts/08_models_change.R
Rscript scripts/09_sensitivity.R
Rscript scripts/10_figures.R
# → results/tables/, results/figures/, results/diagnostics/, results/analysis_counts.json,
#   data/10_map_layers/ (Fig. 2 layers; WDPA polygons, so kept out of results/)

# 11–14. Validation (after script 05; see Validation below)
python scripts/11_validation_displace_transects.py
# → Upload each zip in data/11_validation/upload/ as an Earth Engine table asset
python scripts/12_validation_sample_imagery.py --test
python scripts/12_validation_sample_imagery.py
# → Download all CSVs from Drive folder ndvi_gradient_displaced to data/12_gee_validation/
python scripts/13_validation_edge_metrics.py
Rscript scripts/14_validation_compare.R
```

> **Reproducing the exact published analysis:** Script 01 reads a date-stamped WDPA release
> (`WCMC/WDPA/202508/polygons`, August 2025). Earth Engine keeps historical monthly releases,
> so re-running script 01 reproduces the input used here. `WDPA_polygons.geojson` is not
> included in this repository because the WDPA terms do not allow redistribution (see [License](#license)).

## Project Structure
```
boundary-project/
├── scripts/
│   ├── 01_collect_protected_areas.py   # Python: data pipeline
│   ├── 02_generate_sampling_points.py
│   ├── 03_sample_satellite_imagery.py
│   ├── 04_compute_edge_metrics.py
│   ├── 05_prepare_analysis_data.R      # R: statistical analysis
│   ├── 05b_inspection_export.py        # visual inspection: upload file + checklist
│   ├── 05b_inspection_app.js           #   Earth Engine app (paste into the Code Editor)
│   ├── 06_models_mean.R
│   ├── 07_trends.R
│   ├── 08_models_change.R
│   ├── 09_sensitivity.R
│   ├── 10_figures.R
│   ├── 11_validation_displace_transects.py   # validation: Python 11–13, R 14
│   ├── 12_validation_sample_imagery.py
│   ├── 13_validation_edge_metrics.py
│   └── 14_validation_compare.R
├── modules/                     # Reusable domain logic
│   ├── config.py                # All parameters and paths (Python)
│   ├── remotesensing.py         # Google Earth Engine functions
│   ├── preprocessing.py         # PA selection and transect generation
│   ├── analysis.py              # Edge detection, PA-year metrics, trends
│   ├── validation.py            # Displaced transects (validation)
│   ├── counts.py                # Writes results/pipeline_counts.json
│   └── r/                       # R analysis: config.R (settings), prep.R, trends.R, spatial.R,
│                                #   corHaversine.R, models.R, plots.R, validation.R
├── tests/                       # Python unit tests (python -m pytest tests/)
│   └── r/                       # R unit tests (see Statistical Analysis)
├── data/                        # Each folder is filled by the script with the same number
│   ├── 01_wdpa/                 # WDPA_polygons.geojson               (download after script 01)
│   ├── 02_pas/                  # pas_selected.gpkg, pas_removed.gpkg
│   ├── 02_transects/            # transect_points.parquet
│   │   └── upload/              # transect_points_NNN.zip             (upload to Earth Engine)
│   ├── 03_gee/                  # ndvi_gradient_NNN_part_N.csv        (download after script 03)
│   ├── 04_transect_years/       # ndvi_NNN.parquet, one row per transect-year
│   ├── 05_analysis/             # pa_level.csv, pa_year.csv, pa_trends.csv, scaling.csv,
│   │                            #   excluded_pas.csv, cache/ (fitted models; coordinates stay here)
│   ├── inspection/              # pas_final.zip (upload to Earth Engine), checklist.csv (05b)
│   ├── 10_map_layers/           # pa_map_layers.gpkg, case_study_pas.kml (Fig. 2; script 10)
│   ├── 11_validation/           # sample, candidates, displaced points; upload/ for Earth Engine
│   └── 12_gee_validation/       # displaced-point CSVs and polygon biomes (download after script 12)
└── results/
    ├── pa_edge_metrics_ndvi.csv # FINAL: one row per PA-year
    ├── pipeline_counts.json     # PA and transect counts at every step (for the methods)
    ├── analysis_counts.json     # analysis-set counts, imputation, PA sets (script 05)
    ├── validation_*.csv         # displaced copies: PA-year metrics, candidate checks, log (script 13)
    ├── tables/                  # supplementary tables and figure source tables (scripts 06–09, 14)
    ├── figures/                 # Fig. 3, Fig. 4 (and 4A, 4B case panels), supplementary figures (scripts 10, 14)
    ├── diagnostics/             # unreported checks: spatial ranges, Moran's I, residuals, …
    └── R_sessionInfo.txt        # R and package versions of the last figure run
```

## Outputs

**`results/pa_edge_metrics_ndvi.csv`** — one row per PA-year:
- PA attributes from the WDPA (name, country, IUCN category, establishment year, biome, …), plus `MERGED_PID`, `AREA_KM2`, `SHAPE_RATIO`
- Transect coverage per PA (script 02): `N_TRNST_GEN` (transects generated, one per 500 m of boundary), `N_TRNST` (kept after the whole-line check), `PCT_KEPT` (% kept), and `MAX_GAP_PCT` (longest run of dropped transects as % of the boundary; 0 = full coverage)
- Four versions of every metric, by suffix: none (all transects), `_nowater` (no transects with water), `_nootherpa` (no transects whose outer points fall in another PA), `_noflags` (neither)
  - `n_trnst*`: complete transects that year (all 5 points have a value)
  - `pct_complete`: `n_trnst` as a % of the PA's transects (`N_TRNST`), to spot years with little data
  - `edge_extent*`: share of transects that are edges (`pt_0 > min(pt_1, pt_2)` and `pt_0 > min(pt_m1, pt_m2)`)
  - `D02*`, `D01*`, `D0m1*`, `D0m2*`: Cohen's d of the boundary point vs each other position, on edge transects
  - `edge_intensity*`: mean of the four Cohen's d values
  - `slope_*`, `p_value_*`: per-PA OLS trend on year (≥ 5 years of data)
  - `p_adj_*`: Benjamini–Hochberg adjusted p-value (across PAs, separately for each metric)
  - `trend_*`: `increase` / `decrease` when `p_adj` < 0.05, otherwise `no_trend`
- Covariates: `gHM_mean`, `elevation_mean`, `terrain_slope_mean`, `water_extent_pct`, `in_other_pa_pct`

**`data/04_transect_years/ndvi_NNN.parquet`** — one row per complete transect-year: the five gradient values (`pt_m2` … `pt_2`), `edge`, and the per-transect flags `water_extent` and `in_other_pa`.

**`results/pipeline_counts.json`** — counts at every PA filter and transect check (script 02) and per-year transect completeness (script 04).

## Setup

### Prerequisites
- [VS Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) running
- A Google account with a Google Cloud project [registered for Earth Engine](https://code.earthengine.google.com/register) (free for noncommercial use). Use your own account and project; no access to the authors' project is needed.

### Initial Setup
1. Clone the repository and open in VS Code:
   ```bash
   git clone https://github.com/avanscoyoc/boundary-project.git
   code boundary-project
   ```
2. When prompted, click **"Reopen in Container"** — VS Code builds the Docker container and installs all pinned dependencies (Python from `requirements.txt` + `constraints.txt`, R from `.devcontainer/pixi.lock`; ~5 minutes).
3. Sign in to Earth Engine with your Google account and set your project. Do this once, in the VS Code terminal:
   ```bash
   earthengine authenticate --auth_mode=notebook
   earthengine set_project YOUR-PROJECT-ID
   ```
   Open the printed link, sign in, and paste the code back into the terminal. Your sign-in and project are saved in a Docker volume on your machine (`boundary-project-earthengine`), not in the repo, so they survive **Rebuild Container**. You do not need to edit `modules/config.py`.
4. Run the scripts in order (see [Detailed Workflow](#detailed-workflow)). Every dataset the scripts read is in the public Earth Engine catalog. The uploads (scripts 02, 05b, 11) go into your project, and the exports (scripts 01, 03, 12) go to your Google Drive.

**Sharing a project (optional).** To let someone else use your project and the assets you already uploaded, add their Google account in Cloud Console → IAM with the roles *Service Usage Consumer* and *Earth Engine Resource Writer*. They still sign in as themselves, then run `earthengine set_project YOUR-PROJECT-ID`. To use a different project for a single run, put `GEE_PROJECT=other-project` before the command.

## Detailed Workflow

### Script 01: Collect Protected Areas (Earth Engine)
```bash
python scripts/01_collect_protected_areas.py
```
Filters WDPA polygons (single-part polygons, ≥ 200 km², terrestrial, designated/established/inscribed; no marine PAs or UNESCO-MAB Biosphere Reserves; 11 PIDs excluded by hand), assigns the RESOLVE biome with the largest overlap, and exports to Google Drive.

**⚠️ MANUAL STEP**: Download `WDPA_polygons.geojson` from Google Drive → `data/01_wdpa/`

> **Note on reproducibility:** The WDPA collection is date-stamped (`WDPA_COLLECTION = "WCMC/WDPA/202508/polygons"`
> in `modules/config.py`). Earth Engine keeps historical monthly releases back to July 2017, so re-running this
> script reproduces the published input. Newer releases (e.g., `WCMC/WDPA/current/polygons`) contain updated
> protected area boundaries and may alter results.

### Script 02: Choose PAs and Generate Transects (local)
```bash
python scripts/02_generate_sampling_points.py
```
Part 1 — choose PAs (equal-area Mollweide):
1. Fill holes smaller than 2.25 km²
2. Duplicates (overlap > 90% of both PAs): keep the oldest. Nested PAs (overlap > 90% of the smaller only): keep the larger
3. Same `ORIG_NAME` + `ISO3` and touching or overlapping: dissolve into one PA, keeping the oldest establishment year (`MERGED_PID` lists the merged PAs)
4. Compute geodesic area, perimeter, and shape ratio (perimeter / √area), for information only; there is no shape filter

Part 2 — transects (each PA in its own azimuthal equidistant projection, so distances are true meters):
1. Remove PAs with no interior at least 5.5 km from the edge
2. Boundary points every 500 m; transects of 5 points 2.5 km apart (2 inner, 1 boundary, 2 outer)
3. Keep a transect only if its inner half lies within the PA and its outer half lies outside it
4. Remove PAs whose longest unsampled stretch of boundary is more than 25% of the boundary (`MAX_GAP_PCT_LIMIT`)
5. Flag transects whose outer points fall inside another PA (`in_other_pa`)
6. Write upload zips of 400 whole PAs each, and every point to `transect_points.parquet`

Removed PAs, with the step, reason, and the PA that replaced them, are in `data/02_pas/pas_removed.gpkg`.

**⚠️ MANUAL STEP**: Upload each `data/02_transects/upload/transect_points_NNN.zip` as its own Earth Engine table asset `transect_points_NNN` (script 02 prints the full list). One asset per file: a single large upload can fail.

### Script 03: Sample Satellite Imagery (Earth Engine)
```bash
python scripts/03_sample_satellite_imagery.py --test   # one part of asset 000
python scripts/03_sample_satellite_imagery.py          # everything (~5 h)
```
Computes the gradient magnitude of each annual median NDVI composite (MOD13A1, 2003–2025) on the native MODIS grid and samples it, with water (JRC GSW max extent), human modification (gHM), elevation and slope (Copernicus GLO-30), at every point. Each asset is exported in parts of 50,000 points.

**⚠️ MANUAL STEP**: Download **all** CSVs from Google Drive folder `ndvi_gradient` → `data/03_gee/`

### Script 04: Compute Edge Metrics (local)
```bash
python scripts/04_compute_edge_metrics.py
```
Processes one asset at a time: checks that every exported point arrived exactly once, builds one row per complete transect-year, computes PA-year metrics for the four transect subsets, merges PA attributes, and adds per-PA trends. It also lists the transects with a missing value in any year (`data/04_transect_years/missing_transects.csv`), which script 05 summarises for the analysis set in `analysis_counts.json`. See [Outputs](#outputs).

## Statistical Analysis (R)

Scripts 05–10 implement the statistical analysis in R (4.5.3, installed in the container with pixi; packages pinned in `.devcontainer/pixi.lock`). Settings are in [modules/r/config.R](modules/r/config.R); the functions are in `modules/r/`, and each script writes its own tables, which the figure script reads.

- **05** builds the analysis set from `results/pa_edge_metrics_ndvi.csv`: drops the Mangrove and Rock & Ice biomes, PAs whose boundary is dominated by sea (fewer than 75% of transects complete in a typical year) or by open water (more than 75% of transects with surface water), and PAs missing a covariate or with fewer than 5 years of either metric (`excluded_pas.csv` lists them with the reason), and, when `inspection$apply = TRUE`, the PAs marked `exclude` in the visual-inspection checklist. It imputes missing establishment years (median of 1,000 draws from the country's known years), z-scores the covariates, and computes point-on-surface coordinates.
- **06** fits spatial GLS (exponential correlation in great-circle distance, REML) of mean edge extent and intensity: main-effects and HMI-interaction models (reference levels non-strict / Boreal Forest, as for every model).
- **07** computes per-PA OLS trends with Benjamini–Hochberg classes (Yue–Pilon as a check), standardised change, bootstrap biome means and the biome GLS comparison; the lag-1 autocorrelation of each PA's trend residuals (a diagnostic); and the series and % change of the Fig. 4B case-study PAs (`case_studies` in the config).
- **08** fits spatial GLS of standardised change (reference non-strict / Boreal Forest), unweighted and precision-weighted.
- **09** repeats the headline results on the transect subsets (PA-years with at least 60 complete transects, same PAs as all transects) and on PAs established by 2003 and PAs with a reported year, with the spatial range fixed at the main estimates.
- **10** draws the figures; Fig. 3 shows an interaction panel only where its Wald test is significant. Fig. 4 is panel A (change by biome) over panel B (the case-study series beside grey placeholders for the 2003 and 2025 satellite images, added in the layout). It also writes the Fig. 2 map layers (a GeoPackage of the analysed PAs with their means, trends and change, and a KML of the case studies), which are styled outside the repository.

Scripts 06, 08 and 09 take `--pilot N` (a random subset of PAs, for a quick check) and `--cores K`. At full size each spatial fit takes about an hour and about 1.5 GB of memory, so run 06, 08 and 09 in the background and one at a time on a machine with 8 GB. Fits are cached in `data/05_analysis/cache/` by content, so a rerun only refits what changed.

**Tests.** `Rscript -e 'testthat::test_dir("tests/r")'` runs the R unit tests; `python -m pytest tests/` runs the Python tests.

## Visual inspection (script 05b)

Every analysed PA can be checked on satellite imagery:
1. `python scripts/05b_inspection_export.py` writes `data/inspection/pas_final.zip` (the analysed PAs in review order: biome, country, name) and `data/inspection/checklist.csv` (one row per PA; it never overwrites a checklist that has entries).
2. Upload the zip as the Earth Engine table asset `pas_final`, set `ASSET_ROOT` in `scripts/05b_inspection_app.js` to your project's asset folder, then paste `scripts/05b_inspection_app.js` into the Code Editor and run it (or publish it as an app). It shows one PA at a time with its outline, its transect points (interior, boundary, exterior) and an info panel; the page address keeps your place.
3. Record decisions in the checklist (`decision`: blank = keep, `exclude`, or `check` while undecided; `note`). The app number is the checklist's `order`.
4. When done, set `inspection$apply = TRUE` in `modules/r/config.R` and rerun 05 → 07 → 06 → 08 → 09 → 10, then the validation (11 → 14). Script 05 removes the excluded PAs as its last counted step and stops if any PA is still marked `check`.

## Validation (displaced transects; scripts 11–14)

The validation tests whether the edge signal at PA boundaries exceeds what the same transects record in the surrounding landscape.
- **Sample (script 11, local):** 50 analysed PAs per biome, drawn at random with 25 reserves in draw order. Each PA's whole transect set is translated by a random vector (no rotation; distance uniform between d and 2d, where d is the larger of 50 km and the PA's footprint; at most 10% of displaced transects with a PA boundary within 2.5 km of their centre point), giving up to five candidate copies per PA. The PA polygon is moved with its transects. Script 11 writes upload zips of the displaced points and polygons.
- **Sampling (script 12, Earth Engine):** the copies are sampled exactly as the original transects (script 03's image), and each displaced polygon gets its biome by largest area, as in script 01.
- **Acceptance and metrics (script 13, local):** each copy is measured with script 04's functions and must pass the rules of an analysed PA: the displaced polygon keeps the PA's biome, and the sea, lake and five-year rules of script 05. Per PA the first passing candidate in draw order is used; reserves replace PAs with none.
- **Comparison (script 14, R):** the paired difference, PA minus copy, in the 2003–2025 mean of each edge metric, per biome and for all validation PAs: the mean with a 95% percentile-bootstrap CI (10,000 resamples), the median, the share of PAs above their copy, and a paired Wilcoxon signed-rank test (Holm-adjusted across biomes). "All PAs" weights every biome equally; the table also gives the estimate with biomes weighted by their share of the analysed PAs.

Outputs: `results/tables/TableS_validation.csv`, `results/figures/figS_validation.{pdf,png}` (filled symbols: Wilcoxon p < 0.05), `results/diagnostics/validation_paired_pa.csv` (one row per PA) and `validation_attrition.csv` (sampled PAs by outcome), and script 13's `results/validation_*.csv`. Settings: `VALIDATION_*` in [modules/config.py](modules/config.py) and `validation` in [modules/r/config.R](modules/r/config.R). The `--test` run of script 12 exports part 0 of the first asset again; keep one copy of it in `data/12_gee_validation/`.

## Configuration

Python settings are in [modules/config.py](modules/config.py); the defaults reproduce the analysis. The Earth Engine project (`GEE_PROJECT`) is not set in the file: it comes from `earthengine set_project` or the `GEE_PROJECT` environment variable (see [Setup](#initial-setup)).

R analysis settings (analysis set, reference levels, bootstrap seeds, spatial model, sensitivity, validation) are in [modules/r/config.R](modules/r/config.R).

Main Python settings:
- `START_YEAR`, `END_YEAR`: Analysis period (default: 2003–2025)
- `BOUNDARY_SPACING`: Distance between boundary points (default: 500 m)
- `TRANSECT_SPACING`: Distance between points along a transect (default: 2,500 m)
- `POINTS_PER_SIDE`: Points on each side of the boundary (default: 2)
- `CHUNK_SIZE`: PAs per upload file (default: 400)
- `RUN_TAG`: Optional tag added to Earth Engine asset, Drive folder and upload/export file names (default: none, e.g. `transect_points_000`); set one (e.g. `run2`) for a new full re-run so old and new outputs never mix
- `VALIDATION_*`: validation sample size, displacement rules and seed

## Troubleshooting

**"Input file not found"**
- Ensure previous scripts completed successfully
- Check that manual download/upload steps were completed

**"No CSV files found"** or **"expected points missing"** (script 04)
- Download every CSV from the Drive folder `ndvi_gradient` into `data/03_gee/`
- Script 04 compares the CSVs with `transect_points.parquet` and reports missing or duplicate points

**"duplicate points in the CSVs"** (script 04) or **"candidates do not have all their points"** (script 13)
- A CSV part is missing or was downloaded twice (e.g. the `--test` part and the full run); keep exactly one copy of every part

**GEE authentication errors**
```bash
earthengine authenticate --auth_mode=notebook --force
```
Open the link, sign in, and paste the code into the terminal. Then re-run the script. Use `notebook` mode: the default `gcloud` mode warns that the Google Drive scope is blocked and saves no credentials.

**"GEE_PROJECT has not been set"**
- Run `earthengine set_project YOUR-PROJECT-ID` (saved with your credentials), or set the `GEE_PROJECT` environment variable
- Your project ID is visible at [console.cloud.google.com](https://console.cloud.google.com/)

## Legacy Workflow

The original workflow in `archive/src/` remains available for reference.

## License
The code in this project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Data:** Protected area boundaries and attributes come from the World Database on Protected Areas (WDPA) and are not covered by the MIT License. They are subject to the [Protected Planet Terms and Conditions](https://www.protectedplanet.net/en/legal): the data may not be redistributed, sub-licensed (including within derivative works), or put to commercial use (including derivative works) without prior written permission from UNEP-WCMC. Obtain the data from [Protected Planet](https://www.protectedplanet.net) or Earth Engine (`WCMC/WDPA/202508/polygons`), and cite it as:

> UNEP-WCMC and IUCN (2025), Protected Planet: The World Database on Protected Areas (WDPA) [On-line], August 2025, Cambridge, UK: UNEP-WCMC and IUCN. Available at: www.protectedplanet.net.
