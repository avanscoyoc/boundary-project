# Protected Area Edge Analysis

This project analyzes change in vegetation contrast at protected area (PA) boundaries using Google Earth Engine and geospatial methods. The workflow selects more than 4,000 terrestrial PAs from the WDPA, generates transects perpendicular to each boundary, samples the gradient magnitude of annual median NDVI (MODIS MOD13A1, 2003–2025) plus water, human modification, and topography at every point, and computes yearly edge metrics for each PA. The statistical analysis (R) models mean edge extent and intensity and their change, and a validation compares every sampled PA with a displaced copy of its transects in the surrounding landscape.

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
# 01 Protected areas (Earth Engine, ~2 h)
python scripts/01_collect_protected_areas.py
#  → download WDPA_polygons.geojson from Google Drive to data/01_wdpa/

# 02 Choose PAs and build transects (local)
python scripts/02_generate_sampling_points.py
#  → upload each data/02_transects/upload/transect_points_NNN.zip as the table asset transect_points_NNN
#    (one asset per file; script 02 prints the list)

# 03 Sample NDVI gradients (Earth Engine, ~5 h); --test runs one part of asset 000 first
python scripts/03_sample_satellite_imagery.py --test
python scripts/03_sample_satellite_imagery.py
#  → download every CSV of the Drive folder ndvi_gradient to data/03_gee/ (one copy of each part)

# 04 Edge metrics (local) → results/pa_edge_metrics_ndvi.csv
python scripts/04_compute_edge_metrics.py

# 05–10 Statistical analysis (R; 06, 08 and 09 take hours: run them in the background, one at a time)
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
#  → download every CSV of the Drive folder ndvi_gradient_displaced to data/12_gee/
python scripts/13_validation_edge_metrics.py
Rscript scripts/14_validation_compare.R
```

Scripts 06, 08 and 09 take `--cores K` (fits run in parallel, default 2). At full size each spatial fit takes about an hour and 1.5 GB of memory (8 GB machine: one script at a time). Fits are cached in `data/05_analysis/cache/` by content, so a rerun only refits what changed. **Tests:** `python -m pytest tests/` and `Rscript -e 'testthat::test_dir("tests/r")'`.

## What each script does

**01 Collect PAs (Earth Engine).** Filters the WDPA: terrestrial; no Marine Protected Areas or UNESCO-MAB Biosphere Reserves; designated, established or inscribed; ≥ 200 km²; not one of the 32 PIDs excluded by hand in `EXCLUDED_PIDS` (21 of them after the visual inspection: 19 circular polygons and 2 city PAs); single-part polygons. Counts the PAs each filter removes (`results/pipeline_counts.json`, section `01_wdpa`), assigns the RESOLVE Ecoregions 2017 biome with the largest overlap, and exports GeoJSON to Drive.
The WDPA release is pinned (`WDPA_COLLECTION = "WCMC/WDPA/202508/polygons"`, August 2025): Earth Engine keeps monthly releases, so rerunning 01 reproduces the input; newer releases may change the results. If the release is ever removed, 01 stops with a message; a `WDPA_polygons.geojson` exported earlier from the same release can be used instead, because 02 applies `EXCLUDED_PIDS` again.

**02 Choose PAs and build transects (local).** Part 1 (equal-area Mollweide):
1. Fill holes smaller than 2.25 km².
2. Duplicates (overlap > 90% of both PAs): keep the oldest. Nested PAs (> 90% of the smaller only): keep the larger.
3. Same `ORIG_NAME` + `ISO3`, touching or overlapping: dissolve into one PA (`MERGED_PID` lists the members). It keeps the `WDPA_PID` of its oldest member and the oldest non-zero establishment year; its other attributes (name, IUCN category, `WDPAID`, biome, …) come from the largest member (`ATTR_PID`).
4. Geodesic area, perimeter and shape ratio, for information (no shape filter).

Part 2, each PA in its own azimuthal equidistant projection (true meters):
1. Remove PAs with no interior at least 5.5 km from the edge.
2. Boundary points every 500 m along the outer ring; transects of 5 points 2.5 km apart (2 inner, 1 boundary, 2 outer), perpendicular to the boundary.
3. Keep a transect only if its inner half lies within the PA and its outer half outside it.
4. Remove PAs whose longest unsampled stretch is more than 25% of the boundary (`MAX_GAP_PCT_LIMIT`). The rule caps the longest gap only, so a few PAs keep a small share of their transects (`PCT_KEPT`; script 09 has a sensitivity run without them).
5. Flag transects whose outer points fall inside another PA (`in_other_pa`). "Another PA" is one of script 01's polygons (≥ 200 km², single-part, designated): smaller, multi-part or proposed PAs are not flagged.
6. Write upload zips of 400 whole PAs each, and every point (with its upload file number) to `transect_points.parquet`.

Removed PAs, with the step, the reason and the PA that replaced them, are in `data/02_pas/pas_removed.gpkg`.

**03 Sample imagery (Earth Engine).** The gradient magnitude of each year's median NDVI composite (MOD13A1, all 16-day composites of the year; no quality, cloud or snow masking), on the native MODIS grid (~463 m), plus water (JRC Global Surface Water 1.4, maximum extent), human modification (gHM, 2016) and elevation and slope (Copernicus GLO-30, 2024 release), at every point; exported in parts of 50,000 points.

**04 Edge metrics (local).** Checks that every exported point arrived exactly once, builds one row per complete transect-year (all 5 points have a value), computes PA-year metrics for four transect versions, merges PA attributes, and groups the biomes (`modules/biome_groups.csv`, read by Python and R): 8 groups by vegetation structure and temperature (Tropical Grassland, with flooded grasslands; Temperate & Montane Grassland; Tundra; Desert; Mediterranean; Tropical Forest; Temperate Forest; Boreal Forest), plus Mangrove and RESOLVE's unclassified rock and ice ("N/A", Rock & Ice), which 05 removes. Also lists the transects with a missing value in any year (`data/04_transect_years/missing_transects.csv`).

**R statistics (05–10)** — settings in [modules/r/config.R](modules/r/config.R), functions in `modules/r/`; each script writes its own tables, which 10 reads.
- **05** builds the analysis set: drops Mangrove and Rock & Ice, PAs whose boundary is dominated by sea (fewer than 75% of transects complete in a typical year) or by open water (more than 75% of transects with surface water), and PAs missing a covariate or with fewer than 5 years of either metric (`excluded_pas.csv`). Imputes missing establishment years (median of 1,000 draws from the country's known years), z-scores the covariates, and computes point-on-surface coordinates.
- **06** spatial GLS (exponential correlation in great-circle distance, REML) of mean edge extent and intensity: main effects and HMI interactions. Reference levels: non-strict / Boreal Forest, in every model.
- **07** per-PA OLS trends with Benjamini–Hochberg classes (Yue–Pilon as a check), standardised change, bootstrap biome means of change (Fig. 4A; descriptive: the bootstrap treats PAs as independent), the lag-1 autocorrelation of trend residuals (diagnostic), and the Fig. 4B case-study series (`case_studies` in the config).
- **08** spatial GLS of standardised change, unweighted and precision-weighted (Table SZ), and the biome comparison of change with the same spatial correlation (Table SW: pairwise contrasts, Holm, and letters).
- **09** repeats the headline results on the transect subsets (same PAs; PA-years with at least 60 complete transects in every run) and on three PA subsets: established by 2003, reported establishment year, and keeping ≥ 50% of their transects; spatial ranges fixed at the main estimates.
- **10** figures (Fig. 3 shows an interaction panel only where its Wald test is significant; Fig. 4 = A, change by biome, over B, case-study series beside placeholders for satellite images) and the Fig. 2 map layers (GeoPackage, KML; styled outside the repository).

**Validation (11–14).** Does the edge signal at PA boundaries exceed what the same transects record in the surrounding landscape?
- **11 (local):** 50 analysed PAs per biome group (400 in all), drawn at random with 25 reserves in draw order. Each PA's whole transect set is translated by one random vector (no rotation; distance uniform between d and 2d, d = the larger of 50 km and the PA's footprint), with at most 10% of displaced transects having a PA boundary within 2.5 km of their centre point. Like `in_other_pa`, this boundary check knows only script 01's PAs. Up to five candidate copies per PA; the PA polygon moves with its transects.
- **12 (Earth Engine):** the copies are sampled with script 03's image, and each moved polygon gets its biome as in 01.
- **13 (local):** each copy is measured with script 04's functions and must pass the rules of an analysed PA (same biome; sea, lake and five-year rules of 05). Per PA the first passing candidate is used; reserves replace PAs with none.
- **14 (R):** the paired difference PA − copy in the 2003–2025 mean of each metric, per biome and for all validation PAs: mean with a 95% percentile-bootstrap CI, median, share of PAs above their copy, paired Wilcoxon signed-rank test (Holm across biomes). "All PAs" weights every biome equally; the table also gives the estimate weighted by the biomes' shares of the analysed PAs.

**Visual inspection (optional, after the analysis).** `python scripts/inspection_export.py` writes `data/inspection/pas_final.zip` (the analysed PAs in review order: biome, country, name) and `checklist.csv` (never overwritten once it has entries). Upload the zip as the table asset `pas_final`, set `ASSET_ROOT` in `scripts/inspection_app.js`, and paste the app into the Code Editor (or publish it): one PA at a time with its outline, transect points and an info panel. Record decisions in the checklist (`decision`: blank = keep, `exclude`, `check`; `note`); add PAs to exclude to `EXCLUDED_PIDS` in `modules/config.py` (with a comment) and rerun from script 01.

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

Python settings are in [modules/config.py](modules/config.py) (the defaults reproduce the analysis): `START_YEAR`/`END_YEAR` (2003–2025), `BOUNDARY_SPACING` (500 m), `TRANSECT_SPACING` (2,500 m), `POINTS_PER_SIDE` (2), `CHUNK_SIZE` (400 PAs per upload file), `EXCLUDED_PIDS`, `VALIDATION_*`. R settings (analysis set, reference levels, seeds, spatial model, sensitivity, validation) are in [modules/r/config.R](modules/r/config.R).

## Troubleshooting

- **"Input file not found":** a previous script or a manual download/upload step is missing.
- **Script 04 stops on missing, unexpected or duplicate points, or on unrecognised CSVs:** `data/03_gee/` must hold exactly one copy of every part of this run (remove Drive duplicates such as `name (1).csv` and the second copy of the `--test` part). Script 13 checks `data/12_gee/` the same way.
- **Earth Engine sign-in errors:** `earthengine authenticate --auth_mode=notebook --force` (the default `gcloud` mode saves no credentials for the Drive scope).
- **"GEE_PROJECT has not been set":** run `earthengine set_project YOUR-PROJECT-ID`, or set the `GEE_PROJECT` environment variable (project IDs are listed at [console.cloud.google.com](https://console.cloud.google.com/)).

## License
The code in this project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Data:** Protected area boundaries and attributes come from the World Database on Protected Areas (WDPA) and are not covered by the MIT License. They are subject to the [Protected Planet Terms and Conditions](https://www.protectedplanet.net/en/legal): the data may not be redistributed, sub-licensed (including within derivative works), or put to commercial use (including derivative works) without prior written permission from UNEP-WCMC. Obtain the data from [Protected Planet](https://www.protectedplanet.net) or Earth Engine (`WCMC/WDPA/202508/polygons`), and cite it as:

> UNEP-WCMC and IUCN (2025), Protected Planet: The World Database on Protected Areas (WDPA) [On-line], August 2025, Cambridge, UK: UNEP-WCMC and IUCN. Available at: www.protectedplanet.net.
