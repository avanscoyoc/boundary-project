# Settings for the statistical analysis (scripts 05–10 and 14). No data are read here: scripts
# pass these values to the functions in modules/r/, which tests/r/ also run on simulated data.
# Paths are relative to the repository root, found from the working directory.

repo_root <- function(start = getwd()) {
  d <- normalizePath(start, mustWork = TRUE)
  while (!file.exists(file.path(d, "modules", "r", "config.R"))) {
    parent <- dirname(d)
    if (parent == d) stop("Run from inside the repository (modules/r/config.R not found).")
    d <- parent
  }
  d
}

CFG <- local({
  root <- repo_root()
  p <- function(...) file.path(root, ...)

  list(
    root = root,

    # ── Paths ────────────────────────────────────────────────────────────────
    paths = list(
      pa_metrics   = p("results", "pa_edge_metrics_ndvi.csv"),   # script 04: one row per PA-year
      pas_selected = p("data", "02_pas", "pas_selected.gpkg"),    # script 02: PA polygons (EPSG:4326)
      missing_transects = p("data", "04_transect_years", "missing_transects.csv"),   # script 04: transects with missing years
      analysis_dir = p("data", "05_analysis"),                    # script 05 outputs
      cache_dir    = p("data", "05_analysis", "cache"),           # distance set-ups and fitted models
      tables       = p("results", "tables"),
      figures      = p("results", "figures"),
      diagnostics  = p("results", "diagnostics"),                 # outputs not reported in the paper
      counts       = p("results", "analysis_counts.json"),
      session_info = p("results", "R_sessionInfo.txt"),
      # Fig. 2 map layers (script 10); kept under data/ as WDPA polygons may not be redistributed
      map_layers   = p("data", "10_map_layers")
    ),

    # ── Record and analysis set ──────────────────────────────────────────────
    years     = 2001:2025,
    n_span    = 24L,          # years spanned by the record (2025 − 2001): change = slope × n_span
    min_years = 5L,           # min. complete years of both metrics per PA
    transects = c("all", "nowater", "nootherpa", "noflags"),      # transect versions; "all" is the main analysis
    exclude_biomes = c("Mangrove", "Rock & Ice"),
    # Drop PAs whose boundary is mostly sea or open water (%; a data rule, separate from 02's
    # coverage rule). Sea transects have no MODIS data, so a PA needs a median share of complete
    # transects (04 pct_complete; 0% if none) ≥ min_complete_pct, and ≤ max_water_pct of its
    # transects with JRC water (04 water_extent_pct, mean over years).
    min_complete_pct = 75,
    max_water_pct    = 75,

    # Biome order in tables and figures (groups from 04's recategorize_biome): by vegetation
    # structure, then temperature. Model reference levels are set in `ref`.
    biome_order = c("Tropical Grassland", "Temperate & Montane Grassland", "Tundra",
                    "Desert", "Mediterranean",
                    "Tropical Forest", "Temperate Forest", "Boreal Forest"),

    # IUCN categories → three groups
    iucn_groups = list(
      "strict"     = c("Ia", "Ib", "II"),
      "non-strict" = c("III", "IV", "V", "VI"),
      "unknown"    = c("Not Applicable", "Not Assigned", "Not Reported")
    ),

    # ── Reference levels, shared by all models (06, 08, 09) ──────────────────
    ref = list(iucn = "non-strict", biome = "Boreal Forest"),

    # ── Trends ───────────────────────────────────────────────────────────────
    trend = list(
      fdr    = 0.05,                                  # Benjamini–Hochberg across PAs, per metric
      levels = c("increase", "no_trend", "decrease")  # trend classes of 07 (pa_trends.csv trend_*)
    ),

    # ── Analysis data (05) ───────────────────────────────────────────────────
    # Responses: the yearly metric columns of script 04 (PA means are <col>_mean).
    responses = c(extent = "edge_extent", intensity = "edge_intensity"),
    # Model covariates, z-scored over the analysis set: z column = source column (04 names;
    # terrain_slope_mean is renamed terrain_slope). Names must match models.R.
    covariates = c(STATUS_YR_z = "STATUS_YR", log_area_z = "AREA_KM2", elevation_mean_z = "elevation_mean",
                   terrain_slope_z = "terrain_slope", water_extent_pct_z = "water_extent_pct",
                   gHM_mean_z = "gHM_mean"),
    log_covariates = "log_area_z",   # z of log(AREA_KM2), the geodesic polygon area
    # PA coordinates: point-on-surface computed in Mollweide, then transformed to EPSG:4326
    coords_crs = "+proj=moll +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs",

    # ── Establishment-year imputation (STATUS_YR == 0) ───────────────────────
    impute = list(
      draws       = 1000L,    # draws per PA; the median is used
      min_country = 5L,       # fewer known years in the country → global pool
      seed        = 20261107L
    ),

    # ── Bootstraps (percentile CIs) ──────────────────────────────────────────
    # Fixed seeds and block order make the bootstrap CIs reproducible.
    boot = list(
      R     = 10000L,
      seeds = c(biome = 42L, weighted = 43L, sen = 44L, pooled = 45L)
    ),
    pooled_label = "All PAs",   # the all-PAs row of the biome summaries

    # ── Fig. 4B case studies (07 tables, 10 figures and the KML) ─────────────
    # One PA per panel, top to bottom. `label`: panel title; `expect`: the extent/intensity trend
    # classes the panel illustrates. The PAs were chosen after seeing the results, so check 07's
    # warnings after a rerun (07 warns if `expect` differs, stops if a PA is no longer analysed).
    # Pô-Nazinga-Sissili includes Sissili (28556), merged into it by script 02.
    case_studies = data.frame(
      WDPA_PID = c("555705201", "36531", "34032", "19667"),
      label    = c("Pô-Nazinga-Sissili, Burkina Faso · Trop. Grass.",
                   "Zala, Nigeria · Trop. Grass.",
                   "Igarapé Ribeirão, Brazil · Trop. For.",
                   "Namtok Chat Trakan, Thailand · Trop. For."),
      expect   = c("increase/increase", "decrease/decrease", "increase/increase", "no_trend/no_trend"),
      stringsAsFactors = FALSE),

    # ── Spatial GLS ──────────────────────────────────────────────────────────
    spatial = list(
      method          = "REML",
      range_start_km  = 200,    # optimiser start for the range of corHaversine(mimic = "corExp")
      nugget          = FALSE,
      max_iter        = 200L,   # glsControl(maxIter, msMaxIter)
      jitter_deg      = 1e-4,   # only for PAs with identical coordinates (zero distances stop corHaversine)
      jitter_seed     = 1L,
      jitter_tries    = 10L,
      moran_nsim      = 999L,
      moran_seed      = 1L,
      k_nn            = 8L,
      bands_km        = c(0, 50, 100, 250, 500, 1000, 2000)
    ),

    # ── Validation: displaced transects (14) ─────────────────────────────────
    # Displaced-copy metrics come from Python scripts 11–13 (data/13_validation/).
    validation = list(
      pa_years = p("data", "13_validation", "validation_displaced_pa_years.csv"),
      log      = p("data", "13_validation", "validation_displacement_log.csv"),
      seed     = 47L,
      chance_extent = 8 / 15   # edge rate if the five values were exchangeable (no structure)
    ),

    # ── Sensitivity (09) ─────────────────────────────────────────────────────
    sensitivity = list(
      established_by = 2001L,   # PA subset (a): analysed establishment year ≤ this (the record's first year)
                                # PA subset (b): reported year only (STATUS_YR_imputed == FALSE)
      # Transect subsets: a PA-year counts only with ≥ min_transects complete transects (the
      # fewest of any analysed PA, analysis_counts.json final_set); compared PAs need
      # ≥ min_years such years of both metrics in every version.
      min_transects = 60L,
      # PA subset (c): PAs keeping ≥ min_pct_kept % of the transects 02 generated (PCT_KEPT); 02's
      # coverage rule caps only the longest unsampled stretch, so some PAs keep few transects.
      min_pct_kept  = 50,
      seed          = 46L   # bootstrap CIs of the paired differences
    )
  )
})
