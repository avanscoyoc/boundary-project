# Settings for the statistical analysis (scripts 05–10).
#
# Settings only: no data are read here. Scripts source this file and pass the values they
# need to the functions in modules/r/ explicitly, so the same functions run on the old data
# in the parity tests (tests/r/parity/legacy_inputs.R) and on v2 in the scripts.
# Paths are relative to the repository root (found from the working directory).

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
      diagnostics  = p("results", "diagnostics"),                 # outputs not reported in the paper (D4)
      counts       = p("results", "analysis_counts.json"),
      session_info = p("results", "R_sessionInfo.txt"),
      # Fig. 2 map layers (script 10): WDPA polygons may not be redistributed, so they stay
      # under data/, like the coordinates
      map_layers   = p("data", "10_map_layers"),
      parity_dir   = p("data", "parity"),
      legacy_root  = p("PA_New_Analysis")                         # collaborator's project (read-only)
    ),

    # ── Record and analysis set ──────────────────────────────────────────────
    years     = 2003:2025,
    n_span    = 22L,          # years spanned by the record (2025 − 2003): change = slope × n_span
    min_years = 5L,           # a PA needs ≥ 5 complete years of BOTH metrics (Stats ¶1)
    transects = c("all", "nowater", "nootherpa", "noflags"),      # transect versions (A2); "all" is the main analysis
    exclude_biomes = c("Mangrove", "Rock & Ice"),
    # PAs whose boundary is dominated by sea or open water (user, 2026-11-08). Sea: transects reaching
    # into the sea have no MODIS data and are never complete, so a PA needs a median share of
    # complete transects (04 pct_complete, median over its years) ≥ min_complete_pct; a PA of 02
    # with no complete transect in any year counts as 0% (05, `add_pas_without_metrics()`). This is
    # about data, not geometry: it is separate from 02's coverage rule (the longest stretch of
    # boundary without transects ≤ 25% of the perimeter, MAX_GAP_PCT_LIMIT). Lakes and wetlands:
    # a PA needs ≤ max_water_pct of its transects with JRC water (04 water_extent_pct, the PA's
    # mean over years).
    min_complete_pct = 75,
    max_water_pct    = 75,
    # Visual inspection (Finish plan Step 3; checklist written by script 05b). Set apply = TRUE
    # when the review is finished: 05 then removes the PAs marked "exclude" as its last counted
    # step ("removed after visual inspection") and stops on any PA still marked "check".
    # FALSE: the checklist is not read and the set is unchanged.
    inspection = list(checklist = p("data", "inspection", "checklist.csv"), apply = FALSE),

    # Biome order for tables and figures only. Reference levels are set separately (B2).
    biome_order = c("Grassland & Shrubland", "Tropical Forest", "Temperate Forest",
                    "Boreal Forest", "Desert", "Tundra"),

    # IUCN categories → three groups (00_setup / 01_data_prep)
    iucn_groups = list(
      "strict"     = c("Ia", "Ib", "II"),
      "non-strict" = c("III", "IV", "V", "VI"),
      "unknown"    = c("Not Applicable", "Not Assigned", "Not Reported")
    ),

    # ── Reference levels (D1): one pair for every model (user, 2026-11-10) ───
    # 06, 08, 09; Fig. 3A and Tables SX, SY, SZ. As the collaborator's Fig. 3A contrasts.
    ref = list(iucn = "non-strict", biome = "Boreal Forest"),

    # ── Trends (Stats ¶4–5) ──────────────────────────────────────────────────
    trend = list(
      fdr    = 0.05,                                  # Benjamini–Hochberg across PAs, per metric
      levels = c("increase", "no_trend", "decrease")  # the same codes as Python's trend_* columns (A9)
    ),

    # ── Analysis data (05) ───────────────────────────────────────────────────
    # Responses: the yearly metric columns of script 04 (PA means are <col>_mean).
    responses = c(extent = "edge_extent", intensity = "edge_intensity"),
    # Model covariates, z-scored over the analysis set: z column = source column (raw names as in
    # 04, terrain_slope_mean renamed terrain_slope, A9). log_area_z = z of log(AREA_KM2), the
    # geodesic area of the selected polygon (old: log(AREA_DISSO)). Names must match models.R.
    covariates = c(STATUS_YR_z = "STATUS_YR", log_area_z = "AREA_KM2", elevation_mean_z = "elevation_mean",
                   terrain_slope_z = "terrain_slope", water_extent_pct_z = "water_extent_pct",
                   gHM_mean_z = "gHM_mean"),
    log_covariates = "log_area_z",
    # Point-on-surface is computed in Mollweide, then transformed to EPSG:4326 (as the old 03)
    coords_crs = "+proj=moll +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs",

    # ── Establishment-year imputation (STATUS_YR == 0) ───────────────────────
    impute = list(
      draws       = 1000L,    # draws per PA; the median is used
      min_country = 5L,       # fewer known years in the country → global pool
      seed        = 20261107L
    ),

    # ── Bootstraps (percentile CIs) ──────────────────────────────────────────
    # The old seeds and block order are kept (02_trend_analysis.R), so the parity tests
    # reproduce the old CIs exactly.
    boot = list(
      R     = 10000L,
      seeds = c(biome = 42L, weighted = 43L, sen = 44L, pooled = 45L)
    ),
    pooled_label = "All PAs",   # the all-PAs row of the biome summaries (old tables: "All biomes")

    # ── Fig. 4B case studies (07 tables, 10 figures and the KML; user, 2026-11-11) ─
    # One PA per panel, top to bottom. `label`: panel title; `expect`: the trend classes
    # (extent/intensity) the panel illustrates (07 warns if they differ). Kept from the old
    # figure where the v2 classes match; Sissili (28556) was merged into the Pô-Nazinga-Sissili
    # complex by script 02, and Gougoun (6950, now no trend) is replaced.
    case_studies = data.frame(
      WDPA_PID = c("555705201", "36531", "34032", "19667"),
      label    = c("Pô-Nazinga-Sissili, Burkina Faso · Grassland",
                   "Zala, Nigeria · Grassland",   # user, 2026-11-11 ("for now")
                   "Igarapé Ribeirão, Brazil · Trop. For.",
                   "Namtok Chat Trakan, Thailand · Trop. For."),
      expect   = c("increase/increase", "decrease/decrease", "increase/increase", "no_trend/no_trend"),
      stringsAsFactors = FALSE),

    # ── Spatial GLS (Stats ¶2–3, ¶6) ─────────────────────────────────────────
    spatial = list(
      method          = "REML",
      range_start_km  = 200,    # optimiser start for the range of corHaversine(mimic = "corExp")
      nugget          = FALSE,
      max_iter        = 200L,   # glsControl(maxIter, msMaxIter)
      earth_radius_km = 6371,
      jitter_deg      = 1e-4,   # only for PAs with identical coordinates (zero distances stop corHaversine)
      jitter_seed     = 1L,
      jitter_tries    = 10L,
      moran_nsim      = 999L,
      moran_seed      = 1L,
      k_nn            = 8L,
      bands_km        = c(0, 50, 100, 250, 500, 1000, 2000)
    ),

    # Seed of the random PA subset of a pilot run (scripts 06, 08, 09 --pilot N)
    pilot_seed = 1L,

    # ── Validation: displaced transects (14; draft for review) ───────────────
    # Displaced-copy metrics come from Python scripts 11–13 (results/validation_*.csv).
    validation = list(
      pa_years = p("results", "validation_displaced_pa_years.csv"),
      log      = p("results", "validation_displacement_log.csv"),
      seed     = 47L,
      chance_extent = 8 / 15   # edge rate if the five values were exchangeable (no structure)
    ),

    # ── Sensitivity (09) ─────────────────────────────────────────────────────
    sensitivity = list(
      established_by = 2003L,   # PA subset (a): analysed establishment year ≤ 2003
                                # PA subset (b): reported year only (STATUS_YR_imputed == FALSE)
      # Transect subsets: a PA-year counts in a version only with ≥ min_transects complete
      # transects (the fewest transects of any PA, methods: 60–8,636), and the comparison set is
      # the PAs with ≥ min_years such years of both metrics in every version (user, 2026-11-08).
      min_transects = 60L,
      seed          = 46L   # bootstrap CIs of the paired differences (S1)
    )
  )
})
