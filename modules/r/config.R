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
      analysis_dir = p("data", "05_analysis"),                    # script 05 outputs
      cache_dir    = p("data", "05_analysis", "cache"),           # distance set-ups and fitted models
      tables       = p("results", "tables"),
      figures      = p("results", "figures"),
      diagnostics  = p("results", "diagnostics"),                 # outputs not reported in the paper (D4)
      counts       = p("results", "analysis_counts.json"),
      session_info = p("results", "R_sessionInfo.txt"),
      parity_dir   = p("data", "parity"),
      legacy_root  = p("PA_New_Analysis")                         # collaborator's project (read-only)
    ),

    # ── Record and analysis set ──────────────────────────────────────────────
    years     = 2003:2025,
    n_span    = 22L,          # years spanned by the record (2025 − 2003): change = slope × n_span
    min_years = 5L,           # a PA needs ≥ 5 complete years of BOTH metrics (Stats ¶1)
    transects = c("all", "nowater", "nootherpa", "noflags"),      # transect versions (A2); "all" is the main analysis
    exclude_biomes = c("Mangrove", "Rock & Ice"),

    # Biome order for tables and figures only. Reference levels are set separately (B2).
    biome_order = c("Grassland & Shrubland", "Tropical Forest", "Temperate Forest",
                    "Boreal Forest", "Desert", "Tundra"),

    # IUCN categories → three groups (00_setup / 01_data_prep)
    iucn_groups = list(
      "strict"     = c("Ia", "Ib", "II"),
      "non-strict" = c("III", "IV", "V", "VI"),
      "unknown"    = c("Not Applicable", "Not Assigned", "Not Reported")
    ),

    # ── Reference levels (D1), one pair per model family ─────────────────────
    ref = list(
      mean   = list(iucn = "strict",     biome = "Grassland & Shrubland"),   # 06; Fig. 3A, Table SX
      change = list(iucn = "non-strict", biome = "Boreal Forest")            # 08; Table SZ
    ),

    # ── Trends (Stats ¶4–5) ──────────────────────────────────────────────────
    trend = list(
      fdr    = 0.05,                                  # Benjamini–Hochberg across PAs, per metric
      levels = c("increase", "no_trend", "decrease")  # the same codes as Python's trend_* columns (A9)
    ),

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

    # ── Sensitivity (09) ─────────────────────────────────────────────────────
    sensitivity = list(
      established_by = 2003L    # PA subset (a): analysed establishment year ≤ 2003
                                # PA subset (b): reported year only (STATUS_YR_imputed == FALSE)
    )
  )
})
