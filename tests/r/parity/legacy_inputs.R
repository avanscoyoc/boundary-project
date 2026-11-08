# Old conventions of the collaborator's analysis (PA_New_Analysis/, read-only), in one place.
#
# The functions in modules/r/ know nothing about the old run. The parity tests use this file to
# turn the old frozen inputs into the new column names and settings, run the new functions, and
# compare with the old outputs/tables/. Line references are PA_New_Analysis/scripts/<file>:<line>.
#
# Old → new names: IUCN_strict → IUCN_group, slope_mean → terrain_slope (A9),
# AREA_DISSO_z → log_area_z, trend labels → CFG$trend$levels. Intensity is the old "mean4"
# version, (D02 + D01 + D0m1 + D0m2) / 4 (01:54), which is the v2 `edge_intensity`.

LEGACY_RENAME <- c(
  IUCN_group      = "IUCN_strict",
  terrain_slope   = "slope_mean",
  terrain_slope_z = "slope_mean_z",
  log_area_z      = "AREA_DISSO_z"
)

LEGACY_CLASSES <- c(
  "increasing"          = "increase",
  "no detectable trend" = "no_trend",
  "stable"              = "no_trend",   # pre-2026-11 label, mapped by load_data() (00_setup)
  "decreasing"          = "decrease"
)

# Model terms as named in the old tables → new names (for comparing coefficient tables)
LEGACY_TERMS <- c(
  "AREA_DISSO_z" = "log_area_z",
  "slope_mean_z" = "terrain_slope_z",
  "IUCN_strict"  = "IUCN_group"
)

legacy_file <- function(..., root = CFG$paths$legacy_root) file.path(root, ...)

legacy_available <- function(root = CFG$paths$legacy_root) {
  all(file.exists(file.path(root, c(
    "data/wdpa_df_ndvi_trends.csv",
    "data/processed/pa_level_clean.csv",
    "data/processed/wdpa_map_ready.gpkg",
    "outputs/tables/response_sd_gls.csv"
  ))))
}

legacy_map_classes <- function(x) {
  out <- unname(LEGACY_CLASSES[as.character(x)])
  stopifnot(all(is.na(out) == is.na(x)))
  out
}

# Annual PA-year data as 01_data_prep.R builds `df` (01:17-58, 120-133), in the new long format
# (one `transects` version, "all"). Rows keep the file order within the old PA set.
legacy_annual <- function(root = CFG$paths$legacy_root,
                          exclude_biomes = c("Mangrove", "Rock & Ice"),
                          biome_order = CFG$biome_order, min_years = 5L) {
  raw <- readr::read_csv(legacy_file("data", "wdpa_df_ndvi_trends.csv", root = root),
                         col_types = readr::cols(WDPA_PID = "c", .default = readr::col_guess()),
                         show_col_types = FALSE, progress = FALSE)
  d <- raw[!raw$BIOME_NAME %in% exclude_biomes & !is.na(raw$BIOME_NAME), ]
  d$edge_intensity_mean4 <- (d$D02 + d$D01 + d$D0m1 + d$D0m2) / 4
  # PA set of 01: biome in BIOME_ORDER and ≥ min_years non-NA edge_extent years (01:99, 105-108, 125-126)
  n_ext <- tapply(!is.na(d$edge_extent), d$WDPA_PID, sum)
  biome <- tapply(d$BIOME_NAME, d$WDPA_PID, function(b) dplyr::first(stats::na.omit(b)))
  keep <- names(n_ext)[n_ext >= min_years & biome[names(n_ext)] %in% biome_order]
  d <- d[d$WDPA_PID %in% keep, ]
  data.frame(
    WDPA_PID       = d$WDPA_PID,
    year           = as.integer(d$year),
    transects      = "all",
    n_trnst        = d$n_trnst,
    edge_extent    = d$edge_extent,
    edge_intensity = d$edge_intensity_mean4,
    stringsAsFactors = FALSE
  )
}

# The frozen PA-level table (4,025 PAs; written by 02:580), with new names. Old file order kept.
legacy_pa <- function(root = CFG$paths$legacy_root) {
  pa <- readr::read_csv(legacy_file("data", "processed", "pa_level_clean.csv", root = root),
                        col_types = readr::cols(WDPA_PID = "c", .default = readr::col_guess()),
                        show_col_types = FALSE, progress = FALSE)
  pa <- dplyr::rename(pa, dplyr::all_of(LEGACY_RENAME))
  for (col in intersect(names(pa), c("trend_extent", "trend_intensity",
                                     "trend_extent_zyp", "trend_intensity_zyp"))) {
    pa[[col]] <- legacy_map_classes(pa[[col]])
  }
  as.data.frame(pa)
}

# Point-on-surface coordinates as in 03:94-103 / 02b:76-85: first row per PID of
# wdpa_map_ready.gpkg, point-on-surface in its native Mollweide CRS, then EPSG:4326.
# Returned in the order of `pids`.
legacy_coords <- function(pids, root = CFG$paths$legacy_root) {
  cent <- sf::st_read(legacy_file("data", "processed", "wdpa_map_ready.gpkg", root = root),
                      query = "SELECT WDPA_PID, geom FROM wdpa_map_ready", quiet = TRUE)
  cent$WDPA_PID <- as.character(cent$WDPA_PID)
  cent <- cent[!duplicated(cent$WDPA_PID), ]
  xy <- sf::st_coordinates(sf::st_transform(sf::st_point_on_surface(sf::st_geometry(cent)), 4326))
  i <- match(pids, cent$WDPA_PID)
  data.frame(WDPA_PID = pids, lon = xy[i, 1], lat = xy[i, 2], stringsAsFactors = FALSE)
}

# The old model set: complete cases of the model variables plus coordinates (03:106-124;
# 02b:88-106), in file order. n = 3,914 for both the mean and the change models.
LEGACY_MODEL_VARS <- c("WDPA_PID", "edge_extent_mean", "edge_intensity_mean",
                       "STATUS_YR_z", "log_area_z", "gHM_mean_z", "elevation_mean_z",
                       "terrain_slope_z", "water_extent_pct_z", "IUCN_group", "BIOME_NAME",
                       "lon", "lat")

legacy_model_set <- function(pa, coords, extra_vars = character()) {
  stopifnot(identical(pa$WDPA_PID, coords$WDPA_PID))
  pa$lon <- coords$lon
  pa$lat <- coords$lat
  vars <- c(LEGACY_MODEL_VARS, extra_vars)
  pa[stats::complete.cases(pa[, vars]), , drop = FALSE]
}

# Old settings that the new functions take as arguments.
legacy_settings <- function(pa, root = CFG$paths$legacy_root) {
  sd_tab <- readr::read_csv(legacy_file("outputs", "tables", "response_sd_gls.csv", root = root),
                            show_col_types = FALSE, progress = FALSE)
  list(
    # Response SDs used for the standardised change: the 3,914-PA SDs written by 03 (02:42-61)
    sd_resp = c(extent    = sd_tab$sd_response[sd_tab$response == "extent"],
                intensity = sd_tab$sd_response[sd_tab$response == "intensity"]),
    n_span    = 22L,                   # 02:35
    min_years = 5L,
    fdr       = 0.05,
    # HMI back-transform for Fig. 3 predictions: mean and SD over the 4,025-PA table (03:87-88)
    hmi = c(mean = mean(pa$gHM_mean, na.rm = TRUE), sd = stats::sd(pa$gHM_mean, na.rm = TRUE)),
    biome_order = c("Grassland & Shrubland", "Tropical Forest", "Temperate Forest",
                    "Boreal Forest", "Desert", "Tundra"),   # 00:66-73
    # Factor levels of the old fits: IUCN strict first (03:80-81), BIOME_ORDER (Grassland first),
    # for both the mean (03) and the change models (02b). Contrasts vs non-strict / Boreal were
    # computed afterwards with emmeans (03:63-64, 02b:51-52).
    ref_fit      = list(iucn = "strict",     biome = "Grassland & Shrubland"),
    ref_contrast = list(iucn = "non-strict", biome = "Boreal Forest"),
    boot_R     = 10000L,
    boot_seeds = c(biome = 42L, weighted = 43L, sen = 44L, pooled = 45L),
    jitter_deg = 1e-4, jitter_seed = 1L, jitter_tries = 10L
  )
}

legacy_table <- function(name, root = CFG$paths$legacy_root) {
  as.data.frame(readr::read_csv(legacy_file("outputs", "tables", name, root = root),
                                show_col_types = FALSE, progress = FALSE))
}

# Old model term and column names → new (for comparing model tables).
legacy_rename_terms <- function(x) {
  for (o in names(LEGACY_TERMS)) x <- gsub(o, LEGACY_TERMS[[o]], x, fixed = TRUE)
  x
}

# Saved spatial ranges of the old FULL models (spatial_params_gls.csv). The additive models'
# ranges were not saved; their tables are checked by the optimised refits (run_parity_full.R).
legacy_ranges <- function(root = CFG$paths$legacy_root) {
  sp <- legacy_table("spatial_params_gls.csv", root = root)
  stats::setNames(sp$range_km, sp$response)
}

# The old model data (03:77-141), in the new conventions: the 3,914-PA model set in file order,
# coordinates jittered as 03 (none coincide), factors with the old levels (strict first,
# BIOME_ORDER: Grassland first).
legacy_model_data <- function(pa, coords, settings) {
  d <- legacy_model_set(pa, coords)
  d <- jitter_duplicates(d, settings$jitter_deg, settings$jitter_seed, settings$jitter_tries)
  d$IUCN_group <- set_reference(d$IUCN_group, settings$ref_fit$iucn, c("strict", "non-strict", "unknown"))
  d$BIOME_NAME <- set_reference(d$BIOME_NAME, settings$ref_fit$biome, settings$biome_order)
  rownames(d) <- NULL
  d
}

LEGACY_RESPONSES <- c(extent = "edge_extent_mean", intensity = "edge_intensity_mean")

# Saved spatial ranges of the old change models A (trend_drivers_spatial_params_trend_gls.csv).
# The ranges of the precision-weighted models B were not saved (checked by run_parity_full.R).
legacy_change_ranges <- function(root = CFG$paths$legacy_root) {
  sp <- legacy_table("trend_drivers_spatial_params_trend_gls.csv", root = root)
  stats::setNames(sp$range_km, sp$response)
}

LEGACY_CHANGE <- list(extent    = c(response = "change_extent_z",    se = "se_change_extent_z"),
                      intensity = c(response = "change_intensity_z", se = "se_change_intensity_z"))
