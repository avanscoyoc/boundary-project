# 05_prepare_analysis_data.R: the analysis data (Stats ¶1).
#
# The analysis set (biomes; PAs whose boundary is dominated by sea or by open water; missing
# covariates; too few years), establishment-year imputation, z-scores on the final set,
# point-on-surface coordinates, the response SDs and the HMI mean and SD, the long PA-year table,
# the PA set of the transect-subset sensitivity runs, and the counts.
#
# Run from the repository root:
#   Rscript scripts/05_prepare_analysis_data.R [--out DIR]
# --out DIR writes every output under DIR instead (a dry run; data/ and results/ are untouched).
#
# Inputs:  results/pa_edge_metrics_ndvi.csv (04: one row per PA-year)
#          data/02_pas/pas_selected.gpkg (02: PA polygons; every PA of 02, with its RESOLVE biome)
#          data/04_transect_years/missing_transects.csv (04: transects with a missing year)
# Outputs: data/05_analysis/pa_level.csv   one row per analysis PA, ordered by WDPA_PID: attributes,
#                                          STATUS_YR (imputed where 0) + STATUS_YR_reported,
#                                          STATUS_YR_imputed, impute_pool; IUCN_group; z-scores;
#                                          lon/lat; PA means and n_years per transect version
#                                          (unsuffixed = "all"); pct_complete (median share of
#                                          complete transects); PA-set flags set_* (A3)
#          data/05_analysis/pa_year.csv    analysis PAs × transect version × year (long, A2)
#          data/05_analysis/excluded_pas.csv   every PA of script 02 that is not analysed, the
#                                          step that removed it, pct_complete, water_extent_pct
#          data/05_analysis/scaling.csv    n, mean, SD of each z-scored covariate (gHM_mean_z: the
#                                          HMI back-transform) and of each response's PA means
#                                          (the SD for the standardised change)
#          results/analysis_counts.json    counts for the methods and the Phase 4 checks (final_set:
#                                          transects, points, area, % of transects kept; missing_data:
#                                          transect-years without data at all 5 points)
# Coordinates are written only to data/05_analysis/, never to results/.

source(file.path("modules", "r", "config.R"))
for (f in setdiff(list.files(file.path("modules", "r"), "\\.R$", full.names = TRUE),
                  file.path("modules", "r", "config.R"))) {
  sys.source(f, envir = globalenv())   # corHaversine must live in globalenv()
}

args <- commandArgs(trailingOnly = TRUE)
out_root <- if (any(args == "--out")) args[which(args == "--out") + 1] else NULL
out <- if (is.null(out_root)) {
  list(dir = CFG$paths$analysis_dir, counts = CFG$paths$counts)
} else {
  list(dir = file.path(out_root, "05_analysis"), counts = file.path(out_root, "analysis_counts.json"))
}
say <- function(...) cat(sprintf(...), "\n", sep = "")

# ── 1. Inputs ────────────────────────────────────────────────────────────────
m <- readr::read_csv(CFG$paths$pa_metrics, guess_max = 1e6, show_col_types = FALSE, progress = FALSE,
                     col_types = readr::cols(WDPA_PID = "c", ISO3 = "c", .default = readr::col_guess()))
m <- as.data.frame(m)
names(m)[names(m) == "terrain_slope_mean"] <- "terrain_slope"     # A9
mt <- as.data.frame(readr::read_csv(CFG$paths$missing_transects, show_col_types = FALSE, progress = FALSE,
                                    col_types = readr::cols(WDPA_PID = "c", .default = readr::col_double())))
say("04 metrics: %d PA-years, %d PAs", nrow(m), length(unique(m$WDPA_PID)))

# One row per PA. The PA attributes are constant within a PA in 04's table. The covariates are
# static per transect, but 04 averages them per PA-year over that year's complete transects, so
# they can differ slightly between years: the PA value is their mean over the PA's years.
attr_cols <- c("WDPA_PID", "NAME", "ISO3", "BIOME_NAME", "IUCN_CAT", "STATUS_YR", "AREA_KM2")
cov_cols  <- c("gHM_mean", "elevation_mean", "terrain_slope", "water_extent_pct")
stopifnot(all(c(attr_cols, cov_cols, "pct_complete") %in% names(m)), !anyDuplicated(unique(m[attr_cols])$WDPA_PID))
pa_attr <- m[!duplicated(m$WDPA_PID), attr_cols]
pid <- factor(m$WDPA_PID, levels = pa_attr$WDPA_PID)
for (v in cov_cols) {
  x <- as.vector(tapply(m[[v]], pid, mean, na.rm = TRUE))
  pa_attr[[v]] <- ifelse(is.nan(x), NA_real_, x)
}
# Share of the PA's transects complete (data at all 5 points), median over its years: the sea rule
pa_attr$pct_complete <- as.vector(tapply(m$pct_complete, pid, stats::median, na.rm = TRUE))

# ── 2. PA-year table and PA means (all transect versions) ────────────────────
py <- pa_year_long(m, CFG$transects)
pm <- pa_means(py, unname(CFG$responses))
mean_cols <- setdiff(names(pm), c("WDPA_PID", "transects"))
pa <- merge(pa_attr, pm[pm$transects == "all", c("WDPA_PID", mean_cols)], by = "WDPA_PID")
stopifnot(nrow(pa) == nrow(pa_attr))

# ── 3. Analysis set ──────────────────────────────────────────────────────────
# Every PA of script 02 enters: a PA without any complete transect-year has no metrics in 04
# (e.g. Barrow Island, an island whose transects reach into the sea) and enters at 0% complete,
# so the sea rule removes it (user, 2026-11-11). Biome groups are checked against 04's.
polys <- sf::st_read(CFG$paths$pas_selected, quiet = TRUE)
polys$WDPA_PID <- as.character(polys$WDPA_PID)
pas02 <- sf::st_drop_geometry(polys)[c("WDPA_PID", "NAME", "ISO3", "BIOME_NAME")]
pa_in <- add_pas_without_metrics(pa, pas02)
no_metrics <- setdiff(pa_in$WDPA_PID, pa$WDPA_PID)
say("script 02: %d PAs; %d with metrics; without any complete transect-year: %s", nrow(pas02), nrow(pa),
    if (length(no_metrics)) paste(no_metrics, collapse = ", ") else "none")
year_cols <- paste0("n_years_", CFG$responses)
aset <- analysis_set(pa_in, CFG$exclude_biomes, unique(c("IUCN_CAT", "STATUS_YR", unname(CFG$covariates))),
                     year_cols, CFG$min_years, min_complete = CFG$min_complete_pct,
                     max_water = CFG$max_water_pct, start_label = "PAs from script 02")
d <- aset$set
print(aset$steps, row.names = FALSE)

# ── 4. Establishment year: impute STATUS_YR == 0 (donors: every PA with metrics) ─
imp <- impute_year(d[c("WDPA_PID", "ISO3", "STATUS_YR")], pa[c("WDPA_PID", "ISO3", "STATUS_YR")],
                   CFG$impute$draws, CFG$impute$min_country, CFG$impute$seed)
stopifnot(identical(imp$WDPA_PID, d$WDPA_PID))
d[c("STATUS_YR", "STATUS_YR_reported", "STATUS_YR_imputed", "impute_pool")] <-
  imp[c("STATUS_YR", "STATUS_YR_reported", "STATUS_YR_imputed", "impute_pool")]
say("imputed %d establishment years (%d country pool, %d global)", sum(d$STATUS_YR_imputed),
    sum(d$impute_pool %in% "country"), sum(d$impute_pool %in% "global"))

# ── 5. Groups and z-scores on the final set ──────────────────────────────────
d$IUCN_group <- as.character(iucn_group(d$IUCN_CAT, CFG$iucn_groups))
if (anyNA(d$IUCN_group)) stop("IUCN categories outside the config groups: ",
                              paste(unique(d$IUCN_CAT[is.na(d$IUCN_group)]), collapse = ", "))
stopifnot(all(d$BIOME_NAME %in% CFG$biome_order))
d <- add_zscores(d, CFG$covariates, CFG$log_covariates)
scaling <- attr(d, "scaling")
resp_sc <- do.call(rbind, lapply(CFG$responses, function(r) {
  x <- d[[paste0(r, "_mean")]]
  data.frame(variable = paste0(r, "_mean"), source = r, log = FALSE, n = length(x), mean = mean(x),
             sd = stats::sd(x), stringsAsFactors = FALSE)
}))
scaling <- rbind(scaling, resp_sc)
rownames(scaling) <- NULL

# ── 6. Coordinates: point on surface; jitter only coincident points ──────────
xy <- pa_coordinates(polys, d$WDPA_PID, CFG$coords_crs)
xy <- jitter_duplicates(xy, CFG$spatial$jitter_deg, CFG$spatial$jitter_seed, CFG$spatial$jitter_tries)
d$lon <- xy$lon; d$lat <- xy$lat
say("coordinates: %d PAs, %d jittered", nrow(xy), attr(xy, "n_jittered"))

# ── 7. PA means of the other transect versions (wide, suffixed) and the PA sets (A3) ─
for (v in setdiff(CFG$transects, "all")) {
  x <- pm[pm$transects == v, ]
  i <- match(d$WDPA_PID, x$WDPA_PID)
  for (col in mean_cols) d[[paste0(col, "_", v)]] <- x[[col]][i]
}
# Transect-subset runs: PA-years with ≥ sensitivity$min_transects complete transects in the
# version; PAs with ≥ min_years such years of both metrics in every version
ts <- transect_subset_set(py[py$WDPA_PID %in% d$WDPA_PID, ], CFG$transects, unname(CFG$responses),
                          CFG$sensitivity$min_transects, CFG$min_years)
stopifnot(identical(ts$WDPA_PID, d$WDPA_PID))
d$set_transects   <- ts$ok_every_version
d$set_established <- d$STATUS_YR <= CFG$sensitivity$established_by  # analysed (imputed) year
d$set_reported    <- !d$STATUS_YR_imputed

# ── 8. Write ─────────────────────────────────────────────────────────────────
dir.create(out$dir, recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(out$counts), recursive = TRUE, showWarnings = FALSE)
readr::write_csv(d, file.path(out$dir, "pa_level.csv"), na = "")
readr::write_csv(py[py$WDPA_PID %in% d$WDPA_PID, ], file.path(out$dir, "pa_year.csv"), na = "")
readr::write_csv(scaling, file.path(out$dir, "scaling.csv"))
ex <- merge(aset$removed, pa_in[c("WDPA_PID", "NAME", "ISO3", "BIOME_NAME", "pct_complete", "water_extent_pct")],
            by = "WDPA_PID", sort = FALSE)
readr::write_csv(ex[order(match(ex$step, aset$steps$step), ex$WDPA_PID, method = "radix"), ],
                 file.path(out$dir, "excluded_pas.csv"), na = "")

g <- d[d$STATUS_YR_imputed, ]
counts <- list(
  input = list(pa_years = nrow(m), pas_from_02 = nrow(pas02), pas_with_metrics = nrow(pa),
               pas_without_metrics = I(no_metrics), years = range(m$year)),
  analysis_set = aset$steps,
  n_analysis = nrow(d),
  final_set = final_set_summary(d$WDPA_PID, polys, py, d$AREA_KM2),
  missing_data = missing_data_summary(d$WDPA_PID, polys$N_TRNST[match(d$WDPA_PID, polys$WDPA_PID)], py, mt,
                                      d$BIOME_NAME, CFG$years),
  biome =as.list(table(factor(d$BIOME_NAME, levels = CFG$biome_order))),
  iucn_group = as.list(table(factor(d$IUCN_group, levels = names(CFG$iucn_groups)))),
  imputation = list(
    n_imputed = nrow(g),
    n_country_pool = sum(g$impute_pool == "country"),
    n_global_pool = sum(g$impute_pool == "global"),
    global_pool_countries = sort(unique(g$ISO3[g$impute_pool == "global"])),
    donors = list(pas = nrow(pa), known_year = sum(pa$STATUS_YR > 0)),
    n_half_year = sum(g$STATUS_YR %% 1 != 0),
    imputed_year_quantiles = as.list(stats::quantile(g$STATUS_YR, c(0, 0.25, 0.5, 0.75, 1))),
    settings = CFG$impute),
  coordinates = list(n_jittered = attr(xy, "n_jittered"), crs_point_on_surface = CFG$coords_crs),
  exclusion_settings = list(min_complete_pct = CFG$min_complete_pct, max_water_pct = CFG$max_water_pct),
  pa_sets = list(main = nrow(d),
                 transects = list(n = sum(d$set_transects), min_transects = CFG$sensitivity$min_transects),
                 established_by = list(year = CFG$sensitivity$established_by, n = sum(d$set_established)),
                 reported_year = sum(d$set_reported)),
  transect_versions = as.list(vapply(stats::setNames(CFG$transects, CFG$transects),
                                     function(v) sum(ts[[paste0("ok_", v)]]), integer(1))),
  scaling = scaling)
jsonlite::write_json(counts, out$counts, auto_unbox = TRUE, pretty = TRUE, digits = NA, dataframe = "rows")
say("wrote %s (%d PAs), pa_year.csv (%d rows), scaling.csv, %s", file.path(out$dir, "pa_level.csv"),
    nrow(d), sum(py$WDPA_PID %in% d$WDPA_PID), out$counts)
