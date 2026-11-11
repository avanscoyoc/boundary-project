# Analysis data (Stats ¶1, script 05): the long PA-year table, PA means, the analysis set,
# establishment-year imputation, z-scores and point-on-surface coordinates.
#
# Functions take data frames and explicit settings and read no files; 05 reads the inputs and
# writes data/05_analysis/. Rows are ordered by WDPA_PID (character, C-locale order: radix sort,
# independent of the session locale) throughout.

#' Long PA-year table (A2): one row per PA × transect version × year.
#'
#' `metrics` is script 04's wide table (one row per PA-year). Version "all" uses the unsuffixed
#' columns, version v the columns "<col>_<v>". Returns WDPA_PID, transects (factor, levels =
#' `versions`), year, and `cols`, ordered by WDPA_PID, version, year.
pa_year_long <- function(metrics, versions, cols = c("n_trnst", "edge_extent", "edge_intensity")) {
  stopifnot(!anyDuplicated(paste(metrics$WDPA_PID, metrics$year)))
  out <- do.call(rbind, lapply(versions, function(v) {
    src <- if (v == "all") cols else paste0(cols, "_", v)
    miss <- setdiff(src, names(metrics))
    if (length(miss)) stop("Columns not found: ", paste(miss, collapse = ", "))
    x <- data.frame(WDPA_PID = as.character(metrics$WDPA_PID), transects = v,
                    year = as.integer(metrics$year), stringsAsFactors = FALSE)
    x[cols] <- lapply(metrics[src], as.numeric)
    x
  }))
  out$transects <- factor(out$transects, levels = versions)
  out <- out[order(out$WDPA_PID, out$transects, out$year, method = "radix"), ]
  rownames(out) <- NULL
  out
}

#' PA means of the yearly metrics, per PA and transect version (old 01:96-118).
#'
#' Each metric's mean is unweighted over that metric's non-NA years (E5), and
#' n_years_<metric> counts those years. Returns WDPA_PID, transects, <metric>_mean,
#' n_years_<metric> for each metric, ordered by WDPA_PID, version.
pa_means <- function(py, metrics = c("edge_extent", "edge_intensity")) {
  key <- paste(py$WDPA_PID, as.character(py$transects), sep = "\r")   # factor or character
  first <- !duplicated(key)
  out <- data.frame(WDPA_PID = py$WDPA_PID[first], transects = py$transects[first],
                    stringsAsFactors = FALSE)
  k <- factor(key, levels = key[first])
  for (m in metrics) {
    ok <- !is.na(py[[m]])
    out[[paste0(m, "_mean")]] <- as.vector(tapply(ifelse(ok, py[[m]], 0), k, sum) / tapply(ok, k, sum))
    out[[paste0("n_years_", m)]] <- as.vector(tapply(ok, k, sum))
    out[[paste0(m, "_mean")]][out[[paste0("n_years_", m)]] == 0] <- NA_real_
  }
  out <- out[order(out$WDPA_PID, out$transects, method = "radix"), ]
  rownames(out) <- NULL
  out
}

#' The analysis set (Stats ¶1), step by step, from one row per PA with metrics.
#'
#' Steps, in order: drop PAs in `exclude_biomes` (or with no biome); if `min_complete` is given,
#' drop PAs whose `pct_complete` (median share of complete transects) is below it (boundaries
#' largely on the sea, where MODIS has no data); if `max_water` is given, drop PAs whose
#' `water_extent_pct` is above it (lakes, wetlands; a missing value is left to the covariate
#' step); drop PAs missing any of `covariates` (one step per covariate, in the order given); drop
#' PAs with fewer than `min_years` years of any of `year_cols` (the main version's
#' n_years_<metric>). Returns list(set = the kept rows ordered by WDPA_PID, steps = data frame of
#' step, n_removed, n_left, removed = WDPA_PID and step of every removed PA). `start_label` names
#' the first row of `steps` (the input).
analysis_set <- function(pa, exclude_biomes, covariates, year_cols, min_years,
                         min_complete = NULL, max_water = NULL, start_label = "PAs with metrics") {
  stopifnot(!anyDuplicated(pa$WDPA_PID), all(c(covariates, year_cols, "BIOME_NAME") %in% names(pa)),
            is.null(min_complete) || "pct_complete" %in% names(pa),
            is.null(max_water) || "water_extent_pct" %in% names(pa))
  # Each filter is a function of the remaining rows, applied in order.
  filters <- list(function(x) !is.na(x$BIOME_NAME) & !x$BIOME_NAME %in% exclude_biomes)
  labels <- paste0("excluded biomes (", paste(exclude_biomes, collapse = ", "), ") or no biome")
  if (!is.null(min_complete)) {
    filters <- c(filters, list(function(x) !is.na(x$pct_complete) & x$pct_complete >= min_complete))
    labels <- c(labels, sprintf("median share of complete transects < %g%% (sea)", min_complete))
  }
  if (!is.null(max_water)) {
    filters <- c(filters, list(function(x) is.na(x$water_extent_pct) | x$water_extent_pct <= max_water))
    labels <- c(labels, sprintf("more than %g%% of transects with surface water (lakes, wetlands)", max_water))
  }
  filters <- c(filters, lapply(covariates, function(v) function(x) !is.na(x[[v]])),
               list(function(x) Reduce(`&`, lapply(year_cols, function(v) x[[v]] >= min_years))))
  labels <- c(labels, paste("missing", covariates),
              sprintf("fewer than %d years of %s", min_years, paste(year_cols, collapse = " or ")))
  steps <- data.frame(step = c(start_label, labels), n_removed = NA_integer_, n_left = NA_integer_)
  steps$n_left[1] <- nrow(pa)
  removed <- data.frame(WDPA_PID = character(), step = character(), stringsAsFactors = FALSE)
  for (j in seq_along(filters)) {
    keep <- filters[[j]](pa)
    keep[is.na(keep)] <- FALSE
    steps$n_removed[j + 1] <- sum(!keep); steps$n_left[j + 1] <- sum(keep)
    if (any(!keep)) removed <- rbind(removed, data.frame(WDPA_PID = pa$WDPA_PID[!keep], step = labels[j],
                                                         stringsAsFactors = FALSE))
    pa <- pa[keep, , drop = FALSE]
  }
  pa <- pa[order(pa$WDPA_PID, method = "radix"), , drop = FALSE]
  rownames(pa) <- NULL
  list(set = pa, steps = steps, removed = removed)
}

#' The visual-inspection step (Finish plan Step 3): removes the PAs marked "exclude" in the
#' reviewer's checklist (written by script 05b from this analysis set) from an analysis_set()
#' result, as its last counted step.
#'
#' `checklist`: data frame with WDPA_PID and decision (and optionally order, used in messages).
#' Decisions are compared trimmed and in lower case; blank and "keep" keep the PA. Stops if a
#' PA is still marked "check", if any other value appears, or if the checklist does not list
#' exactly the PAs of the set (e.g. a checklist written from an older set).
apply_inspection <- function(aset, checklist, label = "removed after visual inspection") {
  stopifnot(all(c("WDPA_PID", "decision") %in% names(checklist)))
  pid <- as.character(checklist$WDPA_PID)
  dec <- tolower(trimws(ifelse(is.na(checklist$decision), "", as.character(checklist$decision))))
  where <- function(i) {
    o <- if ("order" %in% names(checklist)) paste0("#", checklist$order[i], " ") else ""
    paste0(o, pid[i], collapse = ", ")
  }
  if (anyDuplicated(pid)) stop("checklist lists a PA twice: ", paste(unique(pid[duplicated(pid)]), collapse = ", "))
  if (any(dec == "check")) stop(sum(dec == "check"), " PAs are still marked 'check' in the checklist: ",
                                where(which(dec == "check")))
  bad <- !dec %in% c("", "keep", "exclude")
  if (any(bad)) stop("checklist decisions other than blank / keep / exclude (",
                     paste(unique(checklist$decision[bad]), collapse = ", "), "): ", where(which(bad)))
  miss <- setdiff(aset$set$WDPA_PID, pid); extra <- setdiff(pid, aset$set$WDPA_PID)
  if (length(miss) || length(extra))
    stop("the checklist does not list the analysis set (", length(miss), " PAs missing, ", length(extra),
         " not in the set); was it written by script 05b from this set?")
  drop <- aset$set$WDPA_PID %in% pid[dec == "exclude"]
  aset$steps <- rbind(aset$steps, data.frame(step = label, n_removed = sum(drop), n_left = sum(!drop)))
  if (any(drop)) aset$removed <- rbind(aset$removed, data.frame(WDPA_PID = aset$set$WDPA_PID[drop], step = label,
                                                                stringsAsFactors = FALSE))
  aset$set <- aset$set[!drop, , drop = FALSE]
  rownames(aset$set) <- NULL
  aset
}

#' PAs for the transect-subset sensitivity runs (A3; user, 2026-11-08).
#'
#' In each transect version a PA-year counts only if it has ≥ `min_transects` complete
#' transects; a PA qualifies in a version with ≥ `min_years` such years with a value of every
#' metric in `metrics`. `py`: pa_year_long() rows (WDPA_PID, transects, year, n_trnst, metrics).
#' Returns one row per PA of `py` (ordered by WDPA_PID): ok_<version> for each of `versions`,
#' and ok_every_version.
transect_subset_set <- function(py, versions, metrics, min_transects, min_years) {
  stopifnot(all(c("WDPA_PID", "transects", "n_trnst", metrics) %in% names(py)))
  ids <- sort(unique(py$WDPA_PID), method = "radix")
  out <- data.frame(WDPA_PID = ids, stringsAsFactors = FALSE)
  for (v in versions) {
    x <- py[py$transects == v & !is.na(py$n_trnst) & py$n_trnst >= min_transects, , drop = FALSE]
    ok <- rep(TRUE, length(ids))
    for (m in metrics) {
      n <- tabulate(match(x$WDPA_PID[!is.na(x[[m]])], ids), nbins = length(ids))
      ok <- ok & n >= min_years
    }
    out[[paste0("ok_", v)]] <- ok
  }
  out$ok_every_version <- Reduce(`&`, out[paste0("ok_", versions)])
  out
}

#' Establishment-year imputation for STATUS_YR == 0 (Decisions: "Establishment year").
#'
#' For each target PA with year 0, in the order of `targets`, draw `draws` years with
#' replacement from the known years (> 0) of the `donors` in the same country; the imputed year
#' is the median of the draws. If the country pool has fewer than `min_country` donors, draw from
#' all donors' known years (global pool). ISO3 may list several countries ("A;B"): the pool is
#' every donor that lists any of them, each donor once. `seed` is set once, before the first
#' target. Known years are returned unchanged.
#'
#' `targets`, `donors`: data frames with WDPA_PID, ISO3, STATUS_YR. Returns `targets`' rows
#' with STATUS_YR (imputed where it was 0), STATUS_YR_reported (the input), STATUS_YR_imputed,
#' impute_pool ("country", "global", or NA for known years) and n_donors (pool size).
impute_year <- function(targets, donors, draws, min_country, seed) {
  stopifnot(all(c("WDPA_PID", "ISO3", "STATUS_YR") %in% names(targets)),
            all(c("WDPA_PID", "ISO3", "STATUS_YR") %in% names(donors)),
            !anyNA(targets$STATUS_YR), !anyNA(donors$STATUS_YR), draws >= 1)
  countries <- function(iso) strsplit(as.character(iso), ";", fixed = TRUE)
  known <- donors[donors$STATUS_YR > 0, , drop = FALSE]
  if (!nrow(known)) stop("No donor has a known establishment year.")
  known_iso <- countries(known$ISO3)
  out <- data.frame(WDPA_PID = targets$WDPA_PID, STATUS_YR = as.numeric(targets$STATUS_YR),
                    STATUS_YR_reported = targets$STATUS_YR, STATUS_YR_imputed = targets$STATUS_YR == 0,
                    impute_pool = NA_character_, n_donors = NA_integer_, stringsAsFactors = FALSE)
  target_iso <- countries(targets$ISO3)
  set.seed(seed)
  for (i in which(out$STATUS_YR_imputed)) {
    in_pool <- vapply(known_iso, function(x) any(x %in% target_iso[[i]]), logical(1))
    global <- sum(in_pool) < min_country
    pool <- if (global) known$STATUS_YR else known$STATUS_YR[in_pool]
    # Index draws: sample(x) with a single number x would draw from 1:x.
    out$STATUS_YR[i] <- stats::median(pool[sample.int(length(pool), draws, replace = TRUE)])
    out$impute_pool[i] <- if (global) "global" else "country"
    out$n_donors[i] <- length(pool)
  }
  out
}

#' z-scores over the rows of `d`: for each element of `vars` (name = new column, value = source
#' column), (x − mean) / SD, with log(x) first for the names in `log_vars`. Returns `d` with the
#' new columns and attribute "scaling" (variable, source, log, n, mean, sd; mean and SD on the
#' scale used, i.e. of log(x) for logged variables).
add_zscores <- function(d, vars, log_vars = character()) {
  stopifnot(!is.null(names(vars)), all(vars %in% names(d)), all(log_vars %in% names(vars)))
  sc <- data.frame(variable = names(vars), source = unname(vars), log = names(vars) %in% log_vars,
                   n = NA_integer_, mean = NA_real_, sd = NA_real_, stringsAsFactors = FALSE)
  for (j in seq_along(vars)) {
    x <- as.numeric(d[[vars[[j]]]])
    if (sc$log[j]) x <- log(x)
    if (anyNA(x) || !all(is.finite(x))) stop("Missing or non-finite values in ", vars[[j]])
    sc$n[j] <- length(x); sc$mean[j] <- mean(x); sc$sd[j] <- stats::sd(x)
    d[[names(vars)[j]]] <- (x - sc$mean[j]) / sc$sd[j]
  }
  attr(d, "scaling") <- sc
  d
}

#' Point-on-surface coordinates (lon, lat in EPSG:4326) of PA polygons, for `pids` in that order.
#'
#' The point is computed in `crs_compute` (config coords_crs: Mollweide, the old convention,
#' 03:94-103) and transformed to EPSG:4326. `polys`: sf with WDPA_PID, one row per PA.
pa_coordinates <- function(polys, pids, crs_compute) {
  stopifnot(inherits(polys, "sf"), !anyDuplicated(polys$WDPA_PID))
  i <- match(pids, as.character(polys$WDPA_PID))
  if (anyNA(i)) stop(sum(is.na(i)), " PAs have no polygon, e.g. ", pids[is.na(i)][1])
  g <- sf::st_transform(sf::st_geometry(polys)[i], crs_compute)
  xy <- sf::st_coordinates(sf::st_transform(sf::st_point_on_surface(g), 4326))
  data.frame(WDPA_PID = pids, lon = xy[, 1], lat = xy[, 2], stringsAsFactors = FALSE)
}

#' IUCN category → group (config iucn_groups); categories in no group give NA.
iucn_group <- function(cat, groups) {
  out <- rep(NA_character_, length(cat))
  for (g in names(groups)) out[cat %in% groups[[g]]] <- g
  factor(out, levels = names(groups))
}

#' Transects and size of the final PA set (Methods, the final dataset; user, 2026-11-10).
#'
#' `polys`: script 02's pas_selected (N_TRNST = transects kept, i.e. the PA's transects in the
#' point file sent to Earth Engine; N_TRNST_GEN = transects generated before the inner/outer-half
#' removals). `py`: pa_year_long() table; its "all" rows give the complete transects per PA-year.
#' `area`: AREA_KM2 of `pids`, in that order. Points: `points_per_transect` per transect
#' (02: 2 × POINTS_PER_SIDE + 1). Median and range of transects per PA, area, % of generated
#' transects kept (and pooled over the set) and complete transects per PA-year.
final_set_summary <- function(pids, polys, py, area, points_per_transect = 5L) {
  i <- match(pids, as.character(polys$WDPA_PID))
  if (anyNA(i)) stop(sum(is.na(i)), " PAs are not in pas_selected, e.g. ", pids[is.na(i)][1])
  stopifnot(length(area) == length(pids))
  kept <- polys$N_TRNST[i]
  gen  <- polys$N_TRNST_GEN[i]
  rng <- function(x) list(median = stats::median(x, na.rm = TRUE), min = min(x, na.rm = TRUE),
                          max = max(x, na.rm = TRUE))
  a <- py[py$transects == "all" & py$WDPA_PID %in% pids, ]
  list(pas = length(pids),
       transects = sum(kept),
       points = sum(kept) * points_per_transect,
       transects_per_pa = rng(kept),
       area_km2 = rng(area),
       pct_transects_kept = c(rng(100 * kept / gen), list(pooled = 100 * sum(kept) / sum(gen))),
       pa_years = sum(!is.na(a$n_trnst)),
       complete_transects_per_pa_year = rng(a$n_trnst))
}

#' RESOLVE Ecoregions 2017 biome names -> the 8 groups of script 04 (`recategorize_biome()` in
#' modules/analysis.py). RESOLVE's "N/A" (rock and ice) is "Rock & Ice"; a missing name, "Unknown",
#' "None" or "" gives NA; any other name is kept as it is (as in Python).
BIOME_GROUPS <- c(
  "Tropical & Subtropical Moist Broadleaf Forests" = "Tropical Forest",
  "Tropical & Subtropical Dry Broadleaf Forests" = "Tropical Forest",
  "Tropical & Subtropical Coniferous Forests" = "Tropical Forest",
  "Temperate Broadleaf & Mixed Forests" = "Temperate Forest",
  "Temperate Conifer Forests" = "Temperate Forest",
  "Mediterranean Forests, Woodlands & Scrub" = "Temperate Forest",
  "Boreal Forests/Taiga" = "Boreal Forest",
  "Tropical & Subtropical Grasslands, Savannas & Shrublands" = "Grassland & Shrubland",
  "Temperate Grasslands, Savannas & Shrublands" = "Grassland & Shrubland",
  "Flooded Grasslands & Savannas" = "Grassland & Shrubland",
  "Montane Grasslands & Shrublands" = "Grassland & Shrubland",
  "Tundra" = "Tundra",
  "Deserts & Xeric Shrublands" = "Desert",
  "Mangroves" = "Mangrove",
  "N/A" = "Rock & Ice"
)

biome_group <- function(x) {
  x <- as.character(x)
  out <- unname(BIOME_GROUPS[x])
  keep <- is.na(out) & !is.na(x) & !x %in% c("Unknown", "None", "")
  out[keep] <- x[keep]
  out
}

#' PAs of script 02 (`pas02`: WDPA_PID, NAME, ISO3, BIOME_NAME = RESOLVE name) without any
#' complete transect-year, so without metrics in 04 (e.g. a small island whose transects reach
#' into the sea), appended to `pa` (one row per PA with metrics, BIOME_NAME grouped) with
#' pct_complete = 0 and their grouped biome, so that the analysis set's rules apply to them (the
#' sea rule removes them unless the biome step does; user, 2026-11-11). Checks that every PA with
#' metrics is in `pas02` with the same grouped biome. Other columns of the added rows are NA.
add_pas_without_metrics <- function(pa, pas02) {
  stopifnot(all(c("WDPA_PID", "BIOME_NAME") %in% names(pas02)), !anyDuplicated(pas02$WDPA_PID))
  i <- match(pa$WDPA_PID, pas02$WDPA_PID)
  if (anyNA(i)) stop(sum(is.na(i)), " PAs with metrics are not in pas_selected, e.g. ", pa$WDPA_PID[is.na(i)][1])
  grp <- biome_group(pas02$BIOME_NAME)
  a <- grp[i]; b <- as.character(pa$BIOME_NAME)
  same <- (is.na(a) & is.na(b)) | (!is.na(a) & !is.na(b) & a == b)
  if (!all(same)) stop("biome groups differ from script 04 for ", sum(!same), " PAs, e.g. ", pa$WDPA_PID[!same][1])
  add <- pas02[!pas02$WDPA_PID %in% pa$WDPA_PID, , drop = FALSE]
  if (!nrow(add)) return(pa)
  new <- pa[rep(NA_integer_, nrow(add)), , drop = FALSE]
  rownames(new) <- NULL
  for (v in intersect(c("WDPA_PID", "NAME", "ISO3"), names(add))) new[[v]] <- as.character(add[[v]])
  new$BIOME_NAME <- grp[match(add$WDPA_PID, pas02$WDPA_PID)]
  new$pct_complete <- 0
  out <- rbind(pa, new)
  rownames(out) <- NULL
  out
}

#' Missing data of the analysis set (user, 2026-11-11). A transect-year is missing when any of its
#' 5 points has no value that year; script 04 leaves it out of that year's metrics.
#'
#' `pids`: the analysis PAs; `n_transects`, `biome`: their transects (N_TRNST) and biomes, same
#' order; `py`: pa_year_long() rows (version "all": complete transects per PA-year; a PA-year
#' without any is absent); `mt`: script 04's missing_transects.csv (one row per transect with a
#' missing year, all PAs); `years`: the record. The missing transect-years of each PA from `py`
#' must equal those in `mt` (checked). Returns a list for analysis_counts.json.
missing_data_summary <- function(pids, n_transects, py, mt, biome, years) {
  stopifnot(length(n_transects) == length(pids), length(biome) == length(pids), !anyDuplicated(pids))
  Y <- length(years)
  f <- function(x) factor(x, levels = pids)
  a <- py[py$transects == "all" & py$WDPA_PID %in% pids, ]
  miss_pa <- n_transects * Y - as.numeric(tapply(a$n_trnst, f(a$WDPA_PID), sum, default = 0))
  m <- mt[mt$WDPA_PID %in% pids, , drop = FALSE]
  from_mt <- as.numeric(tapply(m$n_missing_years, f(m$WDPA_PID), sum, default = 0))
  if (!isTRUE(all.equal(miss_pa, from_mt)))
    stop("missing transect-years differ between the PA-year table and missing_transects.csv for ",
         sum(miss_pa != from_mt), " PAs (rerun script 04?)")
  N <- sum(n_transects); TY <- N * Y; M <- sum(miss_pa)
  never <- m$n_missing_years == Y
  part <- m$n_missing_years[!never]
  pos <- colSums(m[paste0("miss_", c("pt_m2", "pt_m1", "pt_0", "pt_1", "pt_2"))])
  names(pos) <- c("-5 km (interior)", "-2.5 km (interior)", "0 km (boundary)", "+2.5 km (exterior)", "+5 km (exterior)")
  by_year <- sum(n_transects) - as.numeric(tapply(a$n_trnst, factor(a$year, levels = years), sum, default = 0))
  pct_pa <- 100 * miss_pa / (n_transects * Y)
  aff <- miss_pa > 0
  b <- factor(biome)
  list(
    transects = N, transect_years = TY,
    missing_transect_years = M, pct_transect_years_missing = 100 * M / TY,
    transects_never_missing = N - nrow(m),
    transects_never_usable = list(n = sum(never), pct_of_transects = 100 * sum(never) / N,
                                  pct_of_missing = 100 * sum(never) * Y / M),
    transects_part_year = list(n = length(part), pct_of_transects = 100 * length(part) / N,
                               pct_of_missing = 100 * sum(part) / M,
                               years_missing = list(mean = mean(part), median = stats::median(part),
                                                    n_1 = sum(part == 1), n_2_to_5 = sum(part >= 2 & part <= 5),
                                                    n_6_or_more = sum(part >= 6))),
    pct_missing_point_years_at_water = 100 * sum(m$n_missing_point_years_water) / sum(m$n_missing_point_years),
    pct_never_usable_transects_with_water = 100 * mean(m$water_any[never] == 1),
    pct_missing_point_years_by_position = as.list(100 * pos / sum(pos)),
    per_year = list(min = min(by_year), max = max(by_year), mean = mean(by_year)),
    pas_affected = sum(aff), pct_pas_affected = 100 * mean(aff),
    affected_pa_pct_transect_years_missing = as.list(stats::setNames(
      stats::quantile(pct_pa[aff], c(0.5, 0.25, 0.75, 1), names = FALSE), c("median", "q25", "q75", "max"))),
    by_biome = data.frame(BIOME_NAME = levels(b), PAs = as.integer(table(b)),
                          PAs_affected = as.integer(tapply(aff, b, sum)),
                          missing_transect_years = as.numeric(tapply(miss_pa, b, sum)),
                          pct_transect_years_missing = as.numeric(100 * tapply(miss_pa, b, sum) /
                                                                    (tapply(n_transects, b, sum) * Y)),
                          stringsAsFactors = FALSE)
  )
}
