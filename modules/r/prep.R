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
  key <- paste(py$WDPA_PID, as.integer(py$transects), sep = "\r")
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
#' Steps: drop PAs in `exclude_biomes` (or with no biome); drop PAs missing any of `covariates`
#' (one step per covariate, in the order given); drop PAs with fewer than `min_years` years of
#' any of `year_cols` (the main version's n_years_<metric>). Returns list(set = the kept rows
#' ordered by WDPA_PID, steps = data frame of step, n_removed, n_left).
analysis_set <- function(pa, exclude_biomes, covariates, year_cols, min_years) {
  stopifnot(!anyDuplicated(pa$WDPA_PID), all(c(covariates, year_cols, "BIOME_NAME") %in% names(pa)))
  # Each filter is a function of the remaining rows, applied in order.
  filters <- c(
    list(function(x) !is.na(x$BIOME_NAME) & !x$BIOME_NAME %in% exclude_biomes),
    lapply(covariates, function(v) function(x) !is.na(x[[v]])),
    list(function(x) Reduce(`&`, lapply(year_cols, function(v) x[[v]] >= min_years))))
  labels <- c(paste0("excluded biomes (", paste(exclude_biomes, collapse = ", "), ") or no biome"),
              paste("missing", covariates),
              sprintf("fewer than %d years of %s", min_years, paste(year_cols, collapse = " or ")))
  steps <- data.frame(step = c("PAs with metrics", labels), n_removed = NA_integer_, n_left = NA_integer_)
  steps$n_left[1] <- nrow(pa)
  for (j in seq_along(filters)) {
    keep <- filters[[j]](pa)
    keep[is.na(keep)] <- FALSE
    steps$n_removed[j + 1] <- sum(!keep); steps$n_left[j + 1] <- sum(keep)
    pa <- pa[keep, , drop = FALSE]
  }
  pa <- pa[order(pa$WDPA_PID, method = "radix"), , drop = FALSE]
  rownames(pa) <- NULL
  list(set = pa, steps = steps)
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
