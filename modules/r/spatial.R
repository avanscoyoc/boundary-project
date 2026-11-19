# Spatial set-up for the GLS models: the corHaversine correlation structure, cached gls fits and
# Moran's I of residuals. Needs corHaversine.R sourced into globalenv().
# Row order matters: the correlation structure, the fits and the Moran weights refer to the rows
# of the model data in order, so build them all from the same data frame.

#' Jitter PAs whose (lon, lat), rounded to 6 decimals, equals an earlier PA's: corHaversine stops
#' on zero distances. Each duplicate moves by up to ±`jitter_deg` (uniform) in lon and lat, for up
#' to `tries` rounds, after set.seed(seed). Attribute "n_jittered" counts the moved PAs.
jitter_duplicates <- function(d, jitter_deg, seed, tries = 10L) {
  stopifnot(all(c("lon", "lat") %in% names(d)), !anyNA(d$lon), !anyNA(d$lat))
  key <- function(d) paste(round(d$lon, 6), round(d$lat, 6))
  dup <- duplicated(key(d))
  n_dup <- sum(dup)
  set.seed(seed)
  k <- 0L
  while (any(dup) && k < tries) {
    d$lon[dup] <- d$lon[dup] + stats::runif(sum(dup), -jitter_deg, jitter_deg)
    d$lat[dup] <- d$lat[dup] + stats::runif(sum(dup), -jitter_deg, jitter_deg)
    dup <- duplicated(key(d)); k <- k + 1L
  }
  if (any(dup)) stop("Coordinates still coincide after ", tries, " jitter rounds.")
  attr(d, "n_jittered") <- n_dup
  d
}

# Cache file for an object identified by `key_obj` (a list hashed with rlang::hash).
.cache_path <- function(cache_dir, prefix, key_obj) {
  file.path(cache_dir, sprintf("%s_%s.rds", prefix, rlang::hash(key_obj)))
}

#' saveRDS() to a temporary file, then rename it, so a run killed while writing leaves no
#' truncated cache file.
save_atomic <- function(x, path) {
  tmp <- paste0(path, ".tmp", Sys.getpid())
  saveRDS(x, tmp)
  if (!file.rename(tmp, path)) { unlink(tmp); stop("could not write ", path) }
  invisible(path)
}

#' Initialised corHaversine structure (exponential, great-circle km) for the rows of `coords`.
#'
#' `value`: range in km, the optimiser start or, with fixed = TRUE (sensitivity refits), the
#' fixed range; with nugget = TRUE the start is c(value, 0.1). Initialising factors the n × n
#' correlation matrix (about a minute for the analysis set), so the result is cached in
#' `cache_dir` (NULL = no cache), keyed on the coordinates in row order and every setting
#' (attribute "cache_key").
spatial_setup <- function(coords, value, nugget = FALSE, fixed = FALSE, mimic = "corExp",
                          cache_dir = NULL) {
  stopifnot(all(c("lon", "lat") %in% names(coords)), !anyNA(coords$lon), !anyNA(coords$lat),
            exists("corHaversine", envir = globalenv()))
  start <- if (nugget) c(value, 0.1) else value
  key_obj <- list(what = "corHaversine-init-v1", lon = as.numeric(coords$lon),
                  lat = as.numeric(coords$lat), start = start, nugget = nugget, fixed = fixed,
                  mimic = mimic)
  path <- if (!is.null(cache_dir)) .cache_path(cache_dir, "corsp", key_obj)
  if (!is.null(path) && file.exists(path)) return(readRDS(path))
  cs <- nlme::Initialize(
    corHaversine(value = start, form = ~ lon + lat, mimic = mimic, nugget = nugget, fixed = fixed),
    data = data.frame(lon = coords$lon, lat = coords$lat))
  attr(cs, "cache_key") <- basename(path %||% "")
  if (!is.null(path)) {
    dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
    save_atomic(cs, path)
  }
  cs
}

#' nlme::gls fit, cached on disk in `cache_dir` (NULL = no cache).
#'
#' The cache key covers the formula, the data columns used (values, row order and factor levels,
#' so a new reference level is a new fit), the correlation structure (class, settings, start
#' values, a hash of its distances), the variance function (class, formula, values), method,
#' max_iter and the nlme version. The formula is written into the stored call so emmeans and
#' predict see it; still pass `data =` to emmeans. Adds attributes "cache_key" and "fit_minutes".
fit_gls_cached <- function(formula, data, correlation = NULL, weights = NULL, method = "REML",
                           max_iter = 200L, cache_dir = NULL, label = NULL) {
  vars <- unique(c(all.vars(formula),
                   if (!is.null(weights)) all.vars(attr(weights, "formula")),
                   if (!is.null(correlation)) all.vars(attr(correlation, "formula"))))
  stopifnot(all(vars %in% names(data)))
  used <- lapply(data[vars], function(x) if (is.factor(x)) list(levels(x), as.integer(x)) else x)
  # Formulas are keyed as text: a formula's environment differs between sessions.
  txt <- function(f) paste(deparse(f), collapse = " ")
  cor_key <- if (!is.null(correlation)) list(
    class = class(correlation), value = as.numeric(correlation),
    formula = txt(attr(correlation, "formula")),
    attrs = attributes(correlation)[c("nugget", "fixed", "function")],
    covariate = rlang::hash(attr(correlation, "covariate")))
  w_key <- if (!is.null(weights)) list(class = class(weights), formula = txt(attr(weights, "formula")),
                                       value = as.numeric(weights))
  key_obj <- list(what = "gls-v1", formula = txt(formula),
                  data = used, correlation = cor_key, weights = w_key, method = method,
                  max_iter = max_iter, nlme = as.character(utils::packageVersion("nlme")))
  path <- if (!is.null(cache_dir)) .cache_path(cache_dir, "gls", key_obj)
  if (!is.null(path) && file.exists(path)) {
    if (!is.null(label)) message(sprintf("  %-40s cached", label))
    return(readRDS(path))
  }
  t0 <- Sys.time()
  ctrl <- nlme::glsControl(maxIter = max_iter, msMaxIter = max_iter)
  # Only non-NULL arguments enter the call (emmeans reads call$weights). The formula and the small
  # variance function are inlined in the call; the correlation structure is not.
  args <- list(model = formula, data = quote(data), method = method, control = quote(ctrl))
  if (!is.null(correlation)) args$correlation <- quote(correlation)
  if (!is.null(weights)) args$weights <- weights
  m <- eval(as.call(c(quote(nlme::gls), args)))
  mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  if (!is.null(label)) message(sprintf("  %-40s %5.1f min", label, mins))
  attr(m, "cache_key") <- basename(path %||% "")
  attr(m, "fit_minutes") <- mins
  if (!is.null(path)) {
    dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
    save_atomic(m, path)
  }
  m
}

#' Spatial weights for Moran's I of residuals, rows as in `coords`: k nearest neighbours
#' (great-circle, row-standardised) and one set per distance band [bands_km[i], bands_km[i+1]] km
#' (NULL where a band has no links).
moran_weights <- function(coords, k, bands_km) {
  xy <- cbind(coords$lon, coords$lat)
  knn <- spdep::nb2listw(spdep::knn2nb(spdep::knearneigh(xy, k = k, longlat = TRUE)), style = "W")
  bands <- lapply(seq_len(length(bands_km) - 1), function(i) {
    nb <- spdep::dnearneigh(xy, d1 = bands_km[i], d2 = bands_km[i + 1], longlat = TRUE)
    n_links <- sum(spdep::card(nb))
    list(mid_km = (bands_km[i] + bands_km[i + 1]) / 2, n_links = n_links,
         lw = if (n_links > 0) spdep::nb2listw(nb, style = "W", zero.policy = TRUE))
  })
  list(knn = knn, bands = bands, n = nrow(xy))
}

#' Moran's I of residuals `r` (in the row order of the weights): kNN permutation test (moran.mc,
#' `nsim`, after set.seed(seed)) and, if `bands = TRUE`, the distance-band correlogram
#' (moran.test, randomisation). Returns list(knn, bands).
moran_residuals <- function(r, w, nsim, seed, label, bands = TRUE) {
  r <- as.numeric(r)
  stopifnot(length(r) == w$n, !anyNA(r))
  set.seed(seed)
  mc <- spdep::moran.mc(r, w$knn, nsim = nsim)
  knn <- data.frame(model = label, I = unname(mc$statistic), expected_I = -1 / (length(r) - 1),
                    p = mc$p.value, stringsAsFactors = FALSE)
  bd <- if (bands) do.call(rbind, lapply(w$bands, function(b) {
    out <- data.frame(model = label, mid_km = b$mid_km, n_links = b$n_links, I = NA_real_,
                      p = NA_real_, stringsAsFactors = FALSE)
    if (!is.null(b$lw)) {
      mt <- spdep::moran.test(r, b$lw, zero.policy = TRUE)
      out$I <- unname(mt$estimate["Moran I statistic"]); out$p <- mt$p.value
    }
    out
  }))
  list(knn = knn, bands = bd)
}
