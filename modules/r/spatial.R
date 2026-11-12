# Spatial set-up for the GLS models: the corHaversine correlation structure, cached model fits,
# and Moran's I of residuals (diagnostics).
#
# Follows the earlier R implementation of this analysis (not in this repository). Needs
# modules/r/corHaversine.R in globalenv().
#
# Row order matters: the correlation structure, the fits and the Moran weights all refer to the
# rows of the model data in the order given. Build them from the same data frame.

#' Jitter PAs whose coordinates coincide with an earlier PA's.
#'
#' corHaversine stops on zero distances, so a PA whose (lon, lat), rounded to 6 decimals, equals
#' an earlier row's is moved by up to ±`jitter_deg` in each coordinate (uniform; lon first, then
#' lat), repeating for up to `tries` rounds until no duplicates are left. `seed` is set first.
#' Returns `d` with lon/lat changed for the duplicates only; attribute "n_jittered".
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

#' Initialised corHaversine structure (exponential, great-circle km) for the rows of `coords`.
#'
#' `value` is the range in km: the optimiser start (fixed = FALSE) or the fixed range
#' (fixed = TRUE, used by the fast parity tests and the sensitivity refits). With nugget = TRUE
#' the start is c(value, 0.1). Initialisation computes the pairwise distances (≈ 1 s)
#' and nlme's factor of the n × n correlation matrix (≈ 1 min at n ≈ 4,600), so the initialised
#' object is cached in `cache_dir` (NULL = no cache) under a key that covers the coordinates in
#' row order and every setting. Attribute "cache_key" holds that key.
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
    saveRDS(cs, path)
  }
  cs
}

#' nlme::gls fit, cached on disk.
#'
#' The key covers: the formula; the data columns the model uses (values, row order, and factor
#' levels, so a change of reference level is a new fit); the correlation structure (class,
#' settings, start values and a hash of its distances); the variance function (class, formula,
#' values); method; max_iter; and the nlme version. Any change gives a new fit.
#' The formula is written into the stored call, so emmeans and predict see it; always
#' pass `data =` explicitly to emmeans. `cache_dir = NULL` fits without caching.
#' Returns the gls object with attributes "cache_key" and "fit_minutes".
fit_gls_cached <- function(formula, data, correlation = NULL, weights = NULL, method = "REML",
                           max_iter = 200L, cache_dir = NULL, label = NULL) {
  vars <- unique(c(all.vars(formula),
                   if (!is.null(weights)) all.vars(attr(weights, "formula")),
                   if (!is.null(correlation)) all.vars(attr(correlation, "formula"))))
  stopifnot(all(vars %in% names(data)))
  used <- lapply(data[vars], function(x) if (is.factor(x)) list(levels(x), as.integer(x)) else x)
  # Formulas enter the key as text: a formula object carries its environment, which differs
  # between sessions and would make every key new.
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
  # Only non-NULL arguments go into the call (emmeans reads call$weights); the formula and the
  # (small) variance function are written into the call, the correlation structure is not.
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
    saveRDS(m, path)
  }
  m
}

#' Spatial weights for Moran's I of residuals: k nearest neighbours (great-circle,
#' row-standardised) and one set per distance band [bands_km[i], bands_km[i+1]] km (NULL where a
#' band has no links). Build once per PA set; rows as in `coords`.
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

#' Moran's I of residuals `r` (in the row order of the weights): kNN permutation test
#' (moran.mc, `nsim`, after set.seed(seed)) and, if `bands = TRUE`, the distance-band
#' correlogram (moran.test, randomisation). Returns list(knn, bands) data frames.
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
