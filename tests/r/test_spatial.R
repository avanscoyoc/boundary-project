# Unit tests for modules/r/spatial.R and the vectorised distances in modules/r/corHaversine.R.

# The original distance function (PA_New_Analysis/scripts/HaversineLMEfunctions.R:47-57),
# verbatim, as the reference. It uses the unchanged scalar haversine() from corHaversine.R.
haversineDist_loop <- function(xy, radians = FALSE) {
  if (ncol(xy) > 2) stop("Input must have two columns (longitude and latitude)")
  if (!radians) xy <- xy * pi / 180
  hMat <- matrix(NA, ncol = nrow(xy), nrow = nrow(xy))
  for (i in 1:nrow(xy)) {
    for (j in i:nrow(xy)) {
      hMat[j, i] <- haversine(xy[i, 1], xy[j, 1], xy[i, 2], xy[j, 2])
    }
  }
  as.dist(hMat)
}

sim_coords <- function(n, seed) {
  set.seed(seed)
  xy <- cbind(runif(n, -180, 180), runif(n, -90, 90))
  xy[2, ] <- -xy[1, ] + c(180, 0) * sign(xy[1, 1])   # antipodal-ish pair: sqrt(a) ≈ 1
  xy[3, ] <- xy[1, ] + c(1e-5, 0)                      # nearly coincident pair
  xy[4, ] <- c(179.99, 10); xy[5, ] <- c(-179.99, 10)  # across the antimeridian
  xy
}

test_that("vectorised haversineDist is bit-identical to the original loop (300 points)", {
  xy <- sim_coords(300, 11)
  v <- haversineDist(xy)
  l <- haversineDist_loop(xy)
  expect_identical(as.vector(v), as.vector(l))          # values and as.dist order, bit for bit
  expect_identical(attr(v, "Size"), 300L)
  expect_s3_class(v, "dist")
  expect_identical(as.vector(haversineDist(xy * pi / 180, radians = TRUE)),
                   as.vector(haversineDist_loop(xy * pi / 180, radians = TRUE)))
  # Block size does not matter (blocks split the columns of the lower triangle)
  expect_identical(as.vector(haversineDist(xy, block = 1)), as.vector(v))
  expect_identical(as.vector(haversineDist(xy, block = 1000)), as.vector(v))
  # Sanity: a quarter meridian is 6371 · π/2 km
  expect_equal(as.vector(haversineDist(rbind(c(0, 0), c(0, 90)))), 6371 * pi / 2)
  expect_length(as.vector(haversineDist(matrix(c(1, 2), 1))), 0)
})

test_that("the original file's haversineDist gives the same distances", {
  f <- file.path(CFG$paths$legacy_root, "scripts", "HaversineLMEfunctions.R")
  skip_if_not(file.exists(f), "PA_New_Analysis/ not present")
  env <- new.env(parent = globalenv())
  sys.source(f, envir = env)
  xy <- sim_coords(200, 12)
  expect_identical(as.vector(haversineDist(xy)), as.vector(env$haversineDist(xy)))
})

test_that("Initialize(corHaversine) uses the vectorised distances", {
  xy <- sim_coords(120, 13)
  cs <- spatial_setup(data.frame(lon = xy[, 1], lat = xy[, 2]), value = 200)
  expect_identical(as.vector(attr(cs, "covariate")), as.vector(haversineDist_loop(xy)))
  expect_s3_class(cs, "corHaversine")
  # Zero distances stop corHaversine (hence jitter_duplicates)
  xy[7, ] <- xy[6, ]
  expect_error(spatial_setup(data.frame(lon = xy[, 1], lat = xy[, 2]), value = 200), "zero distances")
})

test_that("jitter_duplicates moves only later duplicates, as the old loop", {
  set.seed(3)
  d <- data.frame(lon = round(runif(50, -10, 10), 2), lat = round(runif(50, -10, 10), 2))
  d[c(10, 20, 30), c("lon", "lat")] <- d[c(1, 1, 2), c("lon", "lat")]
  j <- jitter_duplicates(d, jitter_deg = 1e-4, seed = 1)
  moved <- which(j$lon != d$lon | j$lat != d$lat)
  expect_identical(moved, c(10L, 20L, 30L))
  expect_true(all(abs(j$lon - d$lon) <= 1e-4), all(abs(j$lat - d$lat) <= 1e-4))
  expect_false(anyDuplicated(paste(round(j$lon, 6), round(j$lat, 6))) > 0)
  expect_equal(attr(j, "n_jittered"), 3)
  # The old loop (03:131-141) by hand gives the same coordinates
  o <- d; key <- function(d) paste(round(d$lon, 6), round(d$lat, 6))
  dup <- duplicated(key(o)); set.seed(1); tries <- 0
  while (any(dup) && tries < 10) {
    o$lon[dup] <- o$lon[dup] + runif(sum(dup), -1e-4, 1e-4)
    o$lat[dup] <- o$lat[dup] + runif(sum(dup), -1e-4, 1e-4)
    dup <- duplicated(key(o)); tries <- tries + 1
  }
  expect_identical(j$lon, o$lon); expect_identical(j$lat, o$lat)
  expect_identical(jitter_duplicates(d, 1e-4, seed = 1)$lon, j$lon)
})

test_that("spatial_setup and fit_gls_cached cache by content", {
  cache <- file.path(tempdir(), "spatial_cache_test"); unlink(cache, recursive = TRUE)
  set.seed(14)
  n <- 150
  d <- data.frame(lon = runif(n, 0, 10), lat = runif(n, 0, 10), x = rnorm(n),
                  g = factor(sample(c("a", "b", "c"), n, replace = TRUE), levels = c("a", "b", "c")))
  d$y <- 0.5 * d$x + as.integer(d$g) * 0.2 + sin(d$lon / 2) + rnorm(n, 0, 0.3)

  cs1 <- spatial_setup(d, value = 200, cache_dir = cache)
  cs2 <- spatial_setup(d, value = 200, cache_dir = cache)
  for (a in c("covariate", "factor", "logDet")) expect_identical(attr(cs1, a), attr(cs2, a))
  expect_identical(as.numeric(cs1), as.numeric(cs2))
  expect_length(list.files(cache, "^corsp_"), 1)
  d_moved <- d; d_moved$lon[1] <- d_moved$lon[1] + 1e-6
  spatial_setup(d_moved, value = 200, cache_dir = cache)
  spatial_setup(d, value = 100, fixed = TRUE, cache_dir = cache)
  expect_length(list.files(cache, "^corsp_"), 3)        # coordinates and settings are in the key

  f <- y ~ x + g
  m1 <- fit_gls_cached(f, d, correlation = cs1, cache_dir = cache)
  m2 <- fit_gls_cached(f, d, correlation = cs1, cache_dir = cache)
  expect_identical(coef(m1), coef(m2))
  expect_identical(attr(m1, "cache_key"), attr(m2, "cache_key"))
  expect_length(list.files(cache, "^gls_"), 1)
  # Same as a direct gls fit
  m0 <- nlme::gls(f, data = d, correlation = cs1, method = "REML",
                  control = nlme::glsControl(maxIter = 200, msMaxIter = 200))
  expect_equal(coef(m1), coef(m0)); expect_equal(m1$sigma, m0$sigma)
  # The formula is in the stored call, and emmeans works with data = given
  expect_equal(eval(m1$call$model), f)
  expect_s4_class(emmeans::emmeans(m1, ~ g, data = d), "emmGrid")
  mw <- fit_gls_cached(f, d, weights = nlme::varIdent(form = ~ 1 | g))
  expect_s4_class(emmeans::emmeans(mw, ~ g, data = d), "emmGrid")

  # Any change of data, reference level, formula, structure or method → a new fit
  d2 <- d; d2$y[5] <- d2$y[5] + 1e-9
  d3 <- d; d3$g <- relevel(d3$g, "b")
  k <- c(attr(m1, "cache_key"),
         attr(fit_gls_cached(f, d2, correlation = cs1, cache_dir = cache), "cache_key"),
         attr(fit_gls_cached(f, d3, correlation = cs1, cache_dir = cache), "cache_key"),
         attr(fit_gls_cached(y ~ x, d, correlation = cs1, cache_dir = cache), "cache_key"),
         attr(fit_gls_cached(f, d, cache_dir = cache), "cache_key"),
         attr(fit_gls_cached(f, d, correlation = cs1, method = "ML", cache_dir = cache), "cache_key"),
         attr(fit_gls_cached(f, d, weights = nlme::varIdent(form = ~ 1 | g), cache_dir = cache),
              "cache_key"))
  expect_false(anyDuplicated(k) > 0)
  expect_length(list.files(cache, "^gls_"), length(k))
  # Unused columns are not in the key
  d4 <- d; d4$unused <- 1
  expect_identical(attr(fit_gls_cached(f, d4, correlation = cs1, cache_dir = cache), "cache_key"), k[1])
  unlink(cache, recursive = TRUE)
})

test_that("moran_residuals equals spdep's tests", {
  set.seed(15)
  n <- 200
  d <- data.frame(lon = runif(n, 0, 20), lat = runif(n, 0, 20))
  r <- sin(d$lon / 3) + rnorm(n, 0, 0.5)
  w <- suppressWarnings(moran_weights(d, k = 8, bands_km = c(0, 100, 500, 5000)))  # sparse bands
  out <- moran_residuals(r, w, nsim = 99, seed = 1, label = "m")
  set.seed(1)
  mc <- spdep::moran.mc(r, w$knn, nsim = 99)
  expect_equal(out$knn$I, unname(mc$statistic)); expect_equal(out$knn$p, mc$p.value)
  expect_equal(nrow(out$bands), 3)
  mt <- spdep::moran.test(r, w$bands[[2]]$lw, zero.policy = TRUE)
  expect_equal(out$bands$I[2], unname(mt$estimate[1]))
  expect_true(out$knn$I > 0)
  expect_equal(out$bands$mid_km, c(50, 300, 2750))
})
