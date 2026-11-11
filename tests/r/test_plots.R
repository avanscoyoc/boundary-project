# Unit tests for modules/r/plots.R: builders produce plots from small tables (no rendering to file).

builds <- function(p) {
  f <- tempfile(fileext = ".png")
  on.exit(unlink(f))
  ggplot2::ggsave(f, p, width = 8, height = 6, dpi = 30)
  file.exists(f)
}

test_that("fig3a_rows: order, reference, response-SD scaling (contrasts or coefficients)", {
  bo <- CFG$biome_order
  coef <- data.frame(response = "extent",
                     term = c("(Intercept)", names(TERM_LABELS), "IUCN_groupnon-strict", "IUCN_groupunknown",
                              paste0("BIOME_NAME", bo[-1])),
                     estimate = 0.02, conf.low = 0.01, conf.high = 0.03, p.value = 0.01)
  sdt <- data.frame(response = "extent", sd_response = 0.1)
  r <- fig3a_rows(coef, sdt, bo, ref = list(iucn = "strict", biome = bo[1]))
  expect_equal(nrow(r), 6 + 2 + 5)
  expect_equal(r$est, rep(0.2, nrow(r)))
  expect_identical(levels(r$term_label)[1], paste0("Biome: ", tolower(bo[6])))   # bottom row
  expect_identical(attr(r, "ref"), list(iucn = "strict", biome = bo[1]))
  expect_equal(unname(attr(r, "n_groups")), c(6, 2, 5))
  ctr <- data.frame(response = "extent", factor = c("IUCN_group", "IUCN_group", rep("BIOME_NAME", 5)),
                    level = c("strict", "unknown", setdiff(bo, "Boreal Forest")),
                    reference = c("non-strict", "non-strict", rep("Boreal Forest", 5)),
                    estimate = 0.01, lower.CL = 0, upper.CL = 0.02, p.value = 0.2)
  r2 <- fig3a_rows(coef, sdt, bo, contrasts = ctr)
  expect_true("Biome: grassland & shrubland" %in% r2$term_label)
  expect_false("Biome: boreal forest" %in% r2$term_label)
})

test_that("fig4a, slope-correlation and correlogram builders build", {
  set.seed(1)
  bo <- CFG$biome_order[1:3]
  grid <- expand.grid(metric = names(FIG4_COLORS), BIOME_NAME = c("All PAs", bo), stringsAsFactors = FALSE)
  sm <- transform(grid, n = 10, mean_change_z = rnorm(nrow(grid), 0, 0.05))
  sm$ci_lo_z <- sm$mean_change_z - 0.03; sm$ci_hi_z <- sm$mean_change_z + 0.03
  reps <- do.call(rbind, lapply(seq_len(nrow(sm)), function(i)
    data.frame(metric = sm$metric[i], BIOME_NAME = sm$BIOME_NAME[i], change_z = rnorm(50, sm$mean_change_z[i], 0.015))))
  expect_true(builds(fig4a(sm, reps, bo, pooled_label = "All PAs")))

  pa <- data.frame(BIOME_NAME = sample(bo, 200, TRUE), change_extent_z = rnorm(200))
  pa$change_intensity_z <- 0.5 * pa$change_extent_z + rnorm(200)
  ct <- slope_correlation(pa, "change_extent_z", "change_intensity_z", "BIOME_NAME", bo)
  expect_true(builds(fig_slope_correlation(pa, ct, bo)))

  mb <- data.frame(model = rep(c("a", "b"), each = 3), mid_km = c(25, 75, 175), I = c(0.1, 0.05, NA), p = 0.01)
  expect_true(builds(fig_moran_correlogram(mb)))
})

test_that("fig3_panels keeps the supported interactions; fig3 builds with 0 to 4 panels", {
  wald <- data.frame(response = rep(c("extent", "intensity"), each = 2),
                     term = rep(c("gHM_mean_z:IUCN_group", "gHM_mean_z:BIOME_NAME"), 2),
                     numDF = rep(c(2, 5), 2), denDF = 4570, F.value = c(6.6, 5.6, 1.3, 5.1),
                     p.value = c(0.001, 4e-5, 0.27, 1e-4))
  p <- fig3_panels(wald)
  expect_identical(paste(p$factor, p$response),
                   c("IUCN_group extent", "BIOME_NAME extent", "BIOME_NAME intensity"))
  expect_equal(nrow(fig3_panels(wald, alpha = 1e-6)), 0)
  expect_error(fig3_panels(wald[-1, ]))           # each response needs both interaction rows
  # with sensitivity runs: an interaction must have p < alpha in every run
  sens <- data.frame(run = rep(c("main", "a", "b"), each = 4), metric = rep(c("extent", "extent", "intensity", "intensity"), 3),
                     term = rep(c("H2: gHM x IUCN", "H3: gHM x BIOME"), 6),
                     p.value = c(0.001, 4e-5, 0.27, 1e-4,  0.01, 0.2, 0.3, 1e-3,  0.02, 0.01, 0.3, 0.04))
  ps <- fig3_panels(wald, sens = sens)
  expect_identical(paste(ps$factor, ps$response), c("IUCN_group extent", "BIOME_NAME intensity"))
  expect_equal(ps$p_max_sensitivity, c(0.02, 0.04))
  expect_equal(ps$n_runs, c(2, 2))

  bo <- CFG$biome_order
  coef <- data.frame(response = rep(c("extent", "intensity"), each = 13),
                     term = c(names(TERM_LABELS), "IUCN_groupnon-strict", "IUCN_groupunknown", paste0("BIOME_NAME", bo[-1])),
                     estimate = 0.02, conf.low = 0.01, conf.high = 0.03, p.value = 0.01)
  rows <- fig3a_rows(coef, data.frame(response = c("extent", "intensity"), sd_response = 0.1), bo,
                     ref = list(iucn = "strict", biome = bo[1]))
  g <- seq(0, 0.8, length.out = 5)
  pi <- expand.grid(gHM_mean = g, IUCN_group = IUCN_LEVELS, response = c("extent", "intensity"), stringsAsFactors = FALSE)
  pb <- expand.grid(gHM_mean = g, BIOME_NAME = bo, response = c("extent", "intensity"), stringsAsFactors = FALSE)
  for (x in c("pi", "pb")) {
    y <- get(x); y$predicted <- 0.55 + 0.1 * y$gHM_mean; y$conf.low <- y$predicted - 0.01; y$conf.high <- y$predicted + 0.01
    assign(x, y)
  }
  si <- expand.grid(IUCN_group = IUCN_LEVELS, response = c("extent", "intensity"), stringsAsFactors = FALSE); si$p.value <- 0.01
  sb <- expand.grid(BIOME_NAME = bo, response = c("extent", "intensity"), stringsAsFactors = FALSE)
  sb$p.value <- 0.2; sb$n_PA <- 100
  for (a in c(1e-6, 0.05, 1)) {
    pn <- fig3_panels(wald, alpha = a)
    expect_true(builds(fig3(rows, pi, pb, si, sb, wald, bo, pn)), info = nrow(pn))
  }
  expect_equal(fig3_height(3), 12.5)
  # fixed axes: a CI bound beyond the drawn area warns
  hi <- pi[pi$response == "extent", ]; hi$conf.high[1] <- 0.7
  expect_warning(fig3_axes(hi, "extent"), "outside the fixed axes")
  expect_silent(fig3_axes(pi[pi$response == "extent", ], "extent"))
})

test_that("Fig. 4B: placeholders + series, combined Fig. 4, single panels; values kept (B5)", {
  set.seed(2)
  cases <- data.frame(WDPA_PID = c("p1", "p2"), label = c("One, X · Grassland", "Two, Y · Trop. For."), stringsAsFactors = FALSE)
  py <- expand.grid(WDPA_PID = cases$WDPA_PID, year = 2003:2025, stringsAsFactors = FALSE)
  py$edge_extent <- 0.6 + 0.005 * (py$year - 2003) * ifelse(py$WDPA_PID == "p1", 1, -1) + rnorm(nrow(py), 0, 0.02)
  py$edge_intensity <- 1 + 0.01 * (py$year - 2003) + rnorm(nrow(py), 0, 0.05)
  s <- case_series(py, cases, c(extent = "edge_extent", intensity = "edge_intensity"))
  expect_silent(p <- fig4b_series(s))
  # coord_cartesian, not scale limits: every value reaches the plot and the trend lines
  b <- ggplot2::ggplot_build(p)
  expect_equal(sum(!is.na(b$data[[1]]$y)), 2 * nrow(s))
  expect_equal(levels(p$data$label), cases$label)
  # the secondary axis maps the rescaled intensity back exactly
  l <- .fig4b_long(s, FIG4B_Y)
  int <- l$data$metric == "Edge intensity"
  expect_equal(l$back(l$data$value_scaled[int]), l$data$value[int])
  expect_equal(range(l$data$value_scaled[int]), range(s$edge_extent))
  expect_true(builds(fig4b(s)))
  sm <- data.frame(metric = rep(names(FIG4_COLORS), 2), BIOME_NAME = rep(c("All PAs", "Desert"), each = 2), n = 10,
                   mean_change_z = 0.1, ci_lo_z = 0.05, ci_hi_z = 0.15)
  reps <- data.frame(metric = rep(sm$metric, each = 20), BIOME_NAME = rep(sm$BIOME_NAME, each = 20), change_z = rnorm(80, 0.1, 0.02))
  expect_true(builds(fig4(fig4a(sm, reps, "Desert", "All PAs"), fig4b(s))))
  expect_true(builds(fig4b_single(s, "p2")))
  expect_error(fig4b_single(s, "nope"))
  hi <- s; hi$edge_extent[1] <- 0.99
  expect_warning(fig4b_series(hi), "outside the y range")
})

test_that("fig2_layers: one row per analysis PA (WDPA_PID order) and the case PAs", {
  sq <- function(x) sf::st_polygon(list(rbind(c(x, 0), c(x + 1, 0), c(x + 1, 1), c(x, 1), c(x, 0))))
  polys <- sf::st_sf(WDPA_PID = c("3", "1", "2", "9"), geometry = sf::st_sfc(sq(3), sq(1), sq(2), sq(9), crs = 4326))
  pa <- data.frame(WDPA_PID = c("2", "1", "3"), NAME = c("B", "A", "C"), ISO3 = "XXX", BIOME_NAME = "Desert",
                   IUCN_CAT = "II", IUCN_group = "strict", STATUS_YR = 1990, AREA_KM2 = 1,
                   edge_extent_mean = c(0.2, 0.1, 0.3), edge_intensity_mean = 1, stringsAsFactors = FALSE)
  tr <- data.frame(WDPA_PID = c("1", "2", "3"), slope_extent = c(0.01, 0.02, 0.03), p_fdr_extent = 0.01,
                   trend_extent = "increase", change_extent_z = 1, slope_intensity = 0, p_fdr_intensity = 0.5,
                   trend_intensity = "no_trend", change_intensity_z = 0, stringsAsFactors = FALSE)
  cases <- data.frame(WDPA_PID = "3", label = "C, X · Desert", stringsAsFactors = FALSE)
  ml <- fig2_layers(polys, pa, tr, cases)
  expect_identical(ml$pas$WDPA_PID, c("1", "2", "3"))
  expect_equal(ml$pas$slope_extent, c(0.01, 0.02, 0.03))
  expect_equal(ml$pas$edge_extent_mean, c(0.1, 0.2, 0.3))
  expect_true(all(sf::st_equals(ml$pas, polys[match(ml$pas$WDPA_PID, polys$WDPA_PID), ], sparse = FALSE)[cbind(1:3, 1:3)]))
  expect_identical(ml$pas$trend_combined[1], "extent_increase_intensity_no_trend")
  expect_identical(ml$cases$Name, "C, X · Desert")
  expect_identical(ml$cases$WDPA_PID, "3"); expect_identical(ml$cases$PA_NAME, "C")
  expect_error(fig2_layers(polys[-1, ], pa, tr, cases))
  f <- tempfile(fileext = ".kml")
  sf::st_write(ml$cases, f, driver = "KML", quiet = TRUE)
  expect_identical(sf::st_read(f, quiet = TRUE)$Name, "C, X · Desert")
})
