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
