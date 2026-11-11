# Unit tests for modules/r/validation.R (simulated data).

test_that("null_pa_means applies the minimum years per metric", {
  py <- data.frame(WDPA_PID = rep(c("a", "b"), each = 6), year = rep(2003:2008, 2),
                   edge_extent = c(rep(0.5, 6), rep(0.6, 4), NA, NA), edge_intensity = 1)
  r <- null_pa_means(py, c("edge_extent", "edge_intensity"), 5)
  expect_equal(r$WDPA_PID, "a"); expect_equal(r$edge_extent_mean, 0.5)
  expect_equal(nrow(null_pa_means(py, c("edge_extent", "edge_intensity"), 4)), 2)
})

test_that("paired_null_summary: means, shares, Wilcoxon and Holm by group and pooled", {
  set.seed(4)
  d <- data.frame(BIOME_NAME = rep(c("Desert", "Tundra"), each = 40))
  d$d_ext <- c(rnorm(40, 0.05, 0.05), rnorm(40, 0, 0.05))
  d$d_int <- rnorm(80, 0.2, 0.3)
  s <- paired_null_summary(d, c("Edge extent" = "d_ext", "Edge intensity" = "d_int"), "BIOME_NAME",
                           c("Desert", "Tundra"), R = 500, seed = 1)
  expect_equal(nrow(s), 6)
  r <- s[s$metric == "Edge extent" & s$BIOME_NAME == "Desert", ]
  expect_equal(r$mean_diff, mean(d$d_ext[1:40])); expect_equal(r$median, median(d$d_ext[1:40]))
  expect_equal(r$share_positive, mean(d$d_ext[1:40] > 0))
  expect_equal(r$p_wilcoxon, wilcox.test(d$d_ext[1:40], exact = FALSE)$p.value)
  k <- s$metric == "Edge extent" & s$BIOME_NAME != "All PAs"
  expect_equal(s$p_wilcoxon_holm[k], p.adjust(s$p_wilcoxon[k], "holm"))
  expect_true(is.na(s$p_wilcoxon_holm[s$BIOME_NAME == "All PAs"][1]))
  expect_equal(s$n[s$BIOME_NAME == "All PAs"], c(80, 80))
  expect_true(r$ci_lo < r$mean_diff && r$mean_diff < r$ci_hi)
})

test_that("paired_null_summary with n_pop: PA-weighted row (stratified) appended, other rows unchanged", {
  set.seed(4)
  d <- data.frame(BIOME_NAME = rep(c("Desert", "Tundra"), each = 40))
  d$d_ext <- c(rnorm(40, 0.05, 0.05), rnorm(40, 0, 0.05))
  d$d_int <- rnorm(80, 0.2, 0.3)
  cols <- c("Edge extent" = "d_ext", "Edge intensity" = "d_int")
  s0 <- paired_null_summary(d, cols, "BIOME_NAME", c("Desert", "Tundra"), R = 500, seed = 1)
  n_pop <- c(Desert = 300, Tundra = 100)
  s <- paired_null_summary(d, cols, "BIOME_NAME", c("Desert", "Tundra"), R = 500, seed = 1, n_pop = n_pop)
  # biome rows and the equal-weight pooled row unchanged
  expect_equal(s[s$BIOME_NAME != "All PAs, PA-weighted", names(s0)], s0, ignore_attr = TRUE)
  # PA-weighted row: biome shares 0.75 / 0.25
  w <- s[s$BIOME_NAME == "All PAs, PA-weighted" & s$metric == "Edge extent", ]
  x1 <- d$d_ext[1:40]; x2 <- d$d_ext[41:80]
  expect_equal(w$mean_diff, 0.75 * mean(x1) + 0.25 * mean(x2))
  expect_equal(w$share_positive, 0.75 * mean(x1 > 0) + 0.25 * mean(x2 > 0))
  se <- sqrt(0.75^2 * var(x1) / 40 + 0.25^2 * var(x2) / 40)
  expect_equal(w$p_stratified_t, 2 * pt(-abs(w$mean_diff / se), 78))
  expect_true(w$ci_lo < w$mean_diff && w$mean_diff < w$ci_hi)
  expect_equal(w$n, 80)
  expect_true(is.na(w$p_wilcoxon))
  expect_equal(sum(s$BIOME_NAME == "All PAs, PA-weighted"), 2)
  expect_equal(sum(pooled_weights(d$BIOME_NAME, n_pop)), 1)
  expect_equal(weighted_median(c(1, 2, 3), c(0.2, 0.2, 0.6)), 3)
})
