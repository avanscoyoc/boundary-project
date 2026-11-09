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
