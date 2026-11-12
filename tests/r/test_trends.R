sim_series <- function(n_pa = 40, seed = 1) {
  set.seed(seed)
  d <- do.call(rbind, lapply(seq_len(n_pa), function(k) {
    yrs <- 2003:2025
    data.frame(WDPA_PID = sprintf("pa%03d", k), year = yrs,
               value = 0.5 + rnorm(1, 0, 0.003) * (yrs - 2014) + rnorm(length(yrs), 0, 0.05))
  }))
  d$value[sample(nrow(d), round(nrow(d) / 6))] <- NA   # scattered missing years
  d$value[d$WDPA_PID == "pa002"][1:20] <- NA   # ≤ 3 years left: dropped
  d$value[d$WDPA_PID == "pa003"] <- c(rep(NA, 18), 0.5 + 0.01 * (1:5) + rnorm(5, 0, 0.01))  # exactly 5: kept
  d[sample(nrow(d)), ]                         # rows in no particular order
}

test_that("ols_trends equals lm() per PA", {
  d <- sim_series()
  tr <- ols_trends(d, "value", min_years = 5)
  expect_false("pa002" %in% tr$WDPA_PID)
  expect_true("pa003" %in% tr$WDPA_PID)
  # first-appearance order among rows with a value
  expect_identical(tr$WDPA_PID, unique(d$WDPA_PID[!is.na(d$value) & d$WDPA_PID != "pa002"]))

  for (pid in tr$WDPA_PID) {
    s <- summary(lm(value ~ year, data = d[d$WDPA_PID == pid, ]))$coefficients["year", ]
    r <- tr[tr$WDPA_PID == pid, ]
    expect_equal(c(r$slope, r$se, r$stat, r$p_value), unname(s), tolerance = 1e-10, info = pid)
    expect_equal(r$n_years, sum(!is.na(d$value[d$WDPA_PID == pid])))
  }
})

test_that("bh_classes adjusts like p.adjust and classes by sign", {
  est <- c(0.2, -0.1, 0.3, NA, -0.4, 0.01)
  p   <- c(0.001, 0.004, 0.2, NA, 0.03, 0.0001)
  out <- bh_classes(est, p, fdr = 0.05, levels = c("up", "none", "down"))
  expect_equal(out$p_fdr, p.adjust(p, "BH"))
  expect_identical(out$class, c("up", "down", "none", NA, "down", "up"))
  # NAs are left out of the adjustment: same as adjusting the non-missing values alone
  expect_equal(out$p_fdr[!is.na(p)], p.adjust(p[!is.na(p)], "BH"))
})

test_that("yuepilon_trends equals zyp per PA", {
  d <- sim_series(n_pa = 8, seed = 2)
  yp <- yuepilon_trends(d, "value", min_years = 5)
  expect_false("pa002" %in% yp$WDPA_PID)
  for (pid in yp$WDPA_PID) {
    s <- d[d$WDPA_PID == pid & !is.na(d$value), ]
    z <- zyp::zyp.trend.vector(s$value, s$year, method = "yuepilon", conf.intervals = FALSE)
    r <- yp[yp$WDPA_PID == pid, ]
    expect_equal(c(r$sen, r$p), unname(c(z[["trend"]], z[["sig"]])), info = pid)
  }
})

test_that("pa_trends names the columns by metric", {
  d <- sim_series(n_pa = 6, seed = 3)
  tr <- pa_trends(d, "value", "extent", min_years = 5, fdr = 0.05, levels = CFG$trend$levels)
  expect_named(tr, c("WDPA_PID", "slope_extent", "se_extent", "stat_extent", "p_value_extent",
                     "p_fdr_extent", "trend_extent", "sen_extent", "p_zyp_extent",
                     "p_fdr_zyp_extent", "trend_extent_zyp"))
  expect_true(all(tr$trend_extent %in% CFG$trend$levels))
})

# ── Standardised change and the summaries built on the trends ─────────────────
sim_pa <- function(n = 120, seed = 4) {
  set.seed(seed)
  g <- sample(c("A", "B", "C"), n, replace = TRUE, prob = c(0.5, 0.3, 0.2))
  data.frame(WDPA_PID = sprintf("pa%03d", seq_len(n)), group = g,
             area = rlnorm(n, 5, 1),
             s1 = rnorm(n, ifelse(g == "A", 0.002, 0), ifelse(g == "C", 0.004, 0.002)),
             s2 = rnorm(n, 0, 0.003), base = runif(n, 0.3, 0.8),
             stringsAsFactors = FALSE)
}

test_that("standardised_change rescales slope and SE by n_span / sd", {
  ch <- standardised_change(c(0.01, -0.02), c(0.001, 0.002), n_span = 22, sd = 0.1)
  expect_equal(ch$change, c(2.2, -4.4))
  expect_equal(ch$se_change, c(0.22, 0.44))
  expect_error(standardised_change(1, 1, 22, 0))
})

test_that("boot_means equals boot::boot by group and keeps group results under pooled = 'append'", {
  d <- sim_pa(); d$s1[c(3, 9)] <- NA
  cols <- c(m1 = "s1", m2 = "s2")
  lv <- c("C", "A", "B", "Z")   # Z absent; order of levels = run order
  b <- boot_means(d, cols, "group", lv, R = 200, seed = 7)
  expect_identical(b$summary$group, rep(c("C", "A", "B"), 2))
  expect_identical(b$summary$metric, rep(c("m1", "m2"), each = 3))
  expect_equal(nrow(b$reps), 6 * 200)

  # The same stream by hand: set.seed once, then boot() per metric and group in order
  set.seed(7)
  for (m in names(cols)) for (g in c("C", "A", "B")) {
    x <- d[[cols[[m]]]][d$group == g & !is.na(d[[cols[[m]]]])]
    bt <- boot::boot(x, function(v, i) mean(v[i]), R = 200)
    r <- b$summary[b$summary$metric == m & b$summary$group == g, ]
    expect_equal(c(r$n, r$mean, r$ci_lo, r$ci_hi),
                 c(length(x), mean(x), boot::boot.ci(bt, type = "perc")$percent[4:5]))
  }

  a <- boot_means(d, cols, "group", lv, R = 200, seed = 7, pooled = "append", pooled_label = "All")
  expect_equal(a$summary[1:6, ], b$summary)
  expect_identical(a$summary$group[7:8], c("All", "All"))
  expect_equal(a$summary$n[7:8], c(sum(!is.na(d$s1)), nrow(d)))

  o <- boot_means(d, cols, "group", lv, R = 200, seed = 7, pooled = "only", pooled_label = "All")
  expect_identical(o$summary$group, c("All", "All"))
  expect_equal(o$summary$mean, c(mean(d$s1, na.rm = TRUE), mean(d$s2)))
})

test_that("boot_means with weights: weighted mean; equal weights give the unweighted result", {
  d <- sim_pa()
  w <- boot_means(d, c(m1 = "s1"), "group", c("A", "B", "C"), R = 100, seed = 1, weights = "area")
  expect_equal(w$summary$mean[1], weighted.mean(d$s1[d$group == "A"], d$area[d$group == "A"]))
  d$one <- 1
  w1 <- boot_means(d, c(m1 = "s1"), "group", c("A", "B", "C"), R = 100, seed = 1, weights = "one")
  u  <- boot_means(d, c(m1 = "s1"), "group", c("A", "B", "C"), R = 100, seed = 1)
  expect_equal(w1$summary, u$summary)
})

test_that("biome_gls: group means, contrasts, letters and sigma", {
  d <- sim_pa(300, seed = 5)
  d$y <- d$s1 * 100
  r <- biome_gls(d, "y", "group", c("B", "A", "C"))
  expect_identical(r$means$group, c("B", "A", "C"))
  # One factor: the estimated means are the group means
  expect_equal(r$means$emmean, as.vector(tapply(d$y, d$group, mean)[c("B", "A", "C")]))
  expect_identical(r$pairs$contrast, c("B - A", "B - C", "A - C"))
  expect_equal(r$omnibus$numDF, 2)
  expect_equal(r$omnibus$denDF, nrow(d) - 3)
  expect_true(all(nchar(r$letters$letters) > 0))
  # varIdent residual SD per group ≈ the sample SD of the group (REML, one factor)
  expect_equal(r$sigma$resid_sd, as.vector(tapply(d$y, d$group, sd)[c("B", "A", "C")]),
               tolerance = 1e-3)
  # A clearly different group gets a letter of its own
  d$y[d$group == "C"] <- d$y[d$group == "C"] + 5
  r2 <- biome_gls(d, "y", "group", c("B", "A", "C"))
  expect_false(any(strsplit(r2$letters$letters[3], "")[[1]] %in%
                   strsplit(paste(r2$letters$letters[1:2], collapse = ""), "")[[1]]))
})

test_that("slope_correlation equals cor.test, pooled first", {
  d <- sim_pa(); d$s2 <- d$s2 + d$s1; d$s2[5] <- NA
  r <- slope_correlation(d, "s1", "s2", "group", c("A", "B", "C"), pooled_label = "All")
  expect_identical(r$stratum, c("All", "A", "B", "C"))
  ok <- !is.na(d$s2)
  ct <- cor.test(d$s1[ok], d$s2[ok])
  expect_equal(c(r$n[1], r$r_pearson[1], r$r_lo[1], r$r_hi[1]),
               c(sum(ok), unname(ct$estimate), ct$conf.int))
  expect_equal(r$rho_spearman[2], cor(d$s1[d$group == "A" & ok], d$s2[d$group == "A" & ok],
                                      method = "spearman"))
})

test_that("class_counts and trend_agreement count classes by group", {
  lv <- CFG$trend$levels
  d <- data.frame(group = c("A", "A", "A", "B", "B", NA),
                  ols = c(lv[1], lv[2], lv[2], lv[3], NA, lv[1]),
                  zyp = c(lv[1], lv[1], lv[2], lv[3], lv[2], lv[2]), stringsAsFactors = FALSE)
  cc <- class_counts(d, c(m = "ols"), "group", c("A", "B"), lv, pooled_label = "All")
  expect_equal(nrow(cc), 3 * 3)
  expect_equal(cc$n[cc$group == "All"], c(2, 2, 1))     # the NA-group PA counts in All only
  expect_equal(cc$n[cc$group == "B"], c(0, 0, 1))
  expect_equal(as.vector(tapply(cc$pct, cc$group, sum)), c(100, 100, 100))

  ag <- trend_agreement(d, list(m = c("ols", "zyp")), "group", c("A", "B"), lv, pooled_label = "All")
  a <- ag$agreement
  expect_equal(a$n, c(5, 3, 1))
  expect_equal(a$n_concordant, c(3, 2, 1))
  expect_equal(a$pct_increase_zyp[a$group == "A"], 200 / 3)
  expect_equal(sum(ag$crosstab$n), nrow(d))
})

test_that("percent_change uses the mean PA baseline over PAs with a slope", {
  d <- sim_pa(); d$s1[1:4] <- NA
  summ <- data.frame(metric = "m", group = c("A", "All"), n = 0, mean = 0.001,
                     ci_lo = 0.0005, ci_hi = 0.002)
  pc <- percent_change(summ, d, c(m = "s1"), c(m = "base"), "group", n_span = 22, pooled_label = "All")
  bA <- mean(d$base[d$group == "A" & !is.na(d$s1)])
  bAll <- mean(d$base[!is.na(d$s1)])
  expect_equal(pc$baseline, c(bA, bAll))
  expect_equal(pc$pct_change, 100 * 0.001 * 22 / c(bA, bAll))
  expect_equal(pc$pct_hi, 100 * 0.002 * 22 / c(bA, bAll))
})

test_that("cohen_kappa and paired_agreement (sensitivity S1)", {
  lv <- c("increase", "no_trend", "decrease")
  a <- c(rep("no_trend", 8), "increase", "decrease")
  expect_equal(cohen_kappa(a, a, lv), 1)
  b <- a; b[9] <- "no_trend"
  # po = 0.9; marginals a: (0.1, 0.8, 0.1), b: (0, 0.9, 0.1) -> pe = 0.72 + 0.01 = 0.73
  expect_equal(cohen_kappa(a, b, lv), (0.9 - 0.73) / (1 - 0.73))
  expect_true(is.na(cohen_kappa(rep("no_trend", 5), rep("no_trend", 5), lv)))
  expect_equal(cohen_kappa(c(a, NA), c(b, "increase"), lv), cohen_kappa(a, b, lv))
  set.seed(3); x <- rnorm(200)
  r <- paired_agreement(x, x + 0.5, scale = 2, R = 500, seed = 1)
  expect_equal(r$n, 200); expect_equal(r$r, 1); expect_equal(r$mean_diff_sd, 0.25)
  expect_equal(c(r$diff_lo, r$diff_hi), c(0.25, 0.25))
  y <- x + rnorm(200, 0.1, 0.3)
  r2 <- paired_agreement(x, y, 1, 1000, 1)
  expect_true(r2$diff_lo < r2$mean_diff_sd && r2$mean_diff_sd < r2$diff_hi)
  expect_identical(r2, paired_agreement(x, y, 1, 1000, 1))
})

test_that("case_series and case_summary: panel order, % of the PA's own mean", {
  cases <- data.frame(WDPA_PID = c("b", "a"), label = c("B, X · Grassland", "A, Y · Desert"), stringsAsFactors = FALSE)
  py <- expand.grid(WDPA_PID = c("a", "b", "c"), year = 2003:2005, stringsAsFactors = FALSE)
  py$edge_extent <- seq_len(nrow(py)) / 10; py$edge_intensity <- seq_len(nrow(py))
  s <- case_series(py, cases, c(extent = "edge_extent", intensity = "edge_intensity"))
  expect_identical(s$WDPA_PID, rep(c("b", "a"), each = 3))
  expect_identical(s$panel, rep(1:2, each = 3))
  expect_identical(s$year, rep(2003:2005, 2))
  expect_equal(s$edge_extent[s$WDPA_PID == "a" & s$year == 2004], py$edge_extent[py$WDPA_PID == "a" & py$year == 2004])
  expect_identical(names(s), c("panel", "WDPA_PID", "label", "year", "edge_extent", "edge_intensity"))
  expect_error(case_series(py, rbind(cases, data.frame(WDPA_PID = "z", label = "Z")), c(extent = "edge_extent")))

  d <- data.frame(WDPA_PID = c("a", "b", "c"), NAME = c("A", "B", "C"), edge_extent_mean = c(0.5, 0.8, 0.6),
                  slope_extent = c(0.01, -0.004, 0), p_fdr_extent = 0.01, trend_extent = c("increase", "decrease", "no_trend"),
                  change_extent_z = c(1, -0.4, 0), stringsAsFactors = FALSE)
  cs <- case_summary(d, cases, c(extent = "edge_extent"), n_span = 22)
  expect_identical(cs$WDPA_PID, c("b", "a")); expect_identical(cs$NAME, c("B", "A")); expect_identical(cs$panel, 1:2)
  expect_equal(cs$change_extent, c(-0.004, 0.01) * 22)
  expect_equal(cs$pct_change_extent, 100 * c(-0.004 * 22 / 0.8, 0.01 * 22 / 0.5))
  expect_identical(cs$trend_extent, c("decrease", "increase"))
  expect_error(case_summary(d[-1, ], cases, c(extent = "edge_extent"), 22))
})

test_that("residual_autocorrelation: OLS residuals, consecutive-year pairs, min_years; summary", {
  set.seed(5)
  yrs <- 2003:2025
  e_ar <- as.numeric(stats::arima.sim(list(ar = 0.8), length(yrs)))
  d <- rbind(data.frame(WDPA_PID = "ar", year = yrs, y = 0.5 + 0.01 * (yrs - 2003) + 0.05 * e_ar),
             data.frame(WDPA_PID = "alt", year = yrs, y = 0.3 + 0.02 * rep(c(1, -1), length.out = length(yrs))),
             data.frame(WDPA_PID = "gap", year = c(2003:2010, 2012:2020), y = rnorm(17)),
             data.frame(WDPA_PID = "short", year = 2003:2006, y = rnorm(4)))
  d <- d[sample(nrow(d)), ]                                  # order must not matter
  a <- residual_autocorrelation(d, "y", min_years = 5)
  expect_setequal(a$WDPA_PID, c("ar", "alt", "gap"))         # "short" has < 5 years
  ref <- function(pid) {                                     # by hand, from lm
    x <- d[d$WDPA_PID == pid, ]; x <- x[order(x$year), ]
    e <- stats::residuals(stats::lm(y ~ year, x)); k <- which(diff(x$year) == 1)
    sum(e[k + 1] * e[k]) / sum(e^2)
  }
  for (p in a$WDPA_PID) expect_equal(a$r1[a$WDPA_PID == p], ref(p), info = p)
  expect_equal(a$n_pairs[a$WDPA_PID == "gap"], 15L)          # 16 differences, one gap
  expect_true(a$r1[a$WDPA_PID == "ar"] > 0.3)
  expect_true(a$r1[a$WDPA_PID == "alt"] < -0.8)
  expect_equal(a$bound[a$WDPA_PID == "ar"], 1.96 / sqrt(23))

  ac <- data.frame(metric = "m", BIOME_NAME = c("A", "A", "B", "B"), r1 = c(0.5, 0.1, -0.6, NA), bound = 0.4)
  s <- autocorrelation_summary(ac, "BIOME_NAME", c("B", "A"), "All")
  expect_identical(s$BIOME_NAME, c("All", "B", "A"))
  expect_equal(s$n, c(3, 1, 2))
  expect_equal(s$pct_positive_above_bound, c(100 / 3, 0, 50))
  expect_equal(s$pct_abs_above_bound, c(200 / 3, 100, 50))
  expect_equal(s$median_r1[1], 0.1)
})
