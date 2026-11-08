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
