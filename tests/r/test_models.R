# Unit tests for modules/r/models.R (non-spatial gls on simulated data; seconds).

sim_model_data <- function(n = 400, seed = 21) {
  set.seed(seed)
  d <- data.frame(
    STATUS_YR_z = rnorm(n), log_area_z = rnorm(n), elevation_mean_z = rnorm(n),
    terrain_slope_z = rnorm(n), water_extent_pct_z = rnorm(n),
    gHM_mean = runif(n, 0, 0.8),
    IUCN_group = sample(c("strict", "non-strict", "unknown"), n, replace = TRUE),
    BIOME_NAME = sample(CFG$biome_order[1:4], n, replace = TRUE, prob = c(0.4, 0.3, 0.2, 0.1)),
    stringsAsFactors = FALSE)
  d$gHM_mean_z <- (d$gHM_mean - 0.3) / 0.2
  slope_b <- c(0.02, 0, -0.01, 0.03)[match(d$BIOME_NAME, CFG$biome_order)]
  d$y <- 0.5 + 0.01 * d$log_area_z + slope_b * d$gHM_mean_z +
    0.01 * (d$IUCN_group == "strict") * d$gHM_mean_z + rnorm(n, 0, 0.05)
  d$IUCN_group <- set_reference(d$IUCN_group, "strict", names(CFG$iucn_groups))
  d$BIOME_NAME <- set_reference(d$BIOME_NAME, CFG$biome_order[1], CFG$biome_order)
  d
}
fit_ns <- function(d, type) fit_gls_cached(model_formula("y", type), d)

test_that("model_formula and set_reference", {
  f <- model_formula("y", "full")
  expect_setequal(attr(terms(f), "term.labels"),
                  c(MODEL_COVARIATES, "gHM_mean_z", "IUCN_group", "BIOME_NAME",
                    "gHM_mean_z:IUCN_group", "gHM_mean_z:BIOME_NAME"))
  expect_false(any(grepl(":", attr(terms(model_formula("y", "add")), "term.labels"))))
  x <- set_reference(c("b", "c", "a", "b"), "c", c("a", "b", "c", "d"))
  expect_identical(levels(x), c("c", "a", "b"))
  expect_error(set_reference("z", "a", c("a", "b")))
})

test_that("coef_table and wald_tests match summary() and anova()", {
  d <- sim_model_data(); m <- fit_ns(d, "full")
  ct <- coef_table(m, "y")
  expect_equal(ct$estimate, unname(coef(m)))
  expect_equal(ct$conf.low, unname(ct$estimate - qt(0.975, m$dims$N - m$dims$p) * ct$std.error))
  w <- wald_tests(m, "y", hypotheses = TRUE)
  expect_identical(w$hypothesis, c("H2: gHM x IUCN", "H3: gHM x BIOME"))
  expect_equal(w$numDF, c(2, 3))
  expect_equal(w$denDF, rep(m$dims$N - m$dims$p, 2))
  expect_equal(w$F.value, anova(m, type = "marginal")[w$term, "F-value"])
})

test_that("hmi_slopes: biome slopes, IUCN slopes with proportional weights, contrasts", {
  d <- sim_model_data(); m <- fit_ns(d, "full")
  s <- hmi_slopes(m, d, "y", "nonspatial")
  b <- coef(m)
  # Slope of a biome = main slope + its interaction + IUCN interactions weighted by IUCN shares
  w_iucn <- prop.table(table(d$IUCN_group))
  iucn_part <- sum(w_iucn[-1] * b[paste0("gHM_mean_z:IUCN_group", names(w_iucn)[-1])])
  bio <- levels(d$BIOME_NAME)
  manual <- b[["gHM_mean_z"]] + c(0, b[paste0("gHM_mean_z:BIOME_NAME", bio[-1])]) + iucn_part
  expect_equal(s$biome$gHM_mean_z.trend, unname(manual))
  expect_identical(s$biome$BIOME_NAME, bio)
  expect_equal(s$biome$n_PA, as.integer(table(d$BIOME_NAME)))
  expect_equal(s$biome$sd_gHM_z, as.vector(tapply(d$gHM_mean_z, d$BIOME_NAME, sd)))
  # Contrasts vs the first biome = the interaction coefficients
  expect_equal(s$biome_contrasts$estimate, unname(b[paste0("gHM_mean_z:BIOME_NAME", bio[-1])]))
  expect_identical(s$biome_contrasts$contrast, paste(bio[-1], "-", bio[1]))
  expect_equal(nrow(s$iucn_contrasts), 3)
  expect_equal(unique(s$iucn$df), m$dims$N - m$dims$p)   # non-spatial: no covariance parameters
})

test_that("ref_contrasts equal coefficient differences, with Holm p", {
  d <- sim_model_data(); m <- fit_ns(d, "add")
  r <- ref_contrasts(m, d, "IUCN_group", "non-strict", "y")
  b <- coef(m)
  expect_identical(r$level, c("strict", "unknown"))
  expect_equal(r$estimate, c(0 - b[["IUCN_groupnon-strict"]],
                             b[["IUCN_groupunknown"]] - b[["IUCN_groupnon-strict"]]))
  expect_equal(r$p_holm, p.adjust(r$p.value, "holm"))
})

test_that("aicc_gls, pseudo_r2, spatial_params", {
  d <- sim_model_data(); m <- fit_ns(d, "add")
  ll <- logLik(m); k <- attr(ll, "df"); n <- nrow(d)
  expect_equal(aicc_gls(m), AIC(m) + 2 * k * (k + 1) / (n - k - 1))
  expect_equal(pseudo_r2(m, d$y), cor(fitted(m), d$y)^2)
  d$lon <- runif(nrow(d), 0, 5); d$lat <- runif(nrow(d), 0, 5)
  cs <- spatial_setup(d, value = 50, fixed = TRUE)
  msp <- fit_gls_cached(model_formula("y", "add"), d, cs)
  sp <- spatial_params(msp, m, "y")
  expect_true(sp$range_fixed); expect_equal(sp$range_km, 50); expect_true(is.na(sp$range_lo))
  expect_equal(sp$delta_AICc_nonspatial_minus_spatial, aicc_gls(m) - aicc_gls(msp))
})

test_that("hmi_predictions: back-transformed grid within each group's 1–99% HMI range", {
  d <- sim_model_data(); m <- fit_ns(d, "full")
  p <- hmi_predictions(m, d, "BIOME_NAME", "y", hmi_mean = 0.3, hmi_sd = 0.2)
  expect_equal(p$gHM_mean, p$gHM_mean_z * 0.2 + 0.3)
  q <- tapply(d$gHM_mean, d$BIOME_NAME, quantile, probs = c(0.01, 0.99))
  for (b in unique(p$BIOME_NAME)) {
    g <- p$gHM_mean[p$BIOME_NAME == b]
    expect_true(all(g >= q[[b]][1] & g <= q[[b]][2]), info = b)
  }
  expect_true(all(p$conf.low < p$predicted & p$predicted < p$conf.high))
})

test_that("precision_weights: method-of-moments tau2, floor at 1% of the median", {
  se <- c(0.1, 0.2, 0.3, 0.001)
  pw <- precision_weights(sigma = 0.5, se = se, label = "y")
  tau2 <- 0.25 - mean(se^2)
  expect_equal(pw$table$tau2, tau2)
  expect_equal(pw$v, pmax(se^2 + tau2, 0.01 * median(se^2 + tau2)))
  expect_equal(pw$table$share_sampling_var, mean(se^2) / 0.25)
  # σ² below the mean sampling variance: τ² = 0, and the floor applies
  pw0 <- precision_weights(sigma = 0.01, se = c(1e-6, 0.2, 0.3))
  expect_equal(pw0$table$tau2, 0)
  expect_equal(pw0$v[1], 0.01 * median(c(1e-12, 0.04, 0.09)))
})

test_that("fit_change_models / change_model_tables / table_sz", {
  d <- sim_model_data(300, seed = 22)
  d$lon <- runif(nrow(d), 0, 5); d$lat <- runif(nrow(d), 0, 5)
  d$se_y <- runif(nrow(d), 0.01, 0.05)
  d$IUCN_group <- set_reference(d$IUCN_group, "non-strict", names(CFG$iucn_groups))
  cs <- spatial_setup(d, value = 30, fixed = TRUE)
  cm <- fit_change_models(d, "y", "se_y", cs, label = "y")
  expect_equal(cm$tau$sigma2_unweighted, cm$fits$A$sigma^2)
  expect_equal(cm$data$v_precision, precision_weights(cm$fits$A$sigma, d$se_y)$v)
  # B: residual variance proportional to v (varFixed)
  expect_equal(unname(nlme::varWeights(cm$fits$B$modelStruct$varStruct)), 1 / sqrt(cm$data$v_precision))
  expect_null(fit_change_models(d, "y", "se_y", cs, fit_B = FALSE)$fits$B)

  tb <- change_model_tables(cm, "y", "y", list(iucn = "non-strict", biome = CFG$biome_order[1]))
  expect_setequal(unique(tb$coef$model), CHANGE_MODEL_LABELS)
  expect_equal(tb$spatial_params$n, nrow(d))
  expect_equal(tb$spatial_params$pseudoR2_spatial, cor(fitted(cm$fits$A), d$y)^2)
  sz <- table_sz(tb$coef, tb$contrasts)
  expect_equal(sum(sz$group == "Continuous predictor (per 1 SD)"), 2 * length(CONTINUOUS_LABELS))
  expect_true(all(c("strict vs non-strict", "unknown vs non-strict") %in% sz$term_label))
  # Contrasts vs the reference level of the fit are the coefficients
  a <- tb$contrasts[tb$contrasts$model == CHANGE_MODEL_LABELS[["A"]] & tb$contrasts$level == "strict", ]
  expect_equal(a$estimate, unname(coef(cm$fits$A)[["IUCN_groupstrict"]]))
})
