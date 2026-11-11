# Fast parity tests: the new functions on the old frozen inputs reproduce the old outputs/tables/.
# Sourced by tests/r/test_parity.R. Each block skips when PA_New_Analysis/ is absent.

skip_if_no_legacy <- function() {
  testthat::skip_if_not(legacy_available(), "PA_New_Analysis/ not present")
}

# Loaded once; reading the raw CSV and the map geopackage takes a few seconds.
legacy <- local({
  cache <- NULL
  function() {
    if (is.null(cache)) {
      pa <- legacy_pa()
      cache <<- list(
        annual   = legacy_annual(),
        pa       = pa,
        coords   = legacy_coords(pa$WDPA_PID),
        settings = legacy_settings(pa)
      )
    }
    cache
  }
})

test_that("legacy inputs are consistent with the old outputs", {
  skip_if_no_legacy()
  L <- legacy()

  # Same 4,025 PAs in the annual data and the PA table
  expect_equal(nrow(L$pa), 4025)
  expect_setequal(unique(L$annual$WDPA_PID), L$pa$WDPA_PID)

  # PA means of the annual series reproduce the old PA table (01:88-99)
  i <- match(L$pa$WDPA_PID, names(tapply(L$annual$edge_extent, L$annual$WDPA_PID, mean)))
  m_ext <- tapply(L$annual$edge_extent, L$annual$WDPA_PID, mean, na.rm = TRUE)[i]
  m_int <- tapply(L$annual$edge_intensity, L$annual$WDPA_PID, mean, na.rm = TRUE)[i]
  n_yrs <- tapply(!is.na(L$annual$edge_extent), L$annual$WDPA_PID, sum)[i]
  expect_equal(as.vector(m_ext), L$pa$edge_extent_mean, tolerance = 1e-12)
  expect_equal(as.vector(m_int), L$pa$edge_intensity_mean, tolerance = 1e-12)
  expect_equal(as.vector(n_yrs), as.numeric(L$pa$n_years))

  # Old trend labels all map to the new codes
  expect_true(all(stats::na.omit(L$pa$trend_extent) %in% CFG$trend$levels))
  expect_true(all(stats::na.omit(L$pa$trend_intensity_zyp) %in% CFG$trend$levels))

  # The old model set: n = 3,914 with the old biome and IUCN counts (survey of 03/02b)
  ms <- legacy_model_set(L$pa, L$coords)
  expect_equal(nrow(ms), 3914)
  expect_equal(as.vector(table(factor(ms$BIOME_NAME, levels = L$settings$biome_order))),
               c(1292, 1135, 669, 368, 351, 99))
  expect_equal(as.vector(table(factor(ms$IUCN_group, levels = c("strict", "non-strict", "unknown")))),
               c(1056, 1561, 1297))

  # Response SDs on the model set reproduce response_sd_gls.csv (written by 03)
  expect_equal(unname(L$settings$sd_resp),
               c(stats::sd(ms$edge_extent_mean), stats::sd(ms$edge_intensity_mean)),
               tolerance = 1e-12)
})

# ── Step 2: per-PA trends (02:63-128, 437-470) ─────────────────────────────────
# New: pa_trends() on the old annual series. Old: pa_level_clean.csv (all trend columns)
# and pa_trends.csv (written by the same run of 02).
legacy_new_trends <- local({
  cache <- NULL
  function() {
    if (is.null(cache)) {
      L <- legacy(); s <- L$settings
      cache <<- lapply(c(extent = "edge_extent", intensity = "edge_intensity"), function(resp)
        pa_trends(L$annual, resp, sub("edge_", "", resp), s$min_years, s$fdr, CFG$trend$levels))
    }
    cache
  }
})

test_that("per-PA OLS trends, BH classes and Yue-Pilon reproduce the old run", {
  skip_if_no_legacy()
  L <- legacy()
  old_tr <- as.data.frame(readr::read_csv(legacy_file("data", "processed", "pa_trends.csv"),
                                          col_types = readr::cols(WDPA_PID = "c"),
                                          show_col_types = FALSE, progress = FALSE))
  for (m in c("extent", "intensity")) {
    new <- legacy_new_trends()[[m]]
    # Same PA set per metric as the old run
    has_old <- !is.na(L$pa[[paste0("slope_", m)]])
    expect_setequal(new$WDPA_PID, L$pa$WDPA_PID[has_old])
    i <- match(L$pa$WDPA_PID, new$WDPA_PID)
    for (col in paste0(c("slope_", "se_", "stat_", "p_value_", "p_fdr_", "sen_", "p_zyp_",
                         "p_fdr_zyp_"), m)) {
      parity_check("pa_level_clean.csv", col, new[[col]][i], L$pa[[col]], tol = 1e-8)
    }
    for (col in c(paste0("trend_", m), paste0("trend_", m, "_zyp"))) {
      parity_check("pa_level_clean.csv", col, new[[col]][i], L$pa[[col]], kind = "identical")
    }
    j <- match(old_tr$WDPA_PID, new$WDPA_PID)
    for (col in paste0(c("slope_", "se_", "p_value_", "p_fdr_", "sen_", "p_fdr_zyp_"), m)) {
      parity_check("pa_trends.csv", col, new[[col]][j], old_tr[[col]], tol = 1e-8)
    }
    parity_check("pa_trends.csv", paste0("trend_", m), new[[paste0("trend_", m)]][j],
                 legacy_map_classes(old_tr[[paste0("trend_", m)]]), kind = "identical")
  }
})

# ── Step 3: standardised change, bootstraps, biome GLS, correlation, % change ──
# (02:135-150, 185-435, 494-534; 10:35-79; 11:31-52.) The new functions run on the PA table
# with the step 2 trends (new), so this also checks that the chain reproduces the old tables.
# Metric labels and the pooled label are the old ones, so rows match by (metric, group).
LEGACY_METRICS <- c(extent = "Edge extent", intensity = "Edge intensity")
LEGACY_POOLED  <- "All biomes"

legacy_step3 <- local({
  cache <- NULL
  function() {
    if (is.null(cache)) {
      L <- legacy(); s <- L$settings; tr <- legacy_new_trends()
      pa <- L$pa[, c("WDPA_PID", "BIOME_NAME", "area_km2", "edge_extent_mean", "edge_intensity_mean")]
      k <- c()
      for (m in names(LEGACY_METRICS)) {
        pa <- cbind(pa, tr[[m]][match(pa$WDPA_PID, tr[[m]]$WDPA_PID), -1])
        ch <- standardised_change(pa[[paste0("slope_", m)]], pa[[paste0("se_", m)]],
                                  s$n_span, s$sd_resp[[m]])
        pa[[paste0("change_", m, "_z")]]    <- ch$change
        pa[[paste0("se_change_", m, "_z")]] <- ch$se_change
        k[[LEGACY_METRICS[[m]]]] <- ch$k
      }
      cols <- function(prefix) stats::setNames(paste0(prefix, names(LEGACY_METRICS)), LEGACY_METRICS)
      run <- function(cl, seed, ...) boot_means(pa, cl, "BIOME_NAME", s$biome_order, s$boot_R,
                                               seed, pooled_label = LEGACY_POOLED, ...)
      cache <<- list(
        pa = pa, k = unlist(k), cols = cols,
        biome    = run(cols("slope_"), s$boot_seeds[["biome"]]),
        pooled   = run(cols("slope_"), s$boot_seeds[["pooled"]], pooled = "only"),
        weighted = run(cols("slope_"), s$boot_seeds[["weighted"]], pooled = "append",
                       weights = "area_km2"),
        sen      = run(cols("sen_"), s$boot_seeds[["sen"]], pooled = "append")
      )
    }
    cache
  }
})

test_that("standardised change reproduces the old change_*_z", {
  skip_if_no_legacy()
  L <- legacy(); S <- legacy_step3()
  for (m in names(LEGACY_METRICS)) for (col in paste0(c("change_", "se_change_"), m, "_z")) {
    parity_check("pa_level_clean.csv", col, S$pa[[col]], L$pa[[col]], tol = 1e-8)
  }
})

test_that("bootstrap means reproduce the old biome, pooled, area-weighted and Sen summaries", {
  skip_if_no_legacy()
  S <- legacy_step3()
  z <- function(x, metric) x * S$k[metric]

  # Seed 42 (biomes) and 45 (pooled): same seed and order as the old run → identical
  for (b in list(list("biome", "biome_slope_summary.csv", "biome_boot_replicates.csv"),
                 list("pooled", "pooled_slope_summary.csv", "pooled_boot_replicates.csv"))) {
    new <- S[[b[[1]]]]; old <- legacy_table(b[[2]])
    n <- match_rows(new$summary, old, c("metric", "BIOME_NAME"))
    parity_check(b[[2]], "n_pa", n$n, old$n_pa, kind = "identical")
    for (col in c("mean", "ci_lo", "ci_hi")) {
      oc <- c(mean = "mean_slope", ci_lo = "ci_lo", ci_hi = "ci_hi")[[col]]
      parity_check(b[[2]], oc, n[[col]], old[[oc]], tol = 1e-8)
      oz <- c(mean = "mean_change_z", ci_lo = "ci_lo_z", ci_hi = "ci_hi_z")[[col]]
      parity_check(b[[2]], oz, z(n[[col]], n$metric), old[[oz]], tol = 1e-8)
    }
    oreps <- legacy_table(b[[3]])
    nr <- match_rows(new$reps, oreps, c("metric", "BIOME_NAME", "rep"))
    parity_check(b[[3]], "mean_slope", nr$mean, oreps$mean_slope, tol = 1e-8)
    parity_check(b[[3]], "change_z", z(nr$mean, nr$metric), oreps$change_z, tol = 1e-8)
  }

  # Seed 44 (Sen, biomes): same seed and order → identical
  old <- legacy_table("biome_sen_slope_summary.csv")
  n <- match_rows(S$sen$summary, old, c("metric", "BIOME_NAME"))
  parity_check("biome_sen_slope_summary.csv", "n_pa", n$n, old$n_pa, kind = "identical")
  for (col in c("mean", "ci_lo", "ci_hi")) {
    oc <- c(mean = "mean_slope", ci_lo = "ci_lo", ci_hi = "ci_hi")[[col]]
    parity_check("biome_sen_slope_summary.csv", oc, n[[col]], old[[oc]], tol = 1e-8)
  }

  # Seed 43 (area-weighted, biomes): the old block also ran a duplicate unweighted bootstrap,
  # removed here (A4) → point estimates identical, CIs within 5% of the CI width
  old <- legacy_table("biome_slope_weighted_comparison.csv")
  n <- match_rows(S$weighted$summary, old, c("metric", "BIOME_NAME"))
  u <- match_rows(S$biome$summary, old, c("metric", "BIOME_NAME"))
  f <- "biome_slope_weighted_comparison.csv"
  parity_check(f, "n_pa", n$n, old$n_pa, kind = "identical")
  parity_check(f, "mean_slope", u$mean, old$mean_slope, tol = 1e-8)
  parity_check(f, "mean_slope_weighted", n$mean, old$mean_slope_weighted, tol = 1e-8)
  parity_check(f, "mean_slope_weighted_z", z(n$mean, n$metric), old$mean_slope_weighted_z, tol = 1e-8)
  w_w <- old$ci_hi_weighted - old$ci_lo_weighted
  w_u <- old$ci_hi_unweighted - old$ci_lo_unweighted
  parity_check(f, "ci_lo_weighted (/ CI width)", n$ci_lo / w_w, old$ci_lo_weighted / w_w, 0.05, "abs")
  parity_check(f, "ci_hi_weighted (/ CI width)", n$ci_hi / w_w, old$ci_hi_weighted / w_w, 0.05, "abs")
  # The old duplicate unweighted CIs (seed 43) vs the seed-42 CIs: Monte Carlo error only
  parity_check(f, "ci_lo_unweighted (/ CI width)", u$ci_lo / w_u, old$ci_lo_unweighted / w_u, 0.05, "abs")
  parity_check(f, "ci_hi_unweighted (/ CI width)", u$ci_hi / w_u, old$ci_hi_unweighted / w_u, 0.05, "abs")

  # Pooled area-weighted and Sen rows: only in Table SV (10:57-79), rounded to 3 dp, from a
  # stream with a duplicate bootstrap (removed) → points within rounding, CIs within 5% of the
  # CI width plus rounding
  sv <- legacy_table("TableSV_biome_trend_summary.csv")
  sv <- sv[sv$BIOME_NAME == LEGACY_POOLED, ]
  half <- 0.5e-3 + 1e-12
  for (b in list(list("weighted", "change_area_weighted", "aw_lo", "aw_hi"),
                 list("sen", "sen_change_mean", "sen_lo", "sen_hi"))) {
    p <- S[[b[[1]]]]$summary
    p <- match_rows(p[p$BIOME_NAME == LEGACY_POOLED, ], sv, c("metric", "BIOME_NAME"))
    f <- "TableSV_biome_trend_summary.csv (All biomes)"
    parity_check(f, b[[2]], z(p$mean, p$metric), sv[[b[[2]]]], tol = half, kind = "abs")
    w <- sv[[b[[4]]]] - sv[[b[[3]]]]
    for (col in c(3, 4)) {
      parity_check(f, paste0(b[[col]], " (/ CI width)"), z(p[[c("ci_lo", "ci_hi")[col - 2]]], p$metric) / w,
                   sv[[b[[col]]]] / w, tol = 0.05 + half / min(w), kind = "abs")
    }
  }
})

test_that("class counts and Yue-Pilon agreement reproduce the old tables", {
  skip_if_no_legacy()
  S <- legacy_step3(); s <- legacy()$settings
  cls <- c("Edge extent" = "trend_extent", "Edge intensity" = "trend_intensity")
  new <- class_counts(S$pa, cls, "BIOME_NAME", s$biome_order, CFG$trend$levels, LEGACY_POOLED)
  old <- legacy_table("trend_class_counts.csv")
  old$class <- legacy_map_classes(old$class)
  n <- match_rows(new, old, c("metric", "BIOME_NAME", "class"))
  parity_check("trend_class_counts.csv", "n", n$n, old$n, kind = "identical")
  parity_check("trend_class_counts.csv", "pct", n$pct, old$pct, tol = 1e-12)
  # Classes absent from the old table (count() drops them) have n = 0 here
  key <- function(x) paste(x$metric, x$BIOME_NAME, x$class)
  expect_true(all(new$n[!key(new) %in% key(old)] == 0))

  pairs <- list("Edge extent" = c("trend_extent", "trend_extent_zyp"),
                "Edge intensity" = c("trend_intensity", "trend_intensity_zyp"))
  ag <- trend_agreement(S$pa, pairs, "BIOME_NAME", s$biome_order, CFG$trend$levels, LEGACY_POOLED)
  old <- legacy_table("trend_yuepilon_agreement.csv")
  n <- match_rows(ag$agreement, old, c("metric", "BIOME_NAME"))
  f <- "trend_yuepilon_agreement.csv"
  parity_check(f, "n", n$n, old$n, kind = "identical")
  parity_check(f, "n_concordant", n$n_concordant, old$n_concordant, kind = "identical")
  for (col in c("pct_concordant", "pct_increase_ols", "pct_decrease_ols", "pct_increase_zyp",
                "pct_decrease_zyp")) {
    oc <- sub("increase", "increasing", sub("decrease", "decreasing", col))
    parity_check(f, oc, n[[col]], old[[oc]], tol = 1e-12)
  }
  old <- legacy_table("trend_yuepilon_crosstab.csv")
  old$ols <- legacy_map_classes(old$ols); old$zyp <- legacy_map_classes(old$zyp)
  n <- match_rows(ag$crosstab, old, c("metric", "ols", "zyp"))
  parity_check("trend_yuepilon_crosstab.csv", "Freq", n$n, old$Freq, kind = "identical")
  expect_equal(nrow(ag$crosstab), nrow(old))
})

# GLS tolerance: the collaborator's own rerun on this platform differed from the saved tables by
# up to 2e-5 (REML optimiser, BLAS; Phase 1), so 1e-4 here. Group means are exact.
test_that("biome GLS reproduces the old omnibus, means, contrasts, letters and sigma", {
  skip_if_no_legacy()
  S <- legacy_step3(); s <- legacy()$settings
  tab <- function(n) { x <- legacy_table(n); x }
  fits <- lapply(c("Edge extent" = "change_extent_z", "Edge intensity" = "change_intensity_z"),
                 function(r) biome_gls(S$pa, r, "BIOME_NAME", s$biome_order))
  stack <- function(field) do.call(rbind, lapply(names(fits), function(m)
    cbind(metric = m, fits[[m]][[field]], stringsAsFactors = FALSE)))
  tol <- 1e-4

  old <- tab("biome_trend_gls_omnibus.csv"); n <- match_rows(stack("omnibus"), old, "metric")
  f <- "biome_trend_gls_omnibus.csv"
  parity_check(f, "numDF/denDF", c(n$numDF, n$denDF), c(old$numDF, old$denDF), kind = "identical")
  parity_check(f, "F.value", n$F_value, old$F.value, tol = tol)
  parity_check(f, "p.value", n$p_value, old$p.value, tol = tol)

  old <- tab("biome_trend_gls_means.csv"); n <- match_rows(stack("means"), old, c("metric", "BIOME_NAME"))
  f <- "biome_trend_gls_means.csv"
  parity_check(f, "emmean", n$emmean, old$emmean, tol = 1e-8)
  parity_check(f, "df", n$df, old$df, kind = "identical")
  for (col in c("SE", "lower.CL", "upper.CL", "t.ratio", "p.value")) parity_check(f, col, n[[col]], old[[col]], tol = tol)

  old <- tab("biome_trend_gls_pairwise.csv"); n <- match_rows(stack("pairs"), old, c("metric", "contrast"))
  f <- "biome_trend_gls_pairwise.csv"
  parity_check(f, "estimate", n$estimate, old$estimate, tol = 1e-8)
  parity_check(f, "df", n$df, old$df, kind = "identical")
  for (col in c("SE", "lower.CL", "upper.CL", "t.ratio", "p.value")) parity_check(f, col, n[[col]], old[[col]], tol = tol)

  old <- tab("biome_trend_gls_letters.csv"); n <- match_rows(stack("letters"), old, c("metric", "BIOME_NAME"))
  parity_check("biome_trend_gls_letters.csv", "letters", n$letters, old$letters, kind = "identical")

  old <- tab("biome_trend_gls_sigma.csv"); n <- match_rows(stack("sigma"), old, c("metric", "BIOME_NAME"))
  parity_check("biome_trend_gls_sigma.csv", "resid_sd", n$resid_sd, old$resid_sd, tol = tol)
})

test_that("slope correlation and % change reproduce the old tables", {
  skip_if_no_legacy()
  S <- legacy_step3(); s <- legacy()$settings
  # Table SV correlation: only saved rounded to 4 dp (10:138). Run on the OLD slopes: edge extent
  # is a ratio of transect counts, so some PAs have mathematically equal slopes (ties, incl. seven
  # exact zeros computed as ±1e-17 by lm and ±1e-19 by the sums). The two computations split these
  # ties differently in the last bits, so Spearman ranks them differently (Boreal rho moves by
  # 3e-5, just over the rounding). The new-slope result is checked below after rounding both
  # slope sets to 9 significant digits (and |slope| < 1e-15 to 0), which restores the ties.
  L <- legacy()
  new <- slope_correlation(L$pa, "slope_extent", "slope_intensity", "BIOME_NAME", s$biome_order,
                           LEGACY_POOLED)
  old <- legacy_table("TableSV_slope_correlation.csv")
  n <- match_rows(new, old, "stratum")
  f <- "TableSV_slope_correlation.csv"
  parity_check(f, "n", n$n, old$n, kind = "identical")
  for (col in c("r_pearson", "r_lo", "r_hi", "p_pearson", "rho_spearman", "p_spearman")) {
    parity_check(f, col, n[[col]], old[[col]], tol = 0.5e-4 + 1e-12, kind = "abs")
  }
  zero_noise <- function(x) signif(ifelse(abs(x) < 1e-15, 0, x), 9)
  a <- data.frame(BIOME_NAME = L$pa$BIOME_NAME, x = zero_noise(S$pa$slope_extent),
                  y = zero_noise(S$pa$slope_intensity))
  b <- data.frame(BIOME_NAME = L$pa$BIOME_NAME, x = zero_noise(L$pa$slope_extent),
                  y = zero_noise(L$pa$slope_intensity))
  ca <- slope_correlation(a, "x", "y", "BIOME_NAME", s$biome_order, LEGACY_POOLED)
  cb <- slope_correlation(b, "x", "y", "BIOME_NAME", s$biome_order, LEGACY_POOLED)
  for (col in c("r_pearson", "r_lo", "r_hi", "rho_spearman")) {
    parity_check("slope correlation, new vs old slopes (9 significant digits)", col,
                 ca[[col]], cb[[col]], tol = 1e-8)
  }

  # % change (11:31-52): biome rows from the seed-42 summaries, pooled from seed 45
  summ <- rbind(S$biome$summary, S$pooled$summary)
  new <- percent_change(summ, S$pa, S$cols("slope_"),
                        stats::setNames(c("edge_extent_mean", "edge_intensity_mean"), LEGACY_METRICS),
                        "BIOME_NAME", s$n_span, LEGACY_POOLED)
  old <- legacy_table("trend_percent_change_biome.csv")
  n <- match_rows(new, old, c("metric", "BIOME_NAME"))
  f <- "trend_percent_change_biome.csv"
  parity_check(f, "n_pa", n$n, old$n_pa, kind = "identical")
  for (col in c("baseline", "pct_change", "pct_lo", "pct_hi")) parity_check(f, col, n[[col]], old[[col]], tol = 1e-8)
})

# ── Step 4: spatial set-up (HaversineLMEfunctions.R; 03:126-181) ───────────────
# The full check (all 3,914 PAs, 7,657,741 pairs: bit-identical; 0.6 s vs 27 s) was run once on
# 2026-11-08; here the first 800 PAs of the old model set, against the original file's loop.
test_that("vectorised distances on the old model set equal the original function", {
  skip_if_no_legacy()
  L <- legacy(); s <- L$settings
  ms <- legacy_model_set(L$pa, L$coords)
  j <- jitter_duplicates(ms, s$jitter_deg, s$jitter_seed, s$jitter_tries)
  parity_check("model set coordinates", "PAs jittered (none coincide)", attr(j, "n_jittered"), 0,
               kind = "identical")
  env <- new.env(parent = globalenv())
  sys.source(legacy_file("scripts", "HaversineLMEfunctions.R"), envir = env)
  xy <- cbind(j$lon, j$lat)[1:800, ]
  parity_check("HaversineLMEfunctions.R", "haversineDist, first 800 PAs (bitwise)",
               as.vector(haversineDist(xy)), as.vector(env$haversineDist(xy)), kind = "identical")
})

# ── Step 5: mean models (03) at the saved range ────────────────────────────────
# The full spatial models are fitted with the range FIXED at the old estimate (corHaversine
# fixed = TRUE), plus the non-spatial full models. This reproduces every table of the full
# models, emmeans df included (emmeans counts the fixed range as a covariance parameter). The
# additive models (range not saved), the range CI and AICc_spatial are checked by the optimised
# refits (tests/r/parity/run_parity_full.R). Fits are cached in data/parity/cache (≈ 1 GB);
# the first run fits them (≈ +3 min). Normalised residuals of a spatial fit take ≈ 35 s each,
# so this block adds ≈ 2 min even with the cache.
source(testthat::test_path("parity", "mean_model_checks.R"), local = TRUE)

legacy_mean_fixed <- local({
  cache <- NULL
  function() {
    if (is.null(cache)) {
      L <- legacy(); s <- L$settings
      d <- legacy_model_data(L$pa, L$coords, s)
      rng <- legacy_ranges()
      cdir <- file.path(CFG$paths$parity_dir, "cache")
      w <- suppressWarnings(moran_weights(d, CFG$spatial$k_nn, CFG$spatial$bands_km))  # sparse short bands
      cache <<- lapply(stats::setNames(names(LEGACY_RESPONSES), names(LEGACY_RESPONSES)), function(r) {
        resp <- LEGACY_RESPONSES[[r]]
        cs <- spatial_setup(d, value = rng[[r]], fixed = TRUE, cache_dir = cdir)
        fits <- list(
          full = fit_gls_cached(model_formula(resp, "full"), d, cs, cache_dir = cdir),
          full_nosp = fit_gls_cached(model_formula(resp, "full"), d, NULL, cache_dir = cdir))
        fits$add <- fits$full_nosp   # placeholder: additive tables are not compared here
        mean_model_tables(fits, d, resp, r, s$hmi[["mean"]], s$hmi[["sd"]], moran = w,
                          moran_settings = list(nsim = CFG$spatial$moran_nsim, seed = CFG$spatial$moran_seed),
                          pred_by = c(if (r == "extent") "IUCN_group", "BIOME_NAME"))
      })
    }
    cache
  }
})

test_that("mean models at the saved range reproduce the old full-model tables", {
  skip_if_no_legacy()
  check_mean_tables(legacy_mean_fixed(), what = "full", tol = 1e-6, source = "fast")
})

# ── Step 6: change models (02b), model A at its saved range ────────────────────
# Responses are the standardised changes from step 3 (new trends). Model A is fitted at its saved
# range (fixed) with its non-spatial analogue; this also fixes σ_A and so τ² and the weights.
# Model B (range not saved), the ranges, CIs and AICc spatial: run_parity_full.R.
source(testthat::test_path("parity", "change_model_checks.R"), local = TRUE)

legacy_change_data <- function() {
  L <- legacy(); S <- legacy_step3()
  d <- legacy_model_data(L$pa, L$coords, L$settings)
  i <- match(d$WDPA_PID, S$pa$WDPA_PID)
  for (col in c("change_extent_z", "se_change_extent_z", "change_intensity_z", "se_change_intensity_z")) {
    d[[col]] <- S$pa[[col]][i]
  }
  stopifnot(!anyNA(d$change_extent_z), !anyNA(d$change_intensity_z))
  d
}

test_that("change model A at its saved range reproduces the old 02b tables", {
  skip_if_no_legacy()
  s <- legacy()$settings
  d <- legacy_change_data()
  rng <- legacy_change_ranges()
  cdir <- file.path(CFG$paths$parity_dir, "cache")
  w <- suppressWarnings(moran_weights(d, CFG$spatial$k_nn, CFG$spatial$bands_km[1:2]))
  tabs <- lapply(stats::setNames(names(LEGACY_CHANGE), names(LEGACY_CHANGE)), function(r) {
    cs <- spatial_setup(d, value = rng[[r]], fixed = TRUE, cache_dir = cdir)
    cm <- fit_change_models(d, LEGACY_CHANGE[[r]][["response"]], LEGACY_CHANGE[[r]][["se"]], cs,
                            cache_dir = cdir, label = r, fit_B = FALSE)
    change_model_tables(cm, LEGACY_CHANGE[[r]][["response"]], r, s$ref_contrast, moran = w,
                        moran_settings = list(nsim = CFG$spatial$moran_nsim, seed = CFG$spatial$moran_seed),
                        models = "A")
  })
  check_change_tables(tabs, models = "A", optimised = FALSE, tol = 1e-6, source = "fast")
})

# ── Phase 3, D1: the change models fitted with their own reference levels ──────
# The old run fitted the change models with strict / Grassland as the reference and reported
# emmeans contrasts against non-strict / Boreal. The new 08 fits them with non-strict / Boreal
# as the reference (config ref), the same model reparametrised. Model A refitted that way
# at its saved range: the categorical coefficients equal the old contrasts, and the continuous
# coefficients the old ones. (The coefficient CIs use df = N − p, the emmeans contrasts N − p − 1,
# as the fixed range counts as a covariance parameter, so the CIs and p are recomputed at the
# old df: the df is then the only difference.)
test_that("change model A with the change reference levels reproduces the old contrasts", {
  skip_if_no_legacy()
  expect_identical(CFG$ref, legacy()$settings$ref_contrast)
  d <- set_model_references(legacy_change_data(), CFG$ref, names(CFG$iucn_groups),
                            CFG$biome_order)
  rng <- legacy_change_ranges()
  cdir <- file.path(CFG$paths$parity_dir, "cache")
  model_a <- function(x) {
    x <- x[x$model == CHANGE_MODEL_LABELS[["A"]], ]
    for (v in intersect(c("factor", "term"), names(x))) x[[v]] <- legacy_rename_terms(x[[v]])
    x
  }
  oc <- model_a(legacy_table("trend_drivers_contrasts_trend_gls.csv"))
  ob <- model_a(legacy_table("trend_drivers_coef_trend_gls.csv"))
  f_c <- "trend_drivers_contrasts_trend_gls.csv (A refitted, config ref)"
  f_b <- "trend_drivers_coef_trend_gls.csv (A refitted, config ref)"
  cols <- c(estimate = "estimate", std.error = "SE", statistic = "t.ratio", p.value = "p.value",
            conf.low = "lower.CL", conf.high = "upper.CL")
  for (r in names(LEGACY_CHANGE)) {
    cs <- spatial_setup(d, value = rng[[r]], fixed = TRUE, cache_dir = cdir)
    ct <- coef_table(fit_gls_cached(model_formula(LEGACY_CHANGE[[r]][["response"]], "add"), d, cs,
                                    cache_dir = cdir), r)
    o <- oc[oc$response == r, ]
    n <- ct[match(paste0(o$factor, o$level), ct$term), ]
    parity_check(f_c, paste(r, "terms"), sum(!is.na(n$term)), nrow(o), kind = "identical")
    parity_check(f_c, paste(r, "df (coef N - p = contrast df + 1)"), n$df - 1, o$df, kind = "identical")
    q <- stats::qt(0.975, o$df)
    n$conf.low <- n$estimate - q * n$std.error; n$conf.high <- n$estimate + q * n$std.error
    n$p.value <- 2 * stats::pt(-abs(n$statistic), o$df)
    for (k in names(cols)) parity_check(f_c, paste(r, cols[[k]]), n[[k]], o[[cols[[k]]]], tol = 1e-6)
    o <- ob[ob$response == r & ob$term %in% c(MODEL_COVARIATES, MODEL_HMI), ]
    n <- ct[match(o$term, ct$term), ]
    for (k in names(cols)) parity_check(f_b, paste(r, "continuous", k), n[[k]], o[[k]], tol = 1e-6)
  }
})
