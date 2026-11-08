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
