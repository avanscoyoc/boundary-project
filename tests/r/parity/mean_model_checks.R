# Parity checks of the mean-model tables (legacy 03) against the old outputs/tables/, shared by
# the fast suite (fixed range: full models only) and run_parity_full.R (optimised: everything).
#
# `tabs` = named list (extent, intensity) of mean_model_tables() results. `what` selects the
# tables to compare; `tol` the relative tolerance. The emmeans df (N − p − number of covariance
# parameters, fixed ones included) is the same at a fixed range, so df is compared exactly.
check_mean_tables <- function(tabs, what, tol, source) {
  old_tab <- function(name) {
    x <- legacy_table(name)
    names(x) <- legacy_rename_terms(names(x))
    for (v in intersect(c("term", "contrast"), names(x))) x[[v]] <- legacy_rename_terms(x[[v]])
    if ("factor" %in% names(x)) x$factor <- legacy_rename_terms(x$factor)
    x
  }
  stack <- function(field) do.call(rbind, lapply(tabs, `[[`, field))
  # Prediction rows have no exact key (HMI is a float): number them within response × group.
  seq_key <- function(x, by) { x$row_in_group <- stats::ave(seq_len(nrow(x)), x$response, x[[by]], FUN = seq_along); x }
  cmp <- function(file, field, key, num, ident = character(), df_cols = character(), seq_by = NULL) {
    old <- old_tab(file)
    new <- stack(field)
    if (!is.null(seq_by)) {
      new <- seq_key(new[new$response %in% old$response, , drop = FALSE], seq_by)
      old <- seq_key(old, seq_by)
      parity_check(file, "rows (HMI range per group)", nrow(new), nrow(old), kind = "identical", source = source)
    }
    new <- match_rows(new, old, key)
    for (col in c(ident, df_cols)) parity_check(file, col, new[[col]], old[[col]], kind = "identical", source = source)
    for (col in num) parity_check(file, col, new[[col]], old[[col]], tol = tol, source = source)
  }
  est <- c("estimate", "std.error", "statistic", "p.value", "conf.low", "conf.high")
  emm <- c("SE", "lower.CL", "upper.CL", "t.ratio", "p.value")

  if ("full" %in% what) {
    cmp("model_coef_full_gls.csv", "coef_full", c("response", "term"), est, ident = "df")
    cmp("model_wald_full_all_terms_gls.csv", "wald_all", c("response", "term"),
        c("F.value", "p.value"), ident = c("numDF", "denDF"))
    cmp("model_tests_wald_gls.csv", "wald", c("response", "term"), c("F.value", "p.value"),
        ident = c("hypothesis", "numDF", "denDF"))
    cmp("hmi_slopes_iucn_gls.csv", "slopes_iucn", c("response", "IUCN_group"),
        c("gHM_mean_z.trend", emm), ident = "n_PA", df_cols = "df")
    cmp("hmi_slopes_iucn_contrasts_gls.csv", "slopes_iucn_contrasts", c("response", "contrast"),
        c("estimate", emm), df_cols = "df")
    cmp("hmi_slopes_biome_gls.csv", "slopes_biome", c("response", "BIOME_NAME"),
        c("gHM_mean_z.trend", emm, "sd_gHM_z"), ident = "n_PA", df_cols = "df")
    cmp("hmi_slopes_biome_contrasts_gls.csv", "slopes_biome_contrasts", c("response", "contrast"),
        c("estimate", emm), df_cols = "df")
    pred_cols <- c("gHM_mean_z", "gHM_mean", "predicted", "SE", "conf.low", "conf.high")
    cmp("pred_extent_hmi_iucn_gls.csv", "pred_IUCN_group", c("response", "IUCN_group", "row_in_group"),
        pred_cols, df_cols = "df", seq_by = "IUCN_group")
    for (r in c("extent", "intensity")) {
      cmp(sprintf("pred_%s_hmi_biome_gls.csv", r), "pred_BIOME_NAME",
          c("response", "BIOME_NAME", "row_in_group"), pred_cols, df_cols = "df", seq_by = "BIOME_NAME")
    }
    cmp("model_fit_stats_gls.csv", "fit_stats", "response", c("pseudoR2_full", "pseudoR2_full_nonspatial"))
    cmp("spatial_params_gls.csv", "spatial_params", "response", c("sigma", "AICc_nonspatial"))
    cmp("response_sd_gls.csv", "response_sd", "response", c("mean_response", "sd_response"), ident = "n")
    # Moran's I: kNN permutation test (seed 1) and the distance-band correlogram
    cmp("moran_residuals_knn_gls.csv", "moran_knn", "model", c("I", "expected_I", "p"))
    cmp("moran_residuals_bands_gls.csv", "moran_bands", c("model", "mid_km"), c("I", "p"),
        ident = "n_links")
  }
  if ("add" %in% what) {
    cmp("model_coef_additive_gls.csv", "coef_add", c("response", "term"), est, ident = "df")
    cmp("model_contrasts_additive_gls.csv", "contrasts_add", c("response", "factor", "level"),
        c("estimate", emm, "p_holm"), ident = c("reference", "contrast"), df_cols = "df")
    cmp("model_fit_stats_gls.csv", "fit_stats", "response", "pseudoR2_additive")
  }
  if ("optimised" %in% what) {
    cmp("spatial_params_gls.csv", "spatial_params", "response",
        c("range_km", "AICc_spatial", "delta_AICc_nonspatial_minus_spatial"))
    # Range CI (nlme apVar, finite-difference Hessian): within 0.1% of the CI width
    old <- old_tab("spatial_params_gls.csv")
    parity_check_ci_width("spatial_params_gls.csv", match_rows(stack("spatial_params"), old, "response"),
                          old, "range_lo", "range_hi", tol = 1e-3, source = source)
  }
}
