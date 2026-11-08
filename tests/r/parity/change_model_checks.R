# Parity checks of the change-model tables (02b) against the old outputs/tables/, shared by the
# fast suite (A at its saved range) and run_parity_full.R (A and B optimised).
# `tabs` = named list (extent, intensity) of change_model_tables() results; `models` = which of
# "A", "B" were tabulated; `optimised` adds the range, its CI (within 0.1% of the CI width,
# see parity_check_ci_width()) and AICc spatial.
check_change_tables <- function(tabs, models, optimised, tol, source) {
  old_tab <- function(name) {
    x <- legacy_table(name)
    names(x) <- legacy_rename_terms(names(x))
    for (v in intersect(c("term", "contrast", "factor"), names(x))) x[[v]] <- legacy_rename_terms(x[[v]])
    if ("model" %in% names(x)) x <- x[x$model %in% CHANGE_MODEL_LABELS[models], , drop = FALSE]
    x
  }
  stack <- function(field) do.call(rbind, lapply(tabs, `[[`, field))
  cmp <- function(file, field, key, num, ident = character()) {
    old <- old_tab(file)
    new <- match_rows(stack(field), old, key)
    for (col in ident) parity_check(file, col, new[[col]], old[[col]], kind = "identical", source = source)
    for (col in num) parity_check(file, col, new[[col]], old[[col]], tol = tol, source = source)
  }
  sfx <- paste0(" [", paste(models, collapse = "+"), "]")
  cmp("trend_drivers_coef_trend_gls.csv", "coef", c("response", "model", "term"),
      c("estimate", "std.error", "statistic", "p.value", "conf.low", "conf.high"), ident = "df")
  cmp("trend_drivers_wald_trend_gls.csv", "wald", c("response", "model", "term"),
      c("F.value", "p.value"), ident = c("numDF", "denDF"))
  cmp("trend_drivers_contrasts_trend_gls.csv", "contrasts", c("response", "model", "factor", "level"),
      c("estimate", "SE", "lower.CL", "upper.CL", "t.ratio", "p.value", "p_holm"),
      ident = c("reference", "contrast", "df"))
  cmp("trend_drivers_tau2_trend_gls.csv", "tau", "response",
      c("sigma2_unweighted", "mean_se2", "tau2", "share_sampling_var"))
  cmp("trend_drivers_spatial_params_trend_gls.csv", "spatial_params", "response",
      c("sigma", "AICc_nonspatial", "pseudoR2_spatial", "pseudoR2_nonspatial"), ident = "n")
  if (optimised) {
    f <- "trend_drivers_spatial_params_trend_gls.csv"
    cmp(f, "spatial_params", "response", c("range_km", "AICc_spatial", "delta_AICc_nonspatial_minus_spatial"))
    # Range CI (nlme apVar, finite-difference Hessian): within 0.1% of the CI width
    old <- old_tab(f)
    parity_check_ci_width(f, match_rows(stack("spatial_params"), old, "response"), old,
                          "range_lo", "range_hi", tol = 1e-3, source = source)
  }
  cmp("trend_drivers_moran_trend_gls.csv", "moran", "model", c("I", "expected_I", "p"))
  # Table SZ (long): built from the coefficient and contrast tables
  old <- old_tab("TableSZ_trend_drivers_long.csv")
  sz <- do.call(rbind, lapply(tabs, function(t) table_sz(t$coef, t$contrasts)))
  new <- match_rows(sz, old, c("response", "model", "group", "term_label"))
  parity_check("TableSZ_trend_drivers_long.csv", paste0("rows", sfx), nrow(sz), nrow(old),
               kind = "identical", source = source)
  for (col in c("estimate", "conf.low", "conf.high", "p.value", "p_holm")) {
    parity_check("TableSZ_trend_drivers_long.csv", col, new[[col]], old[[col]], tol = tol, source = source)
  }
}
