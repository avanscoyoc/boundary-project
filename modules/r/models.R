# Spatial GLS models of PA means (Stats ¶2–3) and their tables; the change models (¶6) reuse the
# formulas and table builders.
#
# Ported from PA_New_Analysis/scripts/03_crosssectional_model.R (183-196, 219-244, 316-412,
# 414-476). Not ported (not in the methods; plan "Not in the methods"): the "Grassland + Tropical
# vs others" planned contrast (03:366-372), slopes per +0.1 HMI (03:352-362), and the residual
# diagnostic plots (moved to 10_figures.R).
#
# Functions take the model data and explicit settings and pass `data =` to emmeans (B7).
# Model data: one row per PA, in the row order of the correlation structure, with factors
# IUCN_group and BIOME_NAME whose FIRST level is the reference (set_reference()).

MODEL_COVARIATES <- c("STATUS_YR_z", "log_area_z", "elevation_mean_z", "terrain_slope_z",
                      "water_extent_pct_z")
MODEL_HMI <- "gHM_mean_z"

#' Model formula (03:191-196): controls + HMI + IUCN group + biome ("add"), plus the HMI × IUCN
#' and HMI × biome interactions ("full").
model_formula <- function(response, type = c("add", "full")) {
  type <- match.arg(type)
  rhs <- paste(c(MODEL_COVARIATES, MODEL_HMI, "IUCN_group", "BIOME_NAME"), collapse = " + ")
  if (type == "full") rhs <- paste(rhs, "+ gHM_mean_z:IUCN_group + gHM_mean_z:BIOME_NAME")
  stats::as.formula(paste(response, "~", rhs))
}

#' Factor with `ref` as the first (reference) level, the others in `order` (D1, B2).
#' Levels absent from x are dropped.
set_reference <- function(x, ref, order) {
  stopifnot(ref %in% order, all(stats::na.omit(as.character(x)) %in% order))
  droplevels(factor(as.character(x), levels = c(ref, setdiff(order, ref))))
}

#' Model factors of one model family (D1): IUCN_group and BIOME_NAME with the family's
#' reference levels first. `ref` = list(iucn, biome) (config ref$mean or ref$change);
#' `iucn_levels`, `biome_order` = the other levels' order. A different reference refits the same
#' model reparametrised: the categorical coefficients are then the contrasts against `ref`.
set_model_references <- function(d, ref, iucn_levels, biome_order) {
  d$IUCN_group <- set_reference(d$IUCN_group, ref$iucn, iucn_levels)
  d$BIOME_NAME <- set_reference(d$BIOME_NAME, ref$biome, biome_order)
  d
}

#' Fixed-effect table (03:227-235): estimate, SE, t, p and 95% CI (t, df = N − p).
coef_table <- function(m, label) {
  tt <- summary(m)$tTable
  ci <- nlme::intervals(m, which = "coef")$coef
  data.frame(response = label, term = rownames(tt), estimate = tt[, 1], std.error = tt[, 2],
             df = m$dims$N - m$dims$p, statistic = tt[, 3], p.value = tt[, 4],
             conf.low = ci[, 1], conf.high = ci[, 3], stringsAsFactors = FALSE, row.names = NULL)
}

#' Marginal Wald F tests of every term (03:238-244), denDF = N − p. The interaction rows of the
#' full model are hypotheses H2 (HMI × IUCN) and H3 (HMI × biome); `hypotheses = TRUE` keeps only
#' those rows and labels them.
wald_tests <- function(m, label, hypotheses = FALSE) {
  a <- stats::anova(m, type = "marginal")
  out <- data.frame(response = label, term = rownames(a), numDF = a$numDF,
                    denDF = m$dims$N - m$dims$p, F.value = a$`F-value`, p.value = a$`p-value`,
                    stringsAsFactors = FALSE, row.names = NULL)
  if (hypotheses) {
    out <- out[grepl(":", out$term), , drop = FALSE]
    out <- cbind(hypothesis = ifelse(grepl("IUCN", out$term), "H2: gHM x IUCN", "H3: gHM x BIOME"),
                 out, stringsAsFactors = FALSE)
    rownames(out) <- NULL
  }
  out
}

#' HMI slopes within IUCN groups and biomes from the full model (03:316-350, 364), H2/H3.
#'
#' emtrends with sample-proportional weights over the other factor, mode "df.error". Slopes:
#' Holm-adjusted p (Bonferroni CIs). Contrasts: IUCN pairwise (Holm); each biome vs the first
#' biome level (trt.vs.ctrl, Holm). n_PA per level; sd_gHM_z = SD of the HMI z-score within the
#' biome. Returns list(iucn, iucn_contrasts, biome, biome_contrasts); `tag` labels the fit
#' (e.g. "spatial").
hmi_slopes <- function(m, data, label, tag) {
  ei <- emmeans::emtrends(m, ~ IUCN_group, var = MODEL_HMI, data = data,
                          weights = "proportional", mode = "df.error")
  eb <- emmeans::emtrends(m, ~ BIOME_NAME, var = MODEL_HMI, data = data,
                          weights = "proportional", mode = "df.error")
  df <- function(x) {
    x <- as.data.frame(x)
    for (v in intersect(c("IUCN_group", "BIOME_NAME", "contrast"), names(x))) x[[v]] <- as.character(x[[v]])
    cbind(response = label, spatial = tag, x, stringsAsFactors = FALSE)
  }
  iucn <- df(summary(ei, infer = c(TRUE, TRUE), adjust = "holm"))
  iucn$n_PA <- as.integer(table(data$IUCN_group)[iucn$IUCN_group])
  biome <- df(summary(eb, infer = c(TRUE, TRUE), adjust = "holm"))
  biome$n_PA <- as.integer(table(data$BIOME_NAME)[biome$BIOME_NAME])
  biome$sd_gHM_z <- as.numeric(tapply(data[[MODEL_HMI]], data$BIOME_NAME, stats::sd)[biome$BIOME_NAME])
  list(
    iucn = iucn,
    iucn_contrasts = df(summary(emmeans::contrast(ei, method = "pairwise", adjust = "holm"),
                                infer = c(TRUE, TRUE))),
    biome = biome,
    biome_contrasts = df(summary(emmeans::contrast(eb, method = "trt.vs.ctrl", ref = 1,
                                                   adjust = "holm"), infer = c(TRUE, TRUE)))
  )
}

#' Contrasts of each level of `factor` against `ref` (03:376-389), from emmeans (mode
#' "df.error"): unadjusted p, plus p_holm over the contrasts of that factor. These are
#' coefficient differences of the same fit (no refit).
ref_contrasts <- function(m, data, factor, ref, label) {
  em <- emmeans::emmeans(m, stats::as.formula(paste("~", factor)), data = data, mode = "df.error")
  lv <- levels(data[[factor]])
  ctr <- emmeans::contrast(em, method = "trt.vs.ctrl", ref = which(lv == ref))
  tab <- as.data.frame(summary(ctr, infer = c(TRUE, TRUE), adjust = "none"))
  tab$contrast <- as.character(tab$contrast)
  tab$p_holm <- as.data.frame(summary(ctr, adjust = "holm"))$p.value
  cbind(response = label, factor = factor, reference = ref, level = setdiff(lv, ref), tab,
        stringsAsFactors = FALSE)
}

#' AICc of a gls fit (03:220-224). Compare only fits with the same fixed effects (REML).
aicc_gls <- function(m) {
  ll <- stats::logLik(m); k <- attr(ll, "df"); n <- m$dims$N
  -2 * as.numeric(ll) + 2 * k + 2 * k * (k + 1) / (n - k - 1)
}

#' Spatial parameters of a spatial fit and the AICc comparison with its non-spatial analogue
#' (03:399-411). The range CI is NA when it cannot be computed (e.g. a fixed range).
spatial_params <- function(m_sp, m_nosp, label) {
  cs <- m_sp$modelStruct$corStruct
  cpar <- stats::coef(cs, unconstrained = FALSE)
  ci <- tryCatch(nlme::intervals(m_sp, which = "var-cov")$corStruct, error = function(e) NULL)
  fixed <- isTRUE(attr(cs, "fixed"))
  range_km <- if (fixed) stats::coef(cs, unconstrained = FALSE, allCoef = TRUE)[["range"]] else cpar[["range"]]
  a_sp <- aicc_gls(m_sp); a_ns <- aicc_gls(m_nosp)
  data.frame(response = label, nugget = isTRUE(attr(cs, "nugget")), range_fixed = fixed,
             range_km = unname(range_km),
             range_lo = if (!is.null(ci)) ci["range", "lower"] else NA_real_,
             range_hi = if (!is.null(ci)) ci["range", "upper"] else NA_real_,
             sigma = m_sp$sigma, AICc_spatial = a_sp, AICc_nonspatial = a_ns,
             delta_AICc_nonspatial_minus_spatial = a_ns - a_sp, stringsAsFactors = FALSE)
}

#' Pseudo-R² (03:414-419): squared correlation of the fixed-effect fitted values with y.
pseudo_r2 <- function(m, y) stats::cor(stats::fitted(m), y)^2

#' Predicted response across HMI within each level of `by` (03:443-470), for Fig. 3B–D.
#'
#' HMI grid: `n_grid` raw values from the min to the max of `data$gHM_mean`, as z-scores with
#' `hmi_mean`/`hmi_sd` (the z-score's own mean and SD). Other covariates at their means; the
#' other factor weighted in proportion to sample size; mode "df.error"; pointwise 95% CIs.
#' Each level is kept only within its own `probs` quantiles of raw HMI (no extrapolation).
hmi_predictions <- function(m, data, by, label, hmi_mean, hmi_sd, n_grid = 101,
                            probs = c(0.01, 0.99)) {
  g_grid <- seq(min(data$gHM_mean), max(data$gHM_mean), length.out = n_grid)
  z_grid <- (g_grid - hmi_mean) / hmi_sd
  at <- stats::setNames(list(z_grid), MODEL_HMI)
  em <- emmeans::emmeans(m, stats::as.formula(paste("~", MODEL_HMI, "|", by)), at = at,
                         weights = "proportional", data = data, mode = "df.error")
  p <- as.data.frame(summary(em, infer = c(TRUE, FALSE)))
  p$gHM_mean <- p[[MODEL_HMI]] * hmi_sd + hmi_mean
  q <- tapply(data$gHM_mean, data[[by]], stats::quantile, probs = probs)
  lev <- as.character(p[[by]])
  lo <- vapply(q[lev], `[`, numeric(1), 1); hi <- vapply(q[lev], `[`, numeric(1), 2)
  p <- p[p$gHM_mean >= lo & p$gHM_mean <= hi, , drop = FALSE]
  out <- data.frame(response = label, grp = as.character(p[[by]]), gHM_mean_z = p[[MODEL_HMI]],
                    gHM_mean = p$gHM_mean, predicted = p$emmean, SE = p$SE, df = p$df,
                    conf.low = p$lower.CL, conf.high = p$upper.CL, stringsAsFactors = FALSE)
  names(out)[2] <- by
  rownames(out) <- NULL
  out
}

#' The three fits of one response (03:300-305): additive spatial, full spatial, full non-spatial.
#' `correlation` is the spatial_setup() structure for the rows of `data`.
fit_mean_models <- function(data, response, correlation, method = "REML", max_iter = 200L,
                            cache_dir = NULL, label = response) {
  list(
    add       = fit_gls_cached(model_formula(response, "add"), data, correlation, method = method,
                               max_iter = max_iter, cache_dir = cache_dir,
                               label = paste(label, "additive, spatial")),
    full      = fit_gls_cached(model_formula(response, "full"), data, correlation, method = method,
                               max_iter = max_iter, cache_dir = cache_dir,
                               label = paste(label, "full, spatial")),
    full_nosp = fit_gls_cached(model_formula(response, "full"), data, NULL, method = method,
                               max_iter = max_iter, cache_dir = cache_dir,
                               label = paste(label, "full, non-spatial"))
  )
}

#' All tables of one response's mean models (03:307-476). `fits` from fit_mean_models();
#' `contrast_ref` = list(iucn, biome) for ref_contrasts() (NULL = skip); `moran` = weights from
#' moran_weights() (NULL = skip) with `moran_settings` = list(nsim, seed);
#' `pred_by` = factors to predict across HMI for.
mean_model_tables <- function(fits, data, response, label, hmi_mean, hmi_sd,
                              contrast_ref = NULL, moran = NULL, moran_settings = NULL,
                              pred_by = c("IUCN_group", "BIOME_NAME")) {
  y <- data[[response]]
  s_sp <- hmi_slopes(fits$full, data, label, "spatial")
  out <- list(
    wald       = wald_tests(fits$full, label, hypotheses = TRUE),
    wald_all   = wald_tests(fits$full, label),
    coef_add   = coef_table(fits$add, label),
    coef_full  = coef_table(fits$full, label),
    slopes_iucn = s_sp$iucn, slopes_iucn_contrasts = s_sp$iucn_contrasts,
    slopes_biome = s_sp$biome, slopes_biome_contrasts = s_sp$biome_contrasts,
    spatial_params = spatial_params(fits$full, fits$full_nosp, label),
    fit_stats = data.frame(response = label, pseudoR2_additive = pseudo_r2(fits$add, y),
                           pseudoR2_full = pseudo_r2(fits$full, y),
                           pseudoR2_full_nonspatial = pseudo_r2(fits$full_nosp, y)),
    response_sd = data.frame(response = label, n = length(y), mean_response = mean(y),
                             sd_response = stats::sd(y))
  )
  if (!is.null(contrast_ref)) {
    out$contrasts_add <- rbind(ref_contrasts(fits$add, data, "IUCN_group", contrast_ref$iucn, label),
                               ref_contrasts(fits$add, data, "BIOME_NAME", contrast_ref$biome, label))
  }
  if (!is.null(moran)) {
    m1 <- moran_residuals(stats::resid(fits$full_nosp, type = "pearson"), moran,
                          moran_settings$nsim, moran_settings$seed, paste0(label, ": non-spatial"))
    m2 <- moran_residuals(stats::resid(fits$full, type = "normalized"), moran,
                          moran_settings$nsim, moran_settings$seed,
                          paste0(label, ": spatial (normalised)"))
    out$moran_knn <- rbind(m1$knn, m2$knn)
    out$moran_bands <- rbind(m1$bands, m2$bands)
  }
  for (by in pred_by) {
    out[[paste0("pred_", by)]] <- hmi_predictions(fits$full, data, by, label, hmi_mean, hmi_sd)
  }
  out
}

# ── Change models (Stats ¶6) ──────────────────────────────────────────────────
# Ported from PA_New_Analysis/scripts/02b_trend_drivers.R (147-300). Response: standardised
# change (response SD per record); terms as the additive mean model; no interactions.
#   A  primary: unweighted spatial GLS (REML);
#   non-spatial analogue of A (AICc, Moran's I);
#   B  sensitivity: spatial GLS with varFixed(~ v), v = se² + τ² (precision_weights()).

CHANGE_MODEL_LABELS <- c(A = "primary (unweighted)", B = "precision-weighted")

#' Precision weights for the change models (02b:246-250). τ² is the between-PA variance of the
#' trends by the method of moments from the unweighted fit, max(0, σ²_A − mean(se²));
#' v = se² + τ², floored at `floor_frac` × median(v) (guards zero-variance series).
#' Returns list(v, table) where table has sigma2_unweighted, mean_se2, tau2, share_sampling_var.
precision_weights <- function(sigma, se, floor_frac = 0.01, label = NA_character_) {
  se2 <- se^2
  tau2 <- max(0, sigma^2 - mean(se2))
  v <- se2 + tau2
  v <- pmax(v, floor_frac * stats::median(v))
  list(v = v, table = data.frame(response = label, sigma2_unweighted = sigma^2,
                                 mean_se2 = mean(se2), tau2 = tau2,
                                 share_sampling_var = mean(se2) / sigma^2,
                                 stringsAsFactors = FALSE))
}

#' Fits of one change response: A (spatial), A non-spatial, and B (spatial, precision-weighted).
#' `se_col` holds the SE of the response (se_change_*_z). B needs A's σ, so A is fitted first.
#' Returns list(fits = list(A, nosp, B), data = `data` with column `v_precision` (needed for
#' emmeans on B), tau = precision_weights() table). fit_B = FALSE skips B (fits$B is NULL).
fit_change_models <- function(data, response, se_col, correlation, method = "REML",
                              max_iter = 200L, cache_dir = NULL, label = response,
                              floor_frac = 0.01, fit_B = TRUE) {
  f <- model_formula(response, "add")
  A <- fit_gls_cached(f, data, correlation, method = method, max_iter = max_iter,
                      cache_dir = cache_dir, label = paste(label, "change A, spatial"))
  nosp <- fit_gls_cached(f, data, NULL, method = method, max_iter = max_iter,
                         cache_dir = cache_dir, label = paste(label, "change A, non-spatial"))
  pw <- precision_weights(A$sigma, data[[se_col]], floor_frac, label)
  data$v_precision <- pw$v
  B <- if (fit_B) fit_gls_cached(f, data, correlation, weights = nlme::varFixed(~ v_precision),
                                 method = method, max_iter = max_iter, cache_dir = cache_dir,
                                 label = paste(label, "change B, spatial, precision-weighted"))
  list(fits = list(A = A, nosp = nosp, B = B), data = data, tau = pw$table)
}

#' Tables of one change response (02b:255-300). `cm` from fit_change_models(); `contrast_ref` =
#' list(iucn, biome) for ref_contrasts(); `moran` = moran_weights() (NULL = skip) with
#' `moran_settings` = list(nsim, seed). `models` = which of A, B to tabulate.
change_model_tables <- function(cm, response, label, contrast_ref, moran = NULL,
                                moran_settings = NULL, models = c("A", "B")) {
  d <- cm$data; y <- d[[response]]
  with_model <- function(x, k) cbind(x[1], model = CHANGE_MODEL_LABELS[[k]], x[-1],
                                     stringsAsFactors = FALSE)
  per <- lapply(models, function(k) {
    m <- cm$fits[[k]]
    list(coef = with_model(coef_table(m, label), k),
         wald = with_model(wald_tests(m, label), k),
         contr = with_model(rbind(ref_contrasts(m, d, "IUCN_group", contrast_ref$iucn, label),
                                  ref_contrasts(m, d, "BIOME_NAME", contrast_ref$biome, label)), k))
  })
  stack <- function(field) { x <- do.call(rbind, lapply(per, `[[`, field)); rownames(x) <- NULL; x }
  out <- list(coef = stack("coef"), wald = stack("wald"), contrasts = stack("contr"), tau = cm$tau)
  A <- cm$fits$A; nosp <- cm$fits$nosp
  sp <- spatial_params(A, nosp, label)
  out$spatial_params <- cbind(sp[1:2], n = length(y), sp[-(1:2)],
                              pseudoR2_spatial = pseudo_r2(A, y),
                              pseudoR2_nonspatial = pseudo_r2(nosp, y))
  if (!is.null(moran)) {
    m1 <- moran_residuals(stats::resid(nosp, type = "pearson"), moran, moran_settings$nsim,
                          moran_settings$seed, paste0(label, ": non-spatial"), bands = FALSE)
    m2 <- moran_residuals(stats::resid(A, type = "normalized"), moran, moran_settings$nsim,
                          moran_settings$seed, paste0(label, ": spatial (normalised)"), bands = FALSE)
    out$moran <- rbind(m1$knn, m2$knn)
  }
  out
}

# Labels of the continuous predictors in Table SZ (02b:304-306)
CONTINUOUS_LABELS <- c(STATUS_YR_z = "Establishment year", log_area_z = "PA area (log)",
                       gHM_mean_z = "Human modification (HMI)", elevation_mean_z = "Elevation",
                       terrain_slope_z = "Slope", water_extent_pct_z = "Water extent")

#' Table SZ in long format (02b:304-318): continuous terms (per 1 SD) and the categorical
#' contrasts against the reference classes, for each response and model.
table_sz <- function(coef, contrasts) {
  cont <- coef[coef$term %in% names(CONTINUOUS_LABELS), , drop = FALSE]
  a <- data.frame(response = cont$response, model = cont$model,
                  group = "Continuous predictor (per 1 SD)",
                  term_label = unname(CONTINUOUS_LABELS[cont$term]), estimate = cont$estimate,
                  conf.low = cont$conf.low, conf.high = cont$conf.high, p.value = cont$p.value,
                  p_holm = NA_real_, stringsAsFactors = FALSE)
  b <- data.frame(response = contrasts$response, model = contrasts$model,
                  group = ifelse(contrasts$factor == "IUCN_group", "IUCN category", "Biome"),
                  term_label = paste0(contrasts$level, " vs ", contrasts$reference),
                  estimate = contrasts$estimate, conf.low = contrasts$lower.CL,
                  conf.high = contrasts$upper.CL, p.value = contrasts$p.value,
                  p_holm = contrasts$p_holm, stringsAsFactors = FALSE)
  rbind(a, b)
}
