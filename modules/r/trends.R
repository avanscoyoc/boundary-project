# Per-PA trends: OLS slope on year, Benjamini–Hochberg classes, and the Yue–Pilon
# (pre-whitened Sen / Mann–Kendall) robustness check.
#
# Input is long PA-year data: one row per PA and year, with `WDPA_PID`, `year` and the
# response column. One call handles one series (one metric, one transect version), so the
# sensitivity runs are a loop over the same functions.
#
# Follows the earlier R implementation of this analysis (not in this repository). The OLS is
# computed from per-PA sums instead of one lm() per PA; slope, SE, t and p are the same as lm().

# Group index in first-appearance order. rowsum() sorts character groups by the locale, which
# would make the row order depend on the environment.
.group_index <- function(id) {
  f <- factor(id, levels = unique(id))
  list(levels = levels(f), index = as.integer(f))
}

#' Per-PA OLS of `response` on year.
#'
#' Rows with a missing response are dropped. PAs with fewer than `min_years` remaining years
#' are left out. Returns one row per PA, in first-appearance order:
#' WDPA_PID, n_years, slope, se, stat (t), p_value (two-sided, df = n_years − 2).
ols_trends <- function(d, response, min_years) {
  stopifnot(all(c("WDPA_PID", "year", response) %in% names(d)), min_years >= 3)
  y <- d[[response]]
  ok <- !is.na(y)
  x <- as.numeric(d$year[ok]); y <- y[ok]
  g <- .group_index(d$WDPA_PID[ok])

  n <- tabulate(g$index, nbins = length(g$levels))
  keep_pa <- n >= min_years
  in_kept <- keep_pa[g$index]
  x <- x[in_kept]; y <- y[in_kept]
  g <- .group_index(g$levels[g$index[in_kept]])
  n <- tabulate(g$index, nbins = length(g$levels))

  # Two-pass centred sums (exact for year values ~2000)
  xbar <- as.vector(rowsum(x, g$index, reorder = TRUE)) / n
  ybar <- as.vector(rowsum(y, g$index, reorder = TRUE)) / n
  dx <- x - xbar[g$index]
  dy <- y - ybar[g$index]
  sxx <- as.vector(rowsum(dx * dx, g$index, reorder = TRUE))
  sxy <- as.vector(rowsum(dx * dy, g$index, reorder = TRUE))
  syy <- as.vector(rowsum(dy * dy, g$index, reorder = TRUE))

  slope <- sxy / sxx
  rss   <- pmax(syy - slope * sxy, 0)
  df    <- n - 2
  se    <- sqrt(rss / df / sxx)
  stat  <- slope / se
  data.frame(
    WDPA_PID = g$levels,
    n_years  = n,
    slope    = slope,
    se       = se,
    stat     = stat,
    p_value  = 2 * stats::pt(-abs(stat), df),
    stringsAsFactors = FALSE
  )
}

#' Benjamini–Hochberg adjustment across PAs and trend classes.
#'
#' `estimate` is the slope (OLS or Sen); `p` its p-value. Missing p-values are left out of the
#' adjustment (as p.adjust does). Class: `levels[1]` (increase) if estimate > 0 and p_fdr < fdr,
#' `levels[3]` (decrease) if estimate < 0 and p_fdr < fdr, otherwise `levels[2]` (no trend);
#' NA where the estimate is NA. Returns list(p_fdr, class).
bh_classes <- function(estimate, p, fdr, levels) {
  stopifnot(length(estimate) == length(p), length(levels) == 3)
  p_fdr <- stats::p.adjust(p, method = "BH")
  sig <- !is.na(p_fdr) & p_fdr < fdr
  cls <- ifelse(sig & estimate > 0, levels[1],
         ifelse(sig & estimate < 0, levels[3], levels[2]))
  cls[is.na(estimate)] <- NA_character_
  list(p_fdr = p_fdr, class = cls)
}

#' Lag-1 autocorrelation of each PA's OLS trend residuals: a check of the independence behind
#' the per-PA OLS p-values (Yue–Pilon is the robustness check).
#'
#' r1 = Σ e_t e_{t−1} / Σ e_t², the usual ACF estimator, with the sum over pairs of consecutive
#' years only (a missing year breaks a pair). `bound` = 1.96 / √n_years, the approximate 95%
#' limit for white noise. Same PAs as ols_trends() (≥ min_years); first-appearance order.
#' Returns WDPA_PID, n_years, n_pairs, r1, bound.
residual_autocorrelation <- function(d, response, min_years) {
  stopifnot(all(c("WDPA_PID", "year", response) %in% names(d)))
  ok <- !is.na(d[[response]])
  y <- d[[response]][ok]; x <- as.numeric(d$year[ok])
  g <- .group_index(d$WDPA_PID[ok])
  rows <- split(seq_along(y), factor(g$index, levels = seq_along(g$levels)))
  res <- vapply(rows, function(i) {
    n <- length(i)
    if (n < min_years) return(c(n, NA_real_, NA_real_))
    o <- i[order(x[i])]
    e <- stats::lm.fit(cbind(1, x[o]), y[o])$residuals
    pair <- diff(x[o]) == 1
    ss <- sum(e^2)
    c(n, sum(pair), if (ss > 0) sum(e[-1][pair] * e[-n][pair]) / ss else NA_real_)
  }, numeric(3))
  keep <- res[1, ] >= min_years
  data.frame(WDPA_PID = g$levels[keep], n_years = as.integer(res[1, keep]), n_pairs = as.integer(res[2, keep]),
             r1 = res[3, keep], bound = 1.96 / sqrt(res[1, keep]), stringsAsFactors = FALSE)
}

#' Summary of residual_autocorrelation() per metric, for all PAs and per group: n, median and
#' quartiles of r1, % of PAs with r1 > bound (positive autocorrelation) and with |r1| > bound.
#' `ac`: metric, <group>, r1, bound.
autocorrelation_summary <- function(ac, group, group_order, pooled_label = "All PAs") {
  one <- function(x, m, gname) {
    r <- x$r1; ok <- !is.na(r); b <- x$bound[ok]; r <- r[ok]
    q <- stats::quantile(r, c(0.25, 0.5, 0.75), names = FALSE)
    out <- data.frame(metric = m, g = gname, n = length(r), median_r1 = q[2], q25_r1 = q[1], q75_r1 = q[3],
                      pct_positive_above_bound = 100 * mean(r > b), pct_abs_above_bound = 100 * mean(abs(r) > b),
                      stringsAsFactors = FALSE)
    names(out)[2] <- group
    out
  }
  out <- do.call(rbind, lapply(unique(ac$metric), function(m) {
    x <- ac[ac$metric == m, ]
    gs <- group_order[group_order %in% x[[group]]]
    do.call(rbind, c(list(one(x, m, pooled_label)), lapply(gs, function(s) one(x[x[[group]] == s, ], m, s))))
  }))
  rownames(out) <- NULL
  out
}

#' Yue–Pilon trend per PA: Sen slope and the Mann–Kendall p-value after pre-whitening
#' (zyp::zyp.trend.vector, method "yuepilon").
#'
#' Rows with a missing response are dropped; PAs with fewer than `min_years` years are left
#' out. A series for which zyp fails gets NA. Returns WDPA_PID, sen, p (first-appearance order).
yuepilon_trends <- function(d, response, min_years) {
  stopifnot(all(c("WDPA_PID", "year", response) %in% names(d)))
  ok <- !is.na(d[[response]])
  y <- d[[response]][ok]; x <- as.numeric(d$year[ok])
  g <- .group_index(d$WDPA_PID[ok])
  rows <- split(seq_along(y), factor(g$index, levels = seq_along(g$levels)))
  n <- lengths(rows)
  rows <- rows[n >= min_years]
  pids <- g$levels[n >= min_years]

  res <- vapply(rows, function(i) {
    out <- tryCatch(zyp::zyp.trend.vector(y[i], x[i], method = "yuepilon", conf.intervals = FALSE),
                    error = function(e) NULL)
    if (is.null(out)) c(NA_real_, NA_real_) else c(unname(out[["trend"]]), unname(out[["sig"]]))
  }, numeric(2))
  data.frame(WDPA_PID = pids, sen = res[1, ], p = res[2, ], stringsAsFactors = FALSE)
}

#' All per-PA trend columns for one metric: OLS + BH class, and Yue–Pilon + BH class.
#'
#' `suffix` names the metric in the output columns (e.g. "extent"):
#' slope_<s>, se_<s>, stat_<s>, p_value_<s>, p_fdr_<s>, trend_<s>,
#' sen_<s>, p_zyp_<s>, p_fdr_zyp_<s>, trend_<s>_zyp.
#' One row per PA with an OLS trend (≥ min_years), in first-appearance order.
pa_trends <- function(d, response, suffix, min_years, fdr, levels) {
  ols <- ols_trends(d, response, min_years)
  bh  <- bh_classes(ols$slope, ols$p_value, fdr, levels)
  yp  <- yuepilon_trends(d, response, min_years)
  ybh <- bh_classes(yp$sen, yp$p, fdr, levels)
  i <- match(ols$WDPA_PID, yp$WDPA_PID)
  out <- data.frame(
    WDPA_PID = ols$WDPA_PID,
    slope = ols$slope, se = ols$se, stat = ols$stat, p_value = ols$p_value,
    p_fdr = bh$p_fdr, trend = bh$class,
    sen = yp$sen[i], p_zyp = yp$p[i], p_fdr_zyp = ybh$p_fdr[i], trend_zyp = ybh$class[i],
    stringsAsFactors = FALSE
  )
  names(out)[-1] <- c(paste0(c("slope_", "se_", "stat_", "p_value_", "p_fdr_", "trend_"), suffix),
                      paste0(c("sen_", "p_zyp_", "p_fdr_zyp_"), suffix),
                      paste0("trend_", suffix, "_zyp"))
  out
}

# ── Standardised change, bootstraps, biome GLS, correlation, % change ────────
# Input is one row per PA. `cols` arguments are named character vectors, metric label → column,
# e.g. c("Edge extent" = "slope_extent", "Edge intensity" = "slope_intensity"); outputs carry
# the label in a `metric` column and the group in a column named by `group`.

#' Standardised change over the record: slope × n_span / sd, in response SDs per
#' record. The SE is rescaled the same way (used by the precision weights of the change models).
standardised_change <- function(slope, se, n_span, sd) {
  stopifnot(length(sd) == 1, is.finite(sd), sd > 0)
  k <- n_span / sd
  list(change = slope * k, se_change = se * k, k = k)
}

#' Percentile-bootstrap means by group, with boot::boot and boot.ci(type = "perc").
#'
#' PAs are resampled within each group (same n as the group), R times. With `weights` (a column
#' name) the statistic is the weighted mean, resampling (value, weight) pairs; a failed
#' bootstrap gives NA limits. Rows with a missing value or group are left out.
#'
#' The random-number stream: `seed` is set once, then the bootstraps run in this order —
#'   pooled = "none":   for each metric, each group in `levels` order;
#'   pooled = "only":   for each metric, all PAs as one group (`pooled_label`);
#'   pooled = "append": as "none", then for each metric the pooled group.
#' "append" leaves the group results identical to "none", so a pooled row can be added to a
#' block without changing the by-group CIs. Within a group, rows keep their order in `d`
#' (the resampled indices refer to it).
#'
#' Returns list(summary, reps). summary: metric, <group>, n, mean, ci_lo, ci_hi (rows in run
#' order). reps: metric, <group>, rep, mean (the R bootstrap means; for Fig. 4A).
boot_means <- function(d, cols, group, levels, R, seed, weights = NULL,
                       pooled = c("none", "only", "append"), pooled_label = "All PAs") {
  pooled <- match.arg(pooled)
  stopifnot(!is.null(names(cols)), all(c(cols, group, weights) %in% names(d)),
            !pooled_label %in% levels)
  stat <- if (is.null(weights)) {
    function(dat, i) mean(dat$x[i])
  } else {
    function(dat, i) stats::weighted.mean(dat$x[i], w = dat$w[i])
  }
  one <- function(metric, g, x, w) {
    dat <- data.frame(x = x)
    if (!is.null(weights)) dat$w <- w
    bt <- boot::boot(dat, stat, R = R)
    ci <- tryCatch(boot::boot.ci(bt, type = "perc")$percent[4:5],
                   error = function(e) c(NA_real_, NA_real_))
    est <- if (is.null(weights)) mean(x) else stats::weighted.mean(x, w)
    list(summary = data.frame(metric = metric, g = g, n = length(x), mean = est,
                              ci_lo = ci[1], ci_hi = ci[2], stringsAsFactors = FALSE),
         reps = data.frame(metric = metric, g = g, rep = seq_len(R), mean = bt$t[, 1],
                           stringsAsFactors = FALSE))
  }
  jobs <- list()
  for (m in names(cols)) {   # by group
    if (pooled == "only") break
    ok <- !is.na(d[[cols[[m]]]]) & !is.na(d[[group]])
    for (g in levels[levels %in% d[[group]][ok]]) jobs[[length(jobs) + 1]] <- list(m, g, ok & d[[group]] %in% g)
  }
  if (pooled != "none") for (m in names(cols)) {
    ok <- !is.na(d[[cols[[m]]]]) & !is.na(d[[group]])
    jobs[[length(jobs) + 1]] <- list(m, pooled_label, ok)
  }
  if (!is.null(seed)) set.seed(seed)
  res <- lapply(jobs, function(j) {
    rows <- which(j[[3]])
    one(j[[1]], j[[2]], d[[cols[[j[[1]]]]]][rows], if (!is.null(weights)) d[[weights]][rows])
  })
  out <- lapply(c(summary = "summary", reps = "reps"), function(k) {
    x <- do.call(rbind, lapply(res, `[[`, k))
    names(x)[names(x) == "g"] <- group
    rownames(x) <- NULL
    x
  })
  out
}

#' Do the groups differ in mean change? Non-spatial GLS of `response` on `group`
#' with a separate residual variance per group (varIdent), REML; emmeans with mode "df.error".
#'
#' Returns list(omnibus, means, pairs, letters, sigma):
#'   omnibus  marginal Wald F for the group term (denDF = N − p);
#'   means    estimated group means with 95% CIs;
#'   pairs    all pairwise contrasts, Holm-adjusted p (Bonferroni-simultaneous CIs);
#'   letters  compact letter display at `alpha` from the Holm p-values (multcompView);
#'   sigma    residual SD per group (sigma × varIdent ratio).
#' Groups take the order of `levels` (the first is the model's reference); rows with a missing
#' response or group are left out, the remaining rows keep their order in `d`.
biome_gls <- function(d, response, group, levels, max_iter = 200L, alpha = 0.05) {
  stopifnot(all(c(response, group) %in% names(d)))
  dd <- data.frame(y = d[[response]], g = factor(as.character(d[[group]]), levels = levels))
  dd <- dd[!is.na(dd$y) & !is.na(dd$g), , drop = FALSE]
  dd$g <- droplevels(dd$g)
  m <- nlme::gls(y ~ g, data = dd, weights = nlme::varIdent(form = ~ 1 | g), method = "REML",
                 control = nlme::glsControl(maxIter = max_iter, msMaxIter = max_iter))

  a <- stats::anova(m, type = "marginal")
  omnibus <- data.frame(term = group, numDF = a["g", "numDF"], denDF = m$dims$N - m$dims$p,
                        F_value = a["g", "F-value"], p_value = a["g", "p-value"],
                        stringsAsFactors = FALSE)

  em <- emmeans::emmeans(m, ~ g, data = dd, mode = "df.error")
  means <- as.data.frame(summary(em, infer = c(TRUE, TRUE)))
  pr <- emmeans::contrast(em, method = "pairwise")
  pairs <- as.data.frame(summary(pr, infer = c(TRUE, TRUE), adjust = "holm"))

  # Letters from the Holm p-values. emmeans orders the pairs as combn(k, 2); checked on labels.
  lv  <- levels(dd$g)
  idx <- utils::combn(length(lv), 2)
  stopifnot(nrow(pairs) == ncol(idx),
            identical(as.character(pairs$contrast), paste(lv[idx[1, ]], "-", lv[idx[2, ]])))
  pv <- stats::setNames(pairs$p.value, paste(idx[1, ], idx[2, ], sep = "-"))
  L <- multcompView::multcompLetters(pv, threshold = alpha)$Letters
  letters <- data.frame(g = lv, letters = unname(L[as.character(seq_along(lv))]),
                        stringsAsFactors = FALSE)

  vr <- stats::coef(m$modelStruct$varStruct, unconstrained = FALSE, allCoef = TRUE)
  sigma <- data.frame(g = lv, resid_sd = unname(m$sigma * vr[lv]), stringsAsFactors = FALSE)

  means$g <- as.character(means$g)
  pairs$contrast <- as.character(pairs$contrast)
  rename_g <- function(x) { names(x)[names(x) == "g"] <- group; rownames(x) <- NULL; x }
  list(omnibus = omnibus, means = rename_g(means), pairs = rename_g(pairs),
       letters = rename_g(letters), sigma = rename_g(sigma))
}

#' Correlation of two per-PA slopes: Pearson with its 95% CI, and Spearman
#' (exact = FALSE) as a check; all PAs (`pooled_label`) and by group in `levels` order.
#' PAs missing either slope are left out.
slope_correlation <- function(d, x, y, group, levels, pooled_label = "All PAs") {
  one <- function(rows, label) {
    a <- d[[x]][rows]; b <- d[[y]][rows]
    ok <- !is.na(a) & !is.na(b); a <- a[ok]; b <- b[ok]
    ct <- stats::cor.test(a, b, method = "pearson")
    sp <- suppressWarnings(stats::cor.test(a, b, method = "spearman", exact = FALSE))
    data.frame(stratum = label, n = length(a), r_pearson = unname(ct$estimate),
               r_lo = ct$conf.int[1], r_hi = ct$conf.int[2], p_pearson = ct$p.value,
               rho_spearman = unname(sp$estimate), p_spearman = sp$p.value,
               stringsAsFactors = FALSE)
  }
  g <- as.character(d[[group]])
  out <- rbind(one(rep(TRUE, nrow(d)), pooled_label),
               do.call(rbind, lapply(levels[levels %in% g], function(b) one(g %in% b, b))))
  rownames(out) <- NULL
  out
}

#' Trend-class counts and percentages by group and for all PAs (`pooled_label`, first).
#' Every class in `class_levels` gets a row (n = 0 if absent). PAs with no class are left out.
class_counts <- function(d, cols, group, levels, class_levels, pooled_label = "All PAs") {
  g <- as.character(d[[group]])
  out <- do.call(rbind, lapply(names(cols), function(m) {
    cls <- d[[cols[[m]]]]
    do.call(rbind, lapply(c(pooled_label, levels[levels %in% g[!is.na(cls)]]), function(s) {
      rows <- !is.na(cls) & (s == pooled_label | g %in% s)
      n <- as.vector(table(factor(cls[rows], levels = class_levels)))
      data.frame(metric = m, s = s, class = class_levels, n = n, pct = 100 * n / sum(n),
                 stringsAsFactors = FALSE)
    }))
  }))
  names(out)[names(out) == "s"] <- group
  rownames(out) <- NULL
  out
}

#' Agreement of the OLS + BH classes with the Yue–Pilon classes (robustness check).
#' `pairs` is a named list, metric label → c(ols column, Yue–Pilon column).
#' Returns list(agreement, crosstab):
#'   agreement  metric, <group>, n, n_concordant, pct_concordant and the % increase / decrease
#'              under each method; all PAs (`pooled_label`) and by group; PAs with both classes;
#'   crosstab   metric, ols, zyp, n over all PAs (NA classes kept as their own row/column).
trend_agreement <- function(d, pairs, group, levels, class_levels, pooled_label = "All PAs") {
  g <- as.character(d[[group]])
  agreement <- do.call(rbind, lapply(names(pairs), function(m) {
    o <- d[[pairs[[m]][1]]]; z <- d[[pairs[[m]][2]]]
    both <- !is.na(o) & !is.na(z)
    do.call(rbind, lapply(c(pooled_label, levels[levels %in% g[both]]), function(s) {
      r <- both & (s == pooled_label | g %in% s)
      data.frame(metric = m, s = s, n = sum(r), n_concordant = sum(o[r] == z[r]),
                 pct_concordant = 100 * mean(o[r] == z[r]),
                 pct_increase_ols = 100 * mean(o[r] == class_levels[1]),
                 pct_decrease_ols = 100 * mean(o[r] == class_levels[3]),
                 pct_increase_zyp = 100 * mean(z[r] == class_levels[1]),
                 pct_decrease_zyp = 100 * mean(z[r] == class_levels[3]),
                 stringsAsFactors = FALSE)
    }))
  }))
  names(agreement)[names(agreement) == "s"] <- group
  crosstab <- do.call(rbind, lapply(names(pairs), function(m) {
    t <- as.data.frame(table(ols = factor(d[[pairs[[m]][1]]], levels = class_levels),
                             zyp = factor(d[[pairs[[m]][2]]], levels = class_levels),
                             useNA = "ifany"), stringsAsFactors = FALSE)
    data.frame(metric = m, ols = t$ols, zyp = t$zyp, n = t$Freq, stringsAsFactors = FALSE)
  }))
  rownames(agreement) <- rownames(crosstab) <- NULL
  list(agreement = agreement, crosstab = crosstab)
}

#' Mean change over the record as % of the baseline (Results text; diagnostics).
#'
#' `summary` is a boot_means() summary in slope units (metric, <group>, n, mean, ci_lo, ci_hi).
#' The baseline of a group is the unweighted mean of the PA means (`base_cols`, metric → column)
#' over the PAs that enter its trend mean (non-missing slope, `slope_cols`); for
#' `pooled_label`, over all such PAs. pct = 100 × slope × n_span / baseline, for the mean and
#' both CI limits (the baseline is held fixed).
percent_change <- function(summary, d, slope_cols, base_cols, group, n_span,
                           pooled_label = "All PAs") {
  stopifnot(identical(sort(names(slope_cols)), sort(names(base_cols))))
  g <- as.character(d[[group]])
  base <- vapply(seq_len(nrow(summary)), function(i) {
    m <- summary$metric[i]; s <- summary[[group]][i]
    rows <- !is.na(d[[slope_cols[[m]]]]) & !is.na(g) & (s == pooled_label | g %in% s)
    mean(d[[base_cols[[m]]]][rows], na.rm = TRUE)
  }, numeric(1))
  out <- summary[, c("metric", group, "n")]
  out$baseline   <- base
  out$pct_change <- 100 * summary$mean  * n_span / base
  out$pct_lo     <- 100 * summary$ci_lo * n_span / base
  out$pct_hi     <- 100 * summary$ci_hi * n_span / base
  out
}

# ── Fig. 4B case studies (config case_studies) ───────────────────────────────

#' Annual series of the case-study PAs, in panel order.
#' `py`: PA-year table of one transect version (WDPA_PID, year, the response columns);
#' `cases`: data frame WDPA_PID, label (config case_studies); `responses`: c(extent = "edge_extent", ...).
case_series <- function(py, cases, responses) {
  stopifnot(!anyDuplicated(cases$WDPA_PID), all(cases$WDPA_PID %in% py$WDPA_PID))
  x <- py[py$WDPA_PID %in% cases$WDPA_PID, c("WDPA_PID", "year", unname(responses))]
  x$panel <- match(x$WDPA_PID, cases$WDPA_PID)
  x$label <- cases$label[x$panel]
  x <- x[order(x$panel, x$year), c("panel", "WDPA_PID", "label", "year", unname(responses))]
  rownames(x) <- NULL
  x
}

#' Case-study PAs: trends, and the change over the record in metric units (slope × n_span) and
#' as % of the PA's own mean, in panel order.
#' `d`: one row per PA with WDPA_PID, <response>_mean, slope_<m>, p_fdr_<m>, trend_<m>,
#' change_<m>_z (NAME, ISO3, BIOME_NAME kept when present); `cases`, `responses`: as case_series().
case_summary <- function(d, cases, responses, n_span) {
  i <- match(cases$WDPA_PID, d$WDPA_PID)
  stopifnot(!anyNA(i))
  out <- data.frame(panel = seq_len(nrow(cases)), WDPA_PID = cases$WDPA_PID, label = cases$label,
                    d[i, intersect(c("NAME", "ISO3", "BIOME_NAME"), names(d)), drop = FALSE],
                    stringsAsFactors = FALSE)
  for (m in names(responses)) {
    base  <- d[[paste0(responses[[m]], "_mean")]][i]
    slope <- d[[paste0("slope_", m)]][i]
    out[[paste0("mean_", m)]]       <- base
    out[[paste0("slope_", m)]]      <- slope
    out[[paste0("p_fdr_", m)]]      <- d[[paste0("p_fdr_", m)]][i]
    out[[paste0("trend_", m)]]      <- d[[paste0("trend_", m)]][i]
    out[[paste0("change_", m)]]     <- slope * n_span
    out[[paste0("pct_change_", m)]] <- 100 * slope * n_span / base
    out[[paste0("change_", m, "_z")]] <- d[[paste0("change_", m, "_z")]][i]
  }
  rownames(out) <- NULL
  out
}

# ── Sensitivity (09): agreement of per-PA values between two transect versions ──

#' Cohen's kappa of two classifications of the same PAs (pairs with an NA are left out).
#' NA when the expected agreement is 1 (both classifications constant and equal).
cohen_kappa <- function(a, b, levels) {
  ok <- !is.na(a) & !is.na(b)
  t <- table(factor(a[ok], levels = levels), factor(b[ok], levels = levels))
  n <- sum(t)
  po <- sum(diag(t)) / n
  pe <- sum(rowSums(t) * colSums(t)) / n^2
  if (isTRUE(all.equal(pe, 1))) return(NA_real_)
  (po - pe) / (1 - pe)
}

#' Agreement of a per-PA quantity under two versions: n, Pearson r, and the paired mean
#' difference (b − a) / `scale` with a percentile-bootstrap 95% CI over PAs (`R` resamples,
#' `seed` set first). Pairs with an NA are left out.
paired_agreement <- function(a, b, scale, R, seed) {
  stopifnot(length(a) == length(b), length(scale) == 1, scale > 0)
  ok <- !is.na(a) & !is.na(b)
  a <- a[ok]; b <- b[ok]
  dz <- (b - a) / scale
  set.seed(seed)
  bt <- boot::boot(dz, function(x, i) mean(x[i]), R = R)
  # boot.ci prints a note and returns NULL when all replicates are equal: the CI is then that value
  res <- NULL
  utils::capture.output(res <- tryCatch(boot::boot.ci(bt, type = "perc"), error = function(e) NULL))
  ci <- if (is.null(res)) rep(mean(dz), 2) else res$percent[4:5]
  data.frame(n = length(a), r = stats::cor(a, b), mean_diff_sd = mean(dz), diff_lo = ci[1], diff_hi = ci[2])
}
