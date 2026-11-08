# Per-PA trends (Stats ¶4–5): OLS slope on year, Benjamini–Hochberg classes, and the
# Yue–Pilon (pre-whitened Sen / Mann–Kendall) robustness check.
#
# Input is long PA-year data: one row per PA and year, with `WDPA_PID`, `year` and the
# response column. One call handles one series (one metric, one transect version), so the
# sensitivity runs are a loop over the same functions (A1, A2).
#
# Ported from PA_New_Analysis/scripts/02_trend_analysis.R (63-128, 437-470). The OLS is computed
# from per-PA sums instead of one lm() per PA (A6); slope, SE, t and p are the same as lm().

# Group index in first-appearance order. rowsum() sorts character groups by the locale, which
# would make the row order depend on the environment.
.group_index <- function(id) {
  f <- factor(id, levels = unique(id))
  list(levels = levels(f), index = as.integer(f))
}

#' Per-PA OLS of `response` on year.
#'
#' Rows with a missing response are dropped. PAs with fewer than `min_years` remaining years
#' are left out (as 02:68). Returns one row per PA, in first-appearance order:
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

#' Yue–Pilon trend per PA: Sen slope and the Mann–Kendall p-value after pre-whitening
#' (zyp::zyp.trend.vector, method "yuepilon"), as 02:442-450.
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
