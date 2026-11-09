# Validation (methods: Validation; draft for review): each sampled PA against its displaced copy.
# Ported from PA_New_Analysis/scripts/09_null_transects.R (sections 4 and 6): PA means of the
# copy's yearly metrics (as for a PA), paired differences PA − copy, and per biome and for all PAs
# the mean difference with a percentile-bootstrap CI (resampling PAs within the biome), the median,
# the share of PAs above their copy, and a paired Wilcoxon signed-rank test (Holm across biomes
# within a metric). The displaced metrics come from Python script 13.

#' PA means of the displaced copies: `py`: WDPA_PID, year and the metric columns (script 13).
#' PAs with fewer than `min_years` years of any metric are left out (as for the PAs, Stats ¶1).
#' Returns WDPA_PID, <metric>_mean, n_years_<metric>.
null_pa_means <- function(py, metrics, min_years) {
  py$transects <- factor("displaced")
  pm <- pa_means(py, metrics)
  ok <- Reduce(`&`, lapply(metrics, function(m) pm[[paste0("n_years_", m)]] >= min_years))
  pm <- pm[ok, setdiff(names(pm), "transects"), drop = FALSE]
  rownames(pm) <- NULL
  pm
}

#' Paired PA − copy summary per group and for all PAs (`pooled_label`, last).
#'
#' `paired`: one row per PA with `group` and the difference columns `cols` (named: metric label →
#' column). Mean and CI from boot_means() (`R`, `seed`; group rows, then the pooled row). Adds the
#' median difference, the share of PAs with a positive difference, the Wilcoxon signed-rank p
#' (normal approximation) and, for the groups, its Holm adjustment across groups within a metric.
paired_null_summary <- function(paired, cols, group, levels, R, seed, pooled_label = "All PAs") {
  s <- boot_means(paired, cols, group, levels, R, seed, pooled = "append", pooled_label = pooled_label)$summary
  g <- as.character(paired[[group]])
  stat <- t(vapply(seq_len(nrow(s)), function(i) {
    x <- paired[[cols[[s$metric[i]]]]]
    x <- x[!is.na(x) & (s[[group]][i] == pooled_label | g %in% s[[group]][i])]
    p <- if (length(x) >= 3) suppressWarnings(stats::wilcox.test(x, mu = 0, exact = FALSE)$p.value) else NA_real_
    c(median = stats::median(x), share_positive = mean(x > 0), p_wilcoxon = p)
  }, numeric(3)))
  s <- cbind(s, stat)
  s$p_wilcoxon_holm <- NA_real_
  for (m in unique(s$metric)) {
    k <- s$metric == m & s[[group]] != pooled_label
    s$p_wilcoxon_holm[k] <- stats::p.adjust(s$p_wilcoxon[k], "holm")
  }
  names(s)[names(s) %in% c("mean", "ci_lo", "ci_hi")] <- c("mean_diff", "ci_lo", "ci_hi")
  rownames(s) <- NULL
  s
}
