# Validation (methods: Validation; draft for review): each sampled PA against its displaced copy.
# Ported from PA_New_Analysis/scripts/09_null_transects.R (sections 4 and 6): PA means of the
# copy's yearly metrics (as for a PA), paired differences PA − copy, and per biome and for all PAs
# the mean difference with a percentile-bootstrap CI (resampling PAs within the biome), the median,
# the share of PAs above their copy, and a paired Wilcoxon signed-rank test (Holm across biomes
# within a metric). All PAs: the sampled PAs pooled (equal weight per biome; the figure), and the
# PA-weighted (stratified) estimate, biomes weighted by their share of the analysis set (user,
# 2026-11-11; table only). The displaced metrics come from Python script 13.

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

#' PA weights of the stratified (PA-weighted) pooled estimate: PA i in group g gets
#' (n_pop[g] / Σ n_pop) / n_g, n_g = the PAs of `g` in `group_vec` (the groups present; the
#' weights sum to 1). `n_pop`: named vector of the PAs per group in the population (analysis set).
pooled_weights <- function(group_vec, n_pop) {
  g <- as.character(group_vec)
  stopifnot(!is.null(names(n_pop)), all(g %in% names(n_pop)))
  n_g <- table(g)
  W <- n_pop[names(n_g)] / sum(n_pop[names(n_g)])
  as.numeric(W[g] / n_g[g])
}

#' Weighted median: the smallest x whose cumulative weight reaches half.
weighted_median <- function(x, w) {
  o <- order(x)
  x[o][which(cumsum(w[o]) >= sum(w) / 2)[1]]
}

#' Paired PA − copy summary per group and for all PAs (`pooled_label`, last).
#'
#' `paired`: one row per PA with `group` and the difference columns `cols` (named: metric label →
#' column). Mean and CI from boot_means() (`R`, `seed`; group rows, then the pooled row). Adds the
#' median difference, the share of PAs with a positive difference, the Wilcoxon signed-rank p
#' (normal approximation) and, for the groups, its Holm adjustment across groups within a metric.
#'
#' With `n_pop` (PAs per group in the analysis set; user, 2026-11-11) a `weighted_label` row per
#' metric is appended: the PA-weighted (stratified) estimate, each group weighted by its share of
#' the population (pooled_weights()): weighted mean, a percentile-bootstrap CI resampling PAs
#' within groups (boot strata; `R`, `seed`), weighted median and share positive, and
#' `p_stratified_t`, the design-based t test (SE² = Σ W_g² s_g² / n_g, df = n − groups; no
#' finite-population correction, as the bootstrap). The group rows and the (equal-weight)
#' `pooled_label` row are the same with or without `n_pop`.
paired_null_summary <- function(paired, cols, group, levels, R, seed, pooled_label = "All PAs",
                                n_pop = NULL, weighted_label = paste0(pooled_label, ", PA-weighted")) {
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
  if (is.null(n_pop)) return(s)

  s$p_stratified_t <- NA_real_
  set.seed(seed)
  wrows <- lapply(names(cols), function(m) {
    x <- paired[[cols[[m]]]]
    g <- as.character(paired[[group]])
    ok <- !is.na(x) & g %in% levels
    x <- x[ok]; g <- g[ok]
    w <- pooled_weights(g, n_pop)
    bt <- boot::boot(data.frame(x = x, w = w), function(dat, i) sum(dat$w[i] * dat$x[i]), R = R,
                     strata = factor(g))
    ci <- boot::boot.ci(bt, type = "perc")$percent[4:5]
    n_g <- table(g)
    W <- n_pop[names(n_g)] / sum(n_pop[names(n_g)])
    se <- sqrt(sum(W^2 * tapply(x, g, stats::var)[names(n_g)] / as.numeric(n_g)))
    est <- sum(w * x)
    out <- data.frame(metric = m, grp = weighted_label, n = length(x), mean_diff = est, ci_lo = ci[1],
                      ci_hi = ci[2], median = weighted_median(x, w), share_positive = sum(w * (x > 0)),
                      p_wilcoxon = NA_real_, p_wilcoxon_holm = NA_real_,
                      p_stratified_t = 2 * stats::pt(-abs(est / se), df = length(x) - length(n_g)),
                      stringsAsFactors = FALSE)
    names(out)[names(out) == "grp"] <- group
    out
  })
  s <- rbind(s, do.call(rbind, wrows)[names(s)])
  rownames(s) <- NULL
  s
}
