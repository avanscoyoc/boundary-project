# 07_trends.R: per-PA trends and the biome comparison of change (Stats ¶4–5).
#
#   per-PA OLS slope of each metric on year, Benjamini–Hochberg across PAs per metric, classes
#   (increase / no trend / decrease at FDR config trend$fdr); Yue–Pilon (pre-whitened Sen and
#   Mann–Kendall) as the robustness check, with the class agreement;
#   Pearson correlation of the extent and intensity slopes, overall and per biome (Spearman as a
#   check; D2);
#   standardised change = slope × n_span / SD of the metric's PA means over the analysis set
#   (scaling.csv from 05), with its SE (for the precision weights of 08);
#   percentile-bootstrap means of change per biome and for all PAs (Fig. 4A), area-weighted means
#   and Sen-slope means (Table SV); seeds and run order as the old run (config boot$seeds);
#   GLS of change on biome with biome-specific variances, Holm-adjusted pairwise contrasts and
#   letters (Table SW); % change over the record (diagnostic);
#   the Fig. 4B case-study PAs (config case_studies): their annual series, trends and % change
#   of their own mean (old 07:207-235 and 11:58-81);
#   lag-1 autocorrelation of each PA's OLS trend residuals (diagnostic; Finish plan Step 5).
# Main analysis only: transect version "all", the analysis set of 05. The other versions and
# PA subsets are 09's.
#
# Run from the repository root (about a minute):
#   Rscript scripts/07_trends.R [--out DIR]
# --out DIR writes every output under DIR instead (a dry run).
#
# Inputs:  data/05_analysis/pa_level.csv, pa_year.csv, scaling.csv (05)
# Outputs: data/05_analysis/pa_trends.csv            one row per PA: trend columns, change_*_z,
#                                                    se_change_*_z (input of 08, 09 and the S fig)
#          <tables>/TableSV_biome_trend_summary.csv   per biome and all PAs: mean change (CI),
#                                                    area-weighted, % per class, Yue–Pilon
#                                                    agreement, Sen-slope change (unrounded)
#          <tables>/TableSV_slope_correlation.csv
#          <tables>/TableSW_biome_{omnibus,contrasts,letters}.csv
#          <tables>/fig4a_change_summary.csv, fig4a_boot_replicates.csv
#          <tables>/fig4b_case_series.csv (one row per case PA × year), fig4b_case_pas.csv (one
#                         row per case PA: trends, change over the record, % of its own mean)
#          <diagnostics>/trend_{class_counts,yuepilon_agreement,yuepilon_crosstab,
#                         percent_change_biome}.csv, biome_change_gls_{means,sigma}.csv,
#                         trend_residual_autocorrelation{,_summary}.csv (per PA; per biome and
#                         all PAs)

source(file.path("modules", "r", "config.R"))
for (f in setdiff(list.files(file.path("modules", "r"), "\\.R$", full.names = TRUE),
                  file.path("modules", "r", "config.R"))) {
  sys.source(f, envir = globalenv())   # corHaversine must live in globalenv()
}

args <- commandArgs(trailingOnly = TRUE)
out_root <- if (any(args == "--out")) args[which(args == "--out") + 1] else NULL
out <- if (is.null(out_root)) {
  list(data = CFG$paths$analysis_dir, tables = CFG$paths$tables, diagnostics = CFG$paths$diagnostics)
} else {
  list(data = file.path(out_root, "05_analysis"), tables = file.path(out_root, "tables"),
       diagnostics = file.path(out_root, "diagnostics"))
}
say <- function(...) cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"), sprintf(...)))
read <- function(name, ...) {
  as.data.frame(readr::read_csv(file.path(CFG$paths$analysis_dir, name), guess_max = 1e5,
                                show_col_types = FALSE, progress = FALSE, ...))
}

# ── 1. Inputs ────────────────────────────────────────────────────────────────
pa <- read("pa_level.csv", col_types = readr::cols(WDPA_PID = "c", .default = readr::col_guess()))
py <- read("pa_year.csv", col_types = readr::cols(WDPA_PID = "c", transects = "c", .default = readr::col_guess()))
scaling <- read("scaling.csv")
py <- py[py$transects == "all" & py$WDPA_PID %in% pa$WDPA_PID, ]
metrics <- names(CFG$responses)                                   # extent, intensity
labels  <- RESPONSE_LABELS[metrics]                                # Edge extent, Edge intensity
cols <- function(prefix) stats::setNames(paste0(prefix, metrics), labels)
say("07: %d PAs, %d PA-years (transects: all)", nrow(pa), nrow(py))
cases <- CFG$case_studies
gone <- setdiff(cases$WDPA_PID, pa$WDPA_PID)
if (length(gone)) stop("Fig. 4B case-study PAs not in the analysis set: ", paste(gone, collapse = ", "),
                       " (update config case_studies)")

# ── 2. Per-PA trends and standardised change ─────────────────────────────────
d <- pa[c("WDPA_PID", "BIOME_NAME", "AREA_KM2", paste0(CFG$responses, "_mean"))]
k <- c()
for (m in metrics) {
  tr <- pa_trends(py, CFG$responses[[m]], m, CFG$min_years, CFG$trend$fdr, CFG$trend$levels)
  i <- match(d$WDPA_PID, tr$WDPA_PID)
  if (anyNA(i)) stop(sum(is.na(i)), " analysis PAs have no ", m, " trend")
  d <- cbind(d, tr[i, -1])
  sd_m <- scaling$sd[scaling$variable == paste0(CFG$responses[[m]], "_mean")]
  ch <- standardised_change(d[[paste0("slope_", m)]], d[[paste0("se_", m)]], CFG$n_span, sd_m)
  d[[paste0("change_", m, "_z")]] <- ch$change
  d[[paste0("se_change_", m, "_z")]] <- ch$se_change
  k[[labels[[m]]]] <- ch$k
  say("%s: %s", m, paste(names(table(d[[paste0("trend_", m)]])), table(d[[paste0("trend_", m)]]),
                         sep = " ", collapse = ", "))
}
k <- unlist(k)
rownames(d) <- NULL

# Fig. 4B case studies
case_pas <- case_summary(cbind(d, pa[match(d$WDPA_PID, pa$WDPA_PID), c("NAME", "ISO3")]), cases,
                         CFG$responses, CFG$n_span)
case_ser <- case_series(py, cases, CFG$responses)
got <- paste(case_pas$trend_extent, case_pas$trend_intensity, sep = "/")
for (i in seq_len(nrow(cases))) say("Fig. 4B panel %d: %s (%s): %s%s", i, cases$label[i], cases$WDPA_PID[i], got[i],
                                    if (got[i] == cases$expect[i]) "" else paste0("  ← expected ", cases$expect[i]))
if (any(got != cases$expect))
  warning("Fig. 4B: case-study trend classes differ from config case_studies$expect (see above)", call. = FALSE)

# Lag-1 autocorrelation of the trend residuals (diagnostic)
ac <- do.call(rbind, lapply(metrics, function(m) {
  a <- residual_autocorrelation(py, CFG$responses[[m]], CFG$min_years)
  stopifnot(setequal(a$WDPA_PID, d$WDPA_PID))
  data.frame(metric = labels[[m]], BIOME_NAME = d$BIOME_NAME[match(a$WDPA_PID, d$WDPA_PID)], a,
             stringsAsFactors = FALSE)
}))
ac <- ac[order(ac$metric, ac$WDPA_PID), ]
rownames(ac) <- NULL
ac_sum <- autocorrelation_summary(ac, "BIOME_NAME", CFG$biome_order, CFG$pooled_label)
for (i in which(ac_sum$BIOME_NAME == CFG$pooled_label))
  say("%s residual lag-1 autocorrelation: median %.2f (IQR %.2f to %.2f); %.1f%% of PAs above +1.96/sqrt(n)",
      ac_sum$metric[i], ac_sum$median_r1[i], ac_sum$q25_r1[i], ac_sum$q75_r1[i], ac_sum$pct_positive_above_bound[i])

# ── 3. Bootstraps (old seeds and run order; slopes, reported in change units) ─
run <- function(cl, seed, ...) boot_means(d, cl, "BIOME_NAME", CFG$biome_order, CFG$boot$R, seed,
                                          pooled_label = CFG$pooled_label, ...)
B <- list(
  biome    = run(cols("slope_"), CFG$boot$seeds[["biome"]]),
  pooled   = run(cols("slope_"), CFG$boot$seeds[["pooled"]], pooled = "only"),
  weighted = run(cols("slope_"), CFG$boot$seeds[["weighted"]], pooled = "append", weights = "AREA_KM2"),
  sen      = run(cols("sen_"), CFG$boot$seeds[["sen"]], pooled = "append"))
to_z <- function(s) {
  data.frame(metric = s$metric, BIOME_NAME = s$BIOME_NAME, n = s$n, mean_change_z = s$mean * k[s$metric],
             ci_lo_z = s$ci_lo * k[s$metric], ci_hi_z = s$ci_hi * k[s$metric], stringsAsFactors = FALSE)
}
main <- to_z(rbind(B$biome$summary, B$pooled$summary))      # biome rows (seed 42) + all PAs (45)
reps <- rbind(B$biome$reps, B$pooled$reps)
reps <- data.frame(metric = reps$metric, BIOME_NAME = reps$BIOME_NAME, rep = reps$rep,
                   change_z = reps$mean * k[reps$metric], stringsAsFactors = FALSE)
say("bootstraps: %d resamples per group", CFG$boot$R)

# ── 4. Classes, agreement, correlation, biome GLS, % change ──────────────────
groups <- c(CFG$biome_order, CFG$pooled_label)
cc <- class_counts(d, cols("trend_"), "BIOME_NAME", CFG$biome_order, CFG$trend$levels, CFG$pooled_label)
pairs <- stats::setNames(lapply(metrics, function(m) c(paste0("trend_", m), paste0("trend_", m, "_zyp"))), labels)
ag <- trend_agreement(d, pairs, "BIOME_NAME", CFG$biome_order, CFG$trend$levels, CFG$pooled_label)
cor_tab <- slope_correlation(d, "slope_extent", "slope_intensity", "BIOME_NAME", CFG$biome_order,
                             CFG$pooled_label)
gls <- lapply(stats::setNames(paste0("change_", metrics, "_z"), labels), function(r)
  biome_gls(d, r, "BIOME_NAME", CFG$biome_order, CFG$spatial$max_iter))
gstack <- function(field) {
  x <- do.call(rbind, lapply(names(gls), function(m) cbind(metric = m, gls[[m]][[field]], stringsAsFactors = FALSE)))
  rownames(x) <- NULL
  x
}
pct <- percent_change(rbind(B$biome$summary, B$pooled$summary), d, cols("slope_"),
                      stats::setNames(paste0(CFG$responses, "_mean"), labels), "BIOME_NAME", CFG$n_span,
                      CFG$pooled_label)

# Table SV: one row per metric × biome (biome order, then all PAs)
key <- function(x) paste(x$metric, x$BIOME_NAME)
sv <- expand.grid(BIOME_NAME = groups, metric = labels, stringsAsFactors = FALSE)[c("metric", "BIOME_NAME")]
sv <- sv[key(sv) %in% key(main), ]
pick <- function(x, col) x[[col]][match(key(sv), key(x))]
w <- to_z(B$weighted$summary); s <- to_z(B$sen$summary)
cls <- function(cl) pick(transform(cc[cc$class == cl, ], BIOME_NAME = BIOME_NAME), "pct")
sv <- data.frame(sv, n_PAs = pick(main, "n"),
                 change_mean = pick(main, "mean_change_z"), change_lo = pick(main, "ci_lo_z"),
                 change_hi = pick(main, "ci_hi_z"),
                 change_area_weighted = pick(w, "mean_change_z"), aw_lo = pick(w, "ci_lo_z"),
                 aw_hi = pick(w, "ci_hi_z"),
                 pct_decrease = cls(CFG$trend$levels[3]), pct_increase = cls(CFG$trend$levels[1]),
                 pct_no_trend = cls(CFG$trend$levels[2]),
                 n_class_compared = pick(ag$agreement, "n"),
                 pct_class_concordant_yuepilon = pick(ag$agreement, "pct_concordant"),
                 sen_change_mean = pick(s, "mean_change_z"), sen_lo = pick(s, "ci_lo_z"), sen_hi = pick(s, "ci_hi_z"),
                 stringsAsFactors = FALSE)
stopifnot(!anyNA(sv[setdiff(names(sv), c("aw_lo", "aw_hi"))]))
rownames(sv) <- NULL

# ── 5. Write ─────────────────────────────────────────────────────────────────
for (dir in out) dir.create(dir, recursive = TRUE, showWarnings = FALSE)
w_csv <- function(x, dir, name) readr::write_csv(x, file.path(dir, name), na = "")
w_csv(d[c("WDPA_PID", "BIOME_NAME", setdiff(names(d), c("WDPA_PID", "BIOME_NAME", "AREA_KM2",
                                                        paste0(CFG$responses, "_mean"))))],
      out$data, "pa_trends.csv")
w_csv(sv, out$tables, "TableSV_biome_trend_summary.csv")
w_csv(cor_tab, out$tables, "TableSV_slope_correlation.csv")
w_csv(gstack("omnibus"), out$tables, "TableSW_biome_omnibus.csv")
w_csv(gstack("pairs"), out$tables, "TableSW_biome_contrasts.csv")
w_csv(gstack("letters"), out$tables, "TableSW_biome_letters.csv")
w_csv(main, out$tables, "fig4a_change_summary.csv")
w_csv(reps, out$tables, "fig4a_boot_replicates.csv")
w_csv(case_ser, out$tables, "fig4b_case_series.csv")
w_csv(case_pas, out$tables, "fig4b_case_pas.csv")
w_csv(cc, out$diagnostics, "trend_class_counts.csv")
w_csv(ag$agreement, out$diagnostics, "trend_yuepilon_agreement.csv")
w_csv(ag$crosstab, out$diagnostics, "trend_yuepilon_crosstab.csv")
w_csv(pct, out$diagnostics, "trend_percent_change_biome.csv")
w_csv(gstack("means"), out$diagnostics, "biome_change_gls_means.csv")
w_csv(gstack("sigma"), out$diagnostics, "biome_change_gls_sigma.csv")
w_csv(ac, out$diagnostics, "trend_residual_autocorrelation.csv")
w_csv(ac_sum, out$diagnostics, "trend_residual_autocorrelation_summary.csv")

om <- gstack("omnibus")
for (i in seq_len(nrow(om))) say("%s biome GLS: F(%d, %d) = %.2f, p = %.3g", om$metric[i], om$numDF[i],
                                 om$denDF[i], om$F_value[i], om$p_value[i])
pm <- main[main$BIOME_NAME == CFG$pooled_label, ]
for (i in seq_len(nrow(pm))) say("%s, all PAs: mean change %.3f (%.3f, %.3f) SD", pm$metric[i],
                                 pm$mean_change_z[i], pm$ci_lo_z[i], pm$ci_hi_z[i])
say("wrote %s, %s, %s", file.path(out$data, "pa_trends.csv"), out$tables, out$diagnostics)
