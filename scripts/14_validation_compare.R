# 14_validation_compare.R: validation, part 4 — each PA against its displaced copy (Methods:
# Validation).
#
# The displaced copies (Python scripts 11–13) are measured exactly as the PAs: PA means of the
# yearly edge metrics, PAs with ≥ min_years of both metrics. Paired difference PA − copy, per
# biome and for all PAs: mean with a percentile-bootstrap 95% CI (10,000 resamples of PAs within
# the biome), median, share of PAs above their copy, and a paired Wilcoxon signed-rank test
# (Holm-adjusted across biomes within a metric). All PAs: the sampled PAs pooled, each biome
# weighted equally (shown in the figure; the conservative estimate), and in the table also
# "All PAs, PA-weighted": each biome weighted by its share of the analysis set (pa_level.csv),
# with a within-biome bootstrap CI, weighted median and share, and a design-based t test
# (p_stratified_t). Differences are also given in response SDs
# (the analysis set's SD of PA means). The chance rate of edge extent (8/15, five exchangeable
# values) is a reference line.
#
# Run from the repository root, after 05 and Python scripts 11–13:
#   Rscript scripts/14_validation_compare.R [--from DIR] [--out DIR]
# --from DIR  read validation_displaced_pa_years.csv and validation_displacement_log.csv from DIR
# --out DIR   write tables, diagnostics and figures under DIR instead of results/
#
# Inputs:  data/05_analysis/pa_level.csv, scaling.csv (05); results/validation_displaced_pa_years.csv,
#          results/validation_displacement_log.csv (13)
# Outputs: <tables>/TableS_validation.csv            paired summary per biome and all PAs
#          <diagnostics>/validation_paired_pa.csv      one row per validation PA
#          <diagnostics>/validation_attrition.csv      sampled PAs by biome and status
#          <figures>/figS_validation.{pdf,png}

source(file.path("modules", "r", "config.R"))
for (f in setdiff(list.files(file.path("modules", "r"), "\\.R$", full.names = TRUE),
                  file.path("modules", "r", "config.R"))) {
  sys.source(f, envir = globalenv())   # corHaversine must live in globalenv()
}

args <- commandArgs(trailingOnly = TRUE)
arg <- function(name) if (any(args == name)) args[which(args == name) + 1] else NULL
src <- if (is.null(arg("--from"))) list(pa_years = CFG$validation$pa_years, log = CFG$validation$log) else
  list(pa_years = file.path(arg("--from"), basename(CFG$validation$pa_years)),
       log = file.path(arg("--from"), basename(CFG$validation$log)))
out <- if (is.null(arg("--out"))) list(tables = CFG$paths$tables, diagnostics = CFG$paths$diagnostics,
                                       figures = CFG$paths$figures) else
  lapply(c(tables = "tables", diagnostics = "diagnostics", figures = "figures"), function(x) file.path(arg("--out"), x))
say <- function(...) cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"), sprintf(...)))
read <- function(path) as.data.frame(readr::read_csv(path, col_types = readr::cols(WDPA_PID = "c", .default = readr::col_guess()),
                                                     guess_max = 1e5, show_col_types = FALSE, progress = FALSE))

pa <- read(file.path(CFG$paths$analysis_dir, "pa_level.csv"))
scaling <- as.data.frame(readr::read_csv(file.path(CFG$paths$analysis_dir, "scaling.csv"), show_col_types = FALSE,
                                          progress = FALSE))   # no WDPA_PID column
nul <- null_pa_means(read(src$pa_years), unname(CFG$responses), CFG$min_years)
log <- read(src$log)
metrics <- names(CFG$responses)
labels <- RESPONSE_LABELS[metrics]

# Pairs: PA means (main analysis, all transects) and the copy's means
p <- merge(pa[c("WDPA_PID", "BIOME_NAME", paste0(CFG$responses, "_mean"))], nul, by = "WDPA_PID",
           suffixes = c("", "_displaced"))
for (m in metrics) {
  col <- paste0(CFG$responses[[m]], "_mean")
  p[[paste0("diff_", m)]] <- p[[col]] - p[[paste0(col, "_displaced")]]
}
p <- p[order(match(p$BIOME_NAME, CFG$biome_order), p$WDPA_PID, method = "radix"), ]
say("14: %d validation PAs with a displaced copy (%d in the log as selected)", nrow(p), sum(log$status == "selected"))

n_pop <- c(table(pa$BIOME_NAME))                     # PAs per biome in the analysis set (weights)
w_label <- paste0(CFG$pooled_label, ", PA-weighted")
summ <- paired_null_summary(p, stats::setNames(paste0("diff_", metrics), labels), "BIOME_NAME", CFG$biome_order,
                            CFG$boot$R, CFG$validation$seed, CFG$pooled_label, n_pop = n_pop, weighted_label = w_label)
sd_resp <- stats::setNames(vapply(metrics, function(m) scaling$sd[scaling$variable == paste0(CFG$responses[[m]], "_mean")],
                                  numeric(1)), labels)
for (col in c("mean_diff", "ci_lo", "ci_hi")) summ[[paste0(col, "_sd")]] <- summ[[col]] / sd_resp[summ$metric]
lev <- do.call(rbind, lapply(c(CFG$biome_order, CFG$pooled_label, w_label), function(b) {
  r <- if (b %in% c(CFG$pooled_label, w_label)) p else p[p$BIOME_NAME %in% b, ]
  if (!nrow(r)) return(NULL)
  avg <- if (b == w_label) {
    w <- pooled_weights(r$BIOME_NAME, n_pop); function(x) stats::weighted.mean(x, w)
  } else mean
  do.call(rbind, lapply(metrics, function(m) {
    col <- paste0(CFG$responses[[m]], "_mean")
    data.frame(metric = labels[[m]], BIOME_NAME = b, mean_pa = avg(r[[col]]),
               mean_displaced = avg(r[[paste0(col, "_displaced")]]), stringsAsFactors = FALSE)
  }))
}))
summ <- merge(summ, lev, by = c("metric", "BIOME_NAME"), sort = FALSE)
summ$chance_rate <- ifelse(summ$metric == labels[["extent"]], CFG$validation$chance_extent, NA_real_)
attr_tab <- as.data.frame(table(BIOME_NAME = log$BIOME_NAME, status = log$status), stringsAsFactors = FALSE)
attr_tab <- attr_tab[attr_tab$Freq > 0, ]

for (dir in out) dir.create(dir, recursive = TRUE, showWarnings = FALSE)
readr::write_csv(summ, file.path(out$tables, "TableS_validation.csv"), na = "")
readr::write_csv(p, file.path(out$diagnostics, "validation_paired_pa.csv"), na = "")
readr::write_csv(attr_tab, file.path(out$diagnostics, "validation_attrition.csv"))
fig <- fig_validation(summ, CFG$biome_order, CFG$pooled_label)
ggplot2::ggsave(file.path(out$figures, "figS_validation.pdf"), fig, width = 9, height = 4.5, device = grDevices::cairo_pdf)
ggplot2::ggsave(file.path(out$figures, "figS_validation.png"), fig, width = 9, height = 4.5, dpi = 300)
for (i in which(summ$BIOME_NAME == CFG$pooled_label)) {
  say("%s, all PAs (equal weight per biome; figure): PA − copy %.3f (%.3f, %.3f) = %.2f SD; %.0f%% of PAs above their copy; p = %.2g (Wilcoxon)",
      summ$metric[i], summ$mean_diff[i], summ$ci_lo[i], summ$ci_hi[i], summ$mean_diff_sd[i], 100 * summ$share_positive[i],
      summ$p_wilcoxon[i])
}
for (i in which(summ$BIOME_NAME == w_label)) {
  say("%s, all PAs (PA-weighted; table): PA − copy %.3f (%.3f, %.3f) = %.2f SD; %.0f%% of PAs above their copy; p = %.2g (stratified t)",
      summ$metric[i], summ$mean_diff[i], summ$ci_lo[i], summ$ci_hi[i], summ$mean_diff_sd[i], 100 * summ$share_positive[i],
      summ$p_stratified_t[i])
}
say("wrote %s, %s, %s", out$tables, out$diagnostics, out$figures)
