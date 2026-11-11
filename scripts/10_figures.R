# 10_figures.R: the figures, from the tables written by 06–08 (A7: nothing is fitted here).
#
#   Fig. 3   A: main-effects coefficients of the mean models in response SDs, against the
#            reference levels of config ref; right: one panel per supported HMI interaction
#            of the main analysis (Wald p < 0.05; B3; user, 2026-11-11: main run only), chosen by
#            fig3_panels(). The panels shown are listed in fig3_panels.csv (for the caption).
#   Fig. 4A  standardised change by biome and for all PAs (bootstrap means and 95% CIs); biome
#            order from config biome_order.
#   S fig    per-PA change in edge extent vs edge intensity, with Pearson r (D2).
#   Diagnostics (D4): Moran's I correlogram of the mean-model residuals; residual plots of the
#            spatial fits.
# A figure is skipped (with a message) when its input tables are missing, e.g. before 08 has run.
#
# Run from the repository root:
#   Rscript scripts/10_figures.R [--from DIR] [--out DIR]
# --from DIR  read tables from DIR/tables and DIR/diagnostics (e.g. a pilot) instead of results/.
# --out DIR   write to DIR/figures and DIR/diagnostics instead of results/.
#
# Inputs:  results/tables/ (06, 07), results/diagnostics/ (06, 08), data/05_analysis/pa_trends.csv (07)
# Outputs: results/figures/fig3_models.{pdf,png}, fig3_panels.csv, fig4a_trends.{pdf,png},
#          figS_slope_correlation.{pdf,png}; results/diagnostics/fig_moran_correlogram_mean_models.png,
#          fig_residuals_<model>_<response>.png; results/R_sessionInfo.txt (without --out)

source(file.path("modules", "r", "config.R"))
for (f in setdiff(list.files(file.path("modules", "r"), "\\.R$", full.names = TRUE),
                  file.path("modules", "r", "config.R"))) {
  sys.source(f, envir = globalenv())   # corHaversine must live in globalenv()
}

args <- commandArgs(trailingOnly = TRUE)
arg <- function(name) if (any(args == name)) args[which(args == name) + 1] else NULL
src <- if (is.null(arg("--from"))) list(tables = CFG$paths$tables, diagnostics = CFG$paths$diagnostics) else
  list(tables = file.path(arg("--from"), "tables"), diagnostics = file.path(arg("--from"), "diagnostics"))
out <- if (is.null(arg("--out"))) list(figures = CFG$paths$figures, diagnostics = CFG$paths$diagnostics) else
  list(figures = file.path(arg("--out"), "figures"), diagnostics = file.path(arg("--out"), "diagnostics"))
for (dir in out) dir.create(dir, recursive = TRUE, showWarnings = FALSE)
say <- function(...) cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"), sprintf(...)))

# Tables by name ("tables/x.csv" or "diagnostics/x.csv"); NULL if any is missing.
need <- function(...) {
  f <- c(...)
  path <- file.path(unlist(src[sub("/.*", "", f)]), sub("^[^/]*/", "", f))
  if (!all(file.exists(path))) {
    say("skipped: missing %s", paste(path[!file.exists(path)], collapse = ", "))
    return(NULL)
  }
  read <- function(p) {
    has_pid <- "WDPA_PID" %in% names(readr::read_csv(p, n_max = 0, show_col_types = FALSE))
    ct <- if (has_pid) readr::cols(WDPA_PID = "c", .default = readr::col_guess()) else readr::cols(.default = readr::col_guess())
    as.data.frame(readr::read_csv(p, col_types = ct, guess_max = 1e5, show_col_types = FALSE, progress = FALSE))
  }
  stats::setNames(lapply(path, read), sub("\\.csv$", "", basename(f)))
}
save_both <- function(p, name, width, height) {
  ggplot2::ggsave(file.path(out$figures, paste0(name, ".pdf")), p, width = width, height = height,
                  device = grDevices::cairo_pdf)   # cairo keeps the en dash and ×
  ggplot2::ggsave(file.path(out$figures, paste0(name, ".png")), p, width = width, height = height, dpi = 300)
  say("wrote %s.{pdf,png}", file.path(out$figures, name))
}

# ── Fig. 3 ───────────────────────────────────────────────────────────────────
t3 <- need("tables/TableSX_mean_models_main_effects.csv", "tables/mean_models_response_sd.csv",
           "tables/fig3_pred_hmi_iucn.csv", "tables/fig3_pred_hmi_biome.csv",
           "tables/TableSY_hmi_slopes_iucn.csv", "tables/TableSY_hmi_slopes_biome.csv",
           "tables/TableSY_wald_interactions.csv")
if (!is.null(t3)) {
  rows <- fig3a_rows(t3$TableSX_mean_models_main_effects, t3$mean_models_response_sd, CFG$biome_order,
                     ref = CFG$ref)
  panels <- fig3_panels(t3$TableSY_wald_interactions)   # main analysis only (user, 2026-11-11)
  readr::write_csv(panels, file.path(out$figures, "fig3_panels.csv"))
  say("Fig. 3 panels (supported interactions): %s",
      if (nrow(panels)) paste(panels$factor, panels$response, collapse = "; ") else "none")
  p3 <- fig3(rows, t3$fig3_pred_hmi_iucn, t3$fig3_pred_hmi_biome, t3$TableSY_hmi_slopes_iucn,
             t3$TableSY_hmi_slopes_biome, t3$TableSY_wald_interactions, CFG$biome_order, panels)
  save_both(p3, "fig3_models", 13.5, fig3_height(nrow(panels)))
}

# ── Fig. 4A and the S figure ─────────────────────────────────────────────────
t4 <- need("tables/fig4a_change_summary.csv", "tables/fig4a_boot_replicates.csv")
if (!is.null(t4)) {
  p4 <- fig4a(t4$fig4a_change_summary, t4$fig4a_boot_replicates, CFG$biome_order,
              pooled_label = CFG$pooled_label)
  save_both(p4, "fig4a_trends", 10, 10.7 / 3.5 + 0.3)   # the old Fig. 4 panel A size
}
ts <- need("tables/TableSV_slope_correlation.csv")
trends_file <- file.path(CFG$paths$analysis_dir, "pa_trends.csv")
if (!is.null(ts) && file.exists(trends_file)) {
  pa <- as.data.frame(readr::read_csv(trends_file, col_types = readr::cols(WDPA_PID = "c", .default = readr::col_guess()),
                                      show_col_types = FALSE, progress = FALSE))
  save_both(fig_slope_correlation(pa, ts$TableSV_slope_correlation, CFG$biome_order, CFG$pooled_label),
            "figS_slope_correlation", 10, 5.5)
}

# ── Diagnostics (D4) ─────────────────────────────────────────────────────────
tm <- need("diagnostics/mean_models_moran_bands.csv")
if (!is.null(tm)) {
  f <- file.path(out$diagnostics, "fig_moran_correlogram_mean_models.png")
  ggplot2::ggsave(f, fig_moran_correlogram(tm$mean_models_moran_bands), width = 9, height = 6, dpi = 150)
  say("wrote %s", f)
}
for (r in c("mean_models_residuals", "change_models_residuals")) {
  tr <- need(paste0("diagnostics/", r, ".csv"))
  if (is.null(tr)) next
  res <- tr[[1]]
  for (key in unique(paste(res$model, res$response, sep = "\r"))) {
    x <- res[paste(res$model, res$response, sep = "\r") == key, ]
    if (grepl("non-spatial", x$model[1])) next   # spatial fits only (change model A is spatial)
    name <- sprintf("fig_residuals_%s_%s.png", sub("_residuals$", "", r), x$response[1])
    ggplot2::ggsave(file.path(out$diagnostics, name),
                    fig_residual_diagnostics(x, sprintf("%s: %s, %s (normalised residuals)", x$response[1],
                                                        sub("_models_residuals", " model", r), x$model[1]),
                                             CFG$biome_order),
                    width = 9, height = 7, dpi = 150)
    say("wrote %s", file.path(out$diagnostics, name))
  }
}

if (is.null(arg("--out"))) {
  writeLines(utils::capture.output(utils::sessionInfo()), CFG$paths$session_info)
  say("wrote %s", CFG$paths$session_info)
}
