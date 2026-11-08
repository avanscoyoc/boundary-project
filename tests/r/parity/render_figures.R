# Step 7 side-by-side check: render Fig. 3 and Fig. 4A with modules/r/plots.R from the OLD saved
# tables (renamed to the new conventions), for comparison with the collaborator's re-rendered
# figures in data/parity/reference_figures/ (Phase 1). Run from the repo root:
#   Rscript tests/r/parity/render_figures.R
# Writes data/parity/figures/*_port.{pdf,png}: Fig. 3, Fig. 4A, the Moran correlogram, the residual
# diagnostics (extent, from the fast suite's cached fixed-range fit) and the slope-correlation S fig.
source(file.path("tests", "r", "helper_load.R"))
stopifnot(legacy_available())
out <- file.path(CFG$paths$parity_dir, "figures")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
s <- legacy_settings(legacy_pa())

tab <- function(name) {
  x <- legacy_table(name)
  names(x) <- legacy_rename_terms(names(x))
  for (v in intersect(c("term", "factor"), names(x))) x[[v]] <- legacy_rename_terms(x[[v]])
  x
}
# ── Fig. 3 ──
rows <- fig3a_rows(tab("model_coef_additive_gls.csv"), tab("response_sd_gls.csv"), s$biome_order,
                   contrasts = tab("model_contrasts_additive_gls.csv"))
sl_i <- tab("hmi_slopes_iucn_gls.csv"); sl_b <- tab("hmi_slopes_biome_gls.csv")
p3 <- fig3(rows, tab("pred_extent_hmi_iucn_gls.csv"),
           rbind(tab("pred_extent_hmi_biome_gls.csv"), tab("pred_intensity_hmi_biome_gls.csv")),
           sl_i[sl_i$spatial == "spatial", ], sl_b[sl_b$spatial == "spatial", ],
           tab("model_tests_wald_gls.csv"), s$biome_order)
ggplot2::ggsave(file.path(out, "fig3_model_port.pdf"), p3, width = 13.5, height = 12.5, device = grDevices::cairo_pdf)
ggplot2::ggsave(file.path(out, "fig3_model_port.png"), p3, width = 13.5, height = 12.5, dpi = 200)

# ── Fig. 4A ── (old tables already carry the *_z columns and change_z replicates)
sm <- rbind(tab("biome_slope_summary.csv"), tab("pooled_slope_summary.csv"))
sm <- data.frame(metric = sm$metric, BIOME_NAME = sm$BIOME_NAME, n = sm$n_pa,
                 mean_change_z = sm$mean_change_z, ci_lo_z = sm$ci_lo_z, ci_hi_z = sm$ci_hi_z)
reps <- rbind(tab("biome_boot_replicates.csv"), tab("pooled_boot_replicates.csv"))
p4 <- fig4a(sm, reps[, c("metric", "BIOME_NAME", "change_z")], s$biome_order, pooled_label = "All biomes")
# The old Fig. 4 page is 10 × 10.7 in with panel A 1/3.5 of the height; panel A alone here.
ggplot2::ggsave(file.path(out, "fig4a_trend_port.pdf"), p4, width = 10, height = 10.7 / 3.5 + 0.3,
                device = grDevices::cairo_pdf)
ggplot2::ggsave(file.path(out, "fig4a_trend_port.png"), p4, width = 10, height = 10.7 / 3.5 + 0.3, dpi = 200)

# ── Diagnostics and the S figure ──
mb <- tab("moran_residuals_bands_gls.csv")
ggplot2::ggsave(file.path(out, "moran_correlogram_port.png"), fig_moran_correlogram(mb),
                width = 9, height = 6, dpi = 150)
pa <- legacy_pa()
k <- s$n_span / s$sd_resp
pa$change_extent_z <- pa$slope_extent * k[["extent"]]; pa$change_intensity_z <- pa$slope_intensity * k[["intensity"]]
ct <- slope_correlation(pa, "slope_extent", "slope_intensity", "BIOME_NAME", s$biome_order, "All PAs")
ggplot2::ggsave(file.path(out, "figS_slope_correlation_port.png"),
                fig_slope_correlation(pa, ct, s$biome_order), width = 10, height = 5.5, dpi = 150)
# Residual diagnostics from the cached fixed-range full extent fit (fast parity suite), if present
d <- legacy_model_data(pa, legacy_coords(pa$WDPA_PID), s)
cdir <- file.path(CFG$paths$parity_dir, "cache")
cs <- spatial_setup(d, value = legacy_ranges()[["extent"]], fixed = TRUE, cache_dir = cdir)
m <- fit_gls_cached(model_formula("edge_extent_mean", "full"), d, cs, cache_dir = cdir)
ggplot2::ggsave(file.path(out, "diag_extent_port.png"),
                fig_residual_diagnostics(residual_table(m, d, "extent"),
                                         "extent \u2014 full model, spatial (normalised residuals)", s$biome_order),
                width = 9, height = 7, dpi = 150)
cat("written to", out, "\n")
