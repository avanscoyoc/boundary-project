# Figure builders (10_figures.R). They take TABLES (data frames as written by scripts 06–08) and
# return ggplot/patchwork objects; nothing is fitted or read here (A7).
#
# Ported from PA_New_Analysis/scripts/00_setup.R (58-86: colours, theme), 06_fig3_model.R
# (Fig. 3) and 07_fig4_trend.R (1-230: Fig. 4A). Fig. 4B (case studies) is deferred (extras/).
# Changes from the old scripts (layout only, same content):
#   - positions that the old code hard-coded (Fig. 3A separators, Fig. 4A separator) are
#     computed from the rows present;
#   - the font family is set explicitly (Helvetica → Liberation Sans in the container), so PNGs
#     match the PDFs (Phase 1 found DejaVu Sans clipping the Fig. 3 legend).

PLOT_FAMILY <- "Helvetica"

BIOME_COLORS <- c(
  "Grassland & Shrubland" = "#C4A35A",
  "Tropical Forest"       = "#2E8B57",
  "Temperate Forest"      = "#5B8DB8",
  "Boreal Forest"         = "#7B68AA",
  "Desert"                = "#8C5A2B",
  "Tundra"                = "#D6568F"
)
RESPONSE_LABELS <- c(extent = "Edge extent", intensity = "Edge intensity")
FIG3_COLORS <- c("Edge extent" = "#cf6e25", "Edge intensity" = "#179c8c")
FIG3_SHAPES <- c("Edge extent" = 21, "Edge intensity" = 24)
FIG4_COLORS <- c("Edge extent" = "#d95f02", "Edge intensity" = "#00796b")
IUCN_LEVELS <- c("strict", "non-strict", "unknown")
IUCN_LABELS <- c("Strict", "Non-strict", "Unknown")
IUCN_COLORS <- c("Strict" = "#99000d", "Non-strict" = "#e5736b", "Unknown" = "#8c8c8c")
IUCN_LTYPES <- c("Strict" = "solid", "Non-strict" = "dashed", "Unknown" = "dotted")
TERM_LABELS <- c(STATUS_YR_z = "Establishment year", log_area_z = "PA area (log)",
                 gHM_mean_z = "Human modification (HMI)", elevation_mean_z = "Elevation",
                 terrain_slope_z = "Slope", water_extent_pct_z = "Water extent")

theme_pa <- function(base_size = 13, base_family = PLOT_FAMILY) {
  ggplot2::theme_classic(base_size = base_size, base_family = base_family) +
    ggplot2::theme(
      panel.grid.major = ggplot2::element_line(color = "grey92"),
      legend.position  = "bottom",
      legend.box       = "horizontal",
      strip.background = ggplot2::element_rect(fill = "grey92", color = NA),
      strip.text       = ggplot2::element_text(face = "bold"),
      plot.caption     = ggplot2::element_text(size = 8, color = "grey50"),
      plot.title       = ggplot2::element_text(face = "bold", size = 12)
    )
}

fmt_p <- function(p) ifelse(p < 0.001, "p < 0.001", sprintf("p = %.3f", p))
p_stars <- function(p, alpha = 0.05) {
  as.character(cut(p, c(-Inf, 0.001, 0.01, alpha, Inf), labels = c("***", "**", "*", ""), right = FALSE))
}
.wald_label <- function(wald, resp, pattern) {
  r <- wald[wald$response == resp & grepl(pattern, wald$term), ]
  stopifnot(nrow(r) == 1)
  sprintf("F(%d, %d) = %.2f, %s", r$numDF, r$denDF, r$F.value, fmt_p(r$p.value))
}

#' Rows of Fig. 3A: additive-model effects in response-SD units (06:77-135).
#'
#' Continuous terms from `coef` (coef_table of the additive models). Categorical rows: from
#' `contrasts` (ref_contrasts tables: each level vs a reference) if given, otherwise the IUCN_group
#' and BIOME_NAME coefficients of `coef` (each level vs the model's reference level, which must
#' then be given as `ref` = list(iucn, biome)). Estimates and CIs are divided by the response SD
#' (`response_sd`: response, sd_response). `p_col` = the p-value column of the contrasts.
fig3a_rows <- function(coef, response_sd, biome_order, contrasts = NULL, ref = NULL,
                       p_col = "p.value") {
  cont <- coef[coef$term %in% names(TERM_LABELS), ]
  cont <- data.frame(response = cont$response, term_label = unname(TERM_LABELS[cont$term]),
                     estimate = cont$estimate, conf.low = cont$conf.low,
                     conf.high = cont$conf.high, p = cont$p.value, stringsAsFactors = FALSE)
  if (!is.null(contrasts)) {
    cat_rows <- data.frame(response = contrasts$response, factor = contrasts$factor,
                           level = contrasts$level, reference = contrasts$reference,
                           estimate = contrasts$estimate, conf.low = contrasts$lower.CL,
                           conf.high = contrasts$upper.CL, p = contrasts[[p_col]],
                           stringsAsFactors = FALSE)
  } else {
    stopifnot(!is.null(ref))
    k <- grepl("^(IUCN_group|BIOME_NAME)", coef$term) & !grepl(":", coef$term)
    cc <- coef[k, ]
    fac <- ifelse(startsWith(cc$term, "IUCN_group"), "IUCN_group", "BIOME_NAME")
    cat_rows <- data.frame(response = cc$response, factor = fac,
                           level = substring(cc$term, nchar(fac) + 1),
                           reference = ifelse(fac == "IUCN_group", ref$iucn, ref$biome),
                           estimate = cc$estimate, conf.low = cc$conf.low,
                           conf.high = cc$conf.high, p = cc$p.value, stringsAsFactors = FALSE)
  }
  ref_iucn  <- unique(cat_rows$reference[cat_rows$factor == "IUCN_group"])
  ref_biome <- unique(cat_rows$reference[cat_rows$factor == "BIOME_NAME"])
  stopifnot(length(ref_iucn) == 1, length(ref_biome) == 1)
  cat_rows$term_label <- ifelse(cat_rows$factor == "IUCN_group", paste0("IUCN: ", cat_rows$level),
                                paste0("Biome: ", tolower(cat_rows$level)))
  rows <- rbind(cont, cat_rows[, names(cont)])
  sdv <- stats::setNames(response_sd$sd_response, response_sd$response)[rows$response]
  stopifnot(!anyNA(sdv))
  rows$est <- rows$estimate / sdv; rows$lo <- rows$conf.low / sdv; rows$hi <- rows$conf.high / sdv
  cont_order <- c("Human modification (HMI)", "Water extent", "Slope", "Elevation",
                  "PA area (log)", "Establishment year")
  iucn_order <- paste0("IUCN: ", setdiff(IUCN_LEVELS, ref_iucn))
  biome_rows <- paste0("Biome: ", tolower(setdiff(biome_order, ref_biome)))
  order <- c(cont_order, iucn_order, biome_rows)
  stopifnot(all(rows$term_label %in% order))
  rows$term_label <- factor(rows$term_label, levels = rev(order))
  attr(rows, "n_groups") <- c(cont = length(cont_order), iucn = length(iucn_order),
                              biome = length(biome_rows))
  attr(rows, "ref") <- list(iucn = ref_iucn, biome = ref_biome)
  rows
}

#' Fig. 3 (06): A = additive-model effects (fig3a_rows()); B = predicted extent across HMI by
#' IUCN group; C, D = predicted extent / intensity across HMI by biome. B–D carry the Wald F of
#' the interaction; stars = Holm p of each group's HMI slope. `pred_iucn` NULL drops panel B.
fig3 <- function(rows_a, pred_iucn, pred_biome, slopes_iucn, slopes_biome, wald, biome_order,
                 alpha = 0.05, iucn_response = "extent") {
  gg <- ggplot2::ggplot; aes <- ggplot2::aes
  theme_fig3 <- ggplot2::theme(axis.title = ggplot2::element_text(size = 16),
                               axis.text = ggplot2::element_text(size = 14),
                               legend.text = ggplot2::element_text(size = 12))
  annot <- 3.7

  # ── A ──
  ra <- rows_a
  ra$model <- factor(RESPONSE_LABELS[ra$response], levels = RESPONSE_LABELS)
  ra$fillkey <- ifelse(ra$p < alpha, as.character(ra$model), "ns")
  ng <- attr(rows_a, "n_groups")
  dodge <- ggplot2::position_dodge(width = 0.6)
  p_a <- gg(ra, aes(x = est, y = term_label, color = model, shape = model, fill = fillkey, group = model)) +
    ggplot2::geom_vline(xintercept = 0, linetype = "dashed", color = "grey60", linewidth = 0.7) +
    ggplot2::geom_errorbar(aes(xmin = lo, xmax = hi), orientation = "y", width = 0.25,
                           linewidth = 0.7, position = dodge, show.legend = FALSE) +
    ggplot2::geom_point(size = 3.5, stroke = 0.9, position = dodge) +
    ggplot2::geom_hline(yintercept = c(ng[["biome"]] + 0.5, ng[["biome"]] + ng[["iucn"]] + 0.5),
                        color = "grey80", linewidth = 0.5) +
    ggplot2::scale_color_manual(values = FIG3_COLORS, name = NULL) +
    ggplot2::scale_fill_manual(values = c(FIG3_COLORS, ns = "white"), guide = "none") +
    ggplot2::scale_shape_manual(values = FIG3_SHAPES, name = NULL) +
    ggplot2::guides(color = ggplot2::guide_legend(override.aes = list(fill = unname(FIG3_COLORS)))) +
    ggplot2::labs(x = "Standardized coefficient (± 95% CI)", y = NULL) +
    theme_pa() +
    ggplot2::theme(panel.grid.major.x = ggplot2::element_line(color = "grey92"),
                   legend.position = "bottom") +
    theme_fig3

  # ── B ──
  p_b <- NULL
  if (!is.null(pred_iucn)) {
    pe <- pred_iucn[pred_iucn$response == iucn_response, ]
    pe$group <- factor(pe$IUCN_group, levels = IUCN_LEVELS, labels = IUCN_LABELS)
    st <- slopes_iucn[slopes_iucn$response == iucn_response, ]
    st$group <- factor(st$IUCN_group, levels = IUCN_LEVELS, labels = IUCN_LABELS)
    ends <- do.call(rbind, lapply(split(pe, pe$group, drop = TRUE), function(x) x[which.max(x$gHM_mean), ]))
    ends$star <- p_stars(st$p.value, alpha)[match(ends$group, st$group)]
    xr <- range(pe$gHM_mean)
    p_b <- gg(pe, aes(x = gHM_mean, y = predicted, color = group, fill = group, linetype = group)) +
      ggplot2::geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.12, color = NA, show.legend = FALSE) +
      ggplot2::geom_line(linewidth = 1) +
      ggplot2::geom_text(data = ends, aes(label = trimws(paste(group, star))), hjust = -0.08,
                         vjust = 0.5, size = 3.6, family = PLOT_FAMILY, show.legend = FALSE) +
      ggplot2::annotate("text", x = xr[1] + 0.12 * diff(xr), y = Inf, hjust = 0, vjust = 1.5,
                        size = annot, family = PLOT_FAMILY,
                        label = paste0("HMI × IUCN\n", .wald_label(wald, iucn_response, "IUCN"))) +
      ggplot2::scale_color_manual(values = IUCN_COLORS, name = NULL) +
      ggplot2::scale_fill_manual(values = IUCN_COLORS, guide = "none") +
      ggplot2::scale_linetype_manual(values = IUCN_LTYPES, name = NULL) +
      ggplot2::scale_x_continuous(breaks = seq(0, 1, 0.2), expand = ggplot2::expansion(mult = c(0.02, 0.42))) +
      ggplot2::scale_y_continuous(labels = scales::label_number(accuracy = 0.01),
                                  expand = ggplot2::expansion(mult = c(0.05, 0.22))) +
      ggplot2::labs(x = "Human modification index", y = RESPONSE_LABELS[[iucn_response]]) +
      theme_pa() + ggplot2::theme(legend.position = "none") + theme_fig3
  }

  # ── C, D ──
  n_pa <- slopes_biome[!duplicated(slopes_biome$BIOME_NAME), c("BIOME_NAME", "n_PA")]
  bo <- biome_order[biome_order %in% pred_biome$BIOME_NAME]
  biome_lab <- stats::setNames(paste0(bo, " (n = ", format(n_pa$n_PA[match(bo, n_pa$BIOME_NAME)],
                                                           big.mark = ",", trim = TRUE), ")"), bo)
  pb <- pred_biome
  pb$biome <- factor(pb$BIOME_NAME, levels = bo)
  xr_all <- range(pb$gHM_mean)
  panel <- function(resp) {
    d <- pb[pb$response == resp, ]
    st <- slopes_biome[slopes_biome$response == resp, ]
    ends <- do.call(rbind, lapply(split(d, d$biome, drop = TRUE), function(x) x[which.max(x$gHM_mean), ]))
    ends$star <- p_stars(st$p.value, alpha)[match(as.character(ends$biome), st$BIOME_NAME)]
    gg(d, aes(x = gHM_mean, y = predicted, color = biome, fill = biome)) +
      ggplot2::geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.10, color = NA, show.legend = FALSE) +
      ggplot2::geom_line(linewidth = 1) +
      ggplot2::geom_text(data = ends, aes(label = star), hjust = -0.25, vjust = 0.75, size = 5,
                         family = PLOT_FAMILY, show.legend = FALSE) +
      ggplot2::annotate("text", x = xr_all[1] + 0.12 * diff(xr_all), y = Inf, hjust = 0, vjust = 1.5,
                        size = annot, family = PLOT_FAMILY,
                        label = paste0("HMI × biome\n", .wald_label(wald, resp, "BIOME"))) +
      ggplot2::scale_color_manual(values = BIOME_COLORS, labels = biome_lab, name = NULL) +
      ggplot2::scale_fill_manual(values = BIOME_COLORS, guide = "none") +
      ggplot2::scale_x_continuous(breaks = seq(0, 1, 0.2), expand = ggplot2::expansion(mult = c(0.02, 0.18))) +
      ggplot2::scale_y_continuous(labels = scales::label_number(accuracy = 0.01),
                                  expand = ggplot2::expansion(mult = c(0.05, 0.32))) +
      ggplot2::guides(color = ggplot2::guide_legend(ncol = 3, byrow = TRUE)) +
      ggplot2::labs(x = "Human modification index", y = RESPONSE_LABELS[[resp]]) +
      theme_pa() + theme_fig3
  }
  p_c <- panel("extent") + ggplot2::theme(legend.position = "none")
  p_d <- panel("intensity") +
    ggplot2::guides(color = ggplot2::guide_legend(ncol = 3, byrow = TRUE)) +
    ggplot2::theme(legend.position = "bottom", legend.margin = ggplot2::margin(0, 0, 0, 0),
                   legend.key.width = grid::unit(1.2, "lines"), legend.spacing.x = grid::unit(2, "pt"),
                   legend.text = ggplot2::element_text(size = 10))

  if (is.null(p_b)) {
    design <- "AC\nAD\nEE"; parts <- list(p_a, p_c, p_d); heights <- c(1, 1, 0.2)
  } else {
    design <- "AB\nAC\nAD\nEE"; parts <- list(p_a, p_b, p_c, p_d); heights <- c(1, 1, 1, 0.2)
  }
  patchwork::wrap_plots(c(parts, list(patchwork::guide_area()))) +
    patchwork::plot_layout(design = design, guides = "collect", widths = c(1.1, 1), heights = heights) &
    ggplot2::theme(legend.box = "horizontal", legend.box.spacing = grid::unit(0.2, "lines"),
                   legend.justification = "center")
}

#' Fig. 4A (07:25-230): standardised change by biome and for all PAs — half violin of the
#' bootstrap means (left) and mean with 95% percentile CI (right), extent and intensity.
#'
#' `summary`: metric, BIOME_NAME, n, mean_change_z, ci_lo_z, ci_hi_z (biome rows and the pooled
#' row, BIOME_NAME == pooled_label). `reps`: metric, BIOME_NAME, change_z (bootstrap means).
#' The pooled column comes first, separated by a line; `pooled_display` is its axis label.
fig4a <- function(summary, reps, biome_order, pooled_label, pooled_display = "All PAs") {
  aes <- ggplot2::aes
  lv <- c(pooled_label, biome_order)
  lv <- lv[lv %in% summary$BIOME_NAME]
  metrics <- names(FIG4_COLORS)
  s <- summary
  s$BIOME_NAME <- factor(s$BIOME_NAME, levels = lv)
  s$metric <- factor(s$metric, levels = metrics)
  s$significant <- !(s$ci_lo_z < 0 & s$ci_hi_z > 0)
  n_ext <- s[s$metric == metrics[1], ]
  lab <- stats::setNames(paste0(ifelse(as.character(n_ext$BIOME_NAME) == pooled_label, pooled_display,
                                       as.character(n_ext$BIOME_NAME)), "\n(n=", n_ext$n, ")"),
                         as.character(n_ext$BIOME_NAME))
  lab <- lab[lv]
  s$biome_label <- factor(lab[as.character(s$BIOME_NAME)], levels = lab)
  r <- reps
  r$BIOME_NAME <- factor(r$BIOME_NAME, levels = lv)
  r$metric <- factor(r$metric, levels = metrics)
  key <- paste(s$BIOME_NAME, s$metric)
  r$biome_label <- s$biome_label[match(paste(r$BIOME_NAME, r$metric), key)]
  r$significant <- s$significant[match(paste(r$BIOME_NAME, r$metric), key)]

  rng <- range(c(s$ci_lo_z, s$ci_hi_z, 0)); span <- diff(rng)
  dir_len <- 0.22 * span; dir_gap <- 0.06 * span
  x_face <- ifelse(startsWith(levels(s$biome_label), pooled_display), "bold", "plain")
  dodge <- ggplot2::position_dodge(width = 0.6)
  ggplot2::ggplot() +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.7) +
    ggplot2::geom_vline(xintercept = 1.5, color = "black", linewidth = 0.5) +
    gghalves::geom_half_violin(data = r, aes(x = biome_label, y = change_z, fill = metric, color = metric,
                                             alpha = significant, group = interaction(BIOME_NAME, metric)),
                               position = dodge, side = "l", linewidth = 0.4, trim = TRUE) +
    ggplot2::geom_pointrange(data = s, aes(x = biome_label, y = mean_change_z, ymin = ci_lo_z, ymax = ci_hi_z,
                                           color = metric, alpha = significant,
                                           group = interaction(BIOME_NAME, metric)),
                             position = dodge, size = 0.6, linewidth = 0.9, fatten = 3) +
    ggplot2::geom_point(data = s, aes(x = biome_label, y = mean_change_z, group = interaction(BIOME_NAME, metric)),
                        position = dodge, shape = 21, fill = "grey15", size = 2.6,
                        alpha = ifelse(s$significant, 1, 0.5)) +
    ggplot2::annotate("segment", x = 0.38, xend = 0.38, y = dir_gap, yend = dir_gap + dir_len,
                      colour = "grey35", linewidth = 0.5,
                      arrow = grid::arrow(length = grid::unit(0.16, "cm"), type = "closed")) +
    ggplot2::annotate("segment", x = 0.38, xend = 0.38, y = -dir_gap, yend = -dir_gap - dir_len,
                      colour = "grey35", linewidth = 0.5,
                      arrow = grid::arrow(length = grid::unit(0.16, "cm"), type = "closed")) +
    ggplot2::annotate("text", x = 0.16, y = dir_gap + dir_len / 2, label = "    Increase", angle = 90,
                      size = 3.3, colour = "grey30", family = PLOT_FAMILY) +
    ggplot2::annotate("text", x = 0.16, y = -dir_gap - dir_len / 2, label = "Decrease    ", angle = 90,
                      size = 3.3, colour = "grey30", family = PLOT_FAMILY) +
    ggplot2::scale_alpha_manual(values = c("TRUE" = 0.7, "FALSE" = 0.15), guide = "none") +
    ggplot2::scale_fill_manual(values = FIG4_COLORS, name = NULL) +
    ggplot2::scale_color_manual(values = FIG4_COLORS, name = NULL) +
    ggplot2::scale_x_discrete(labels = function(x) stringr::str_wrap(x, width = 12),
                              expand = ggplot2::expansion(add = c(1.0, 0.6))) +
    ggplot2::labs(x = NULL, y = "Change in edge metric, 2003–2025\n(among-PA SD)") +
    theme_pa(base_size = 12) +
    ggplot2::theme(axis.text.x = ggplot2::element_text(face = x_face),
                   legend.position = "inside", legend.position.inside = c(0.5, 0.2),
                   legend.direction = "horizontal", legend.background = ggplot2::element_blank(),
                   legend.key = ggplot2::element_blank(), legend.key.size = grid::unit(0.35, "cm"),
                   legend.text = ggplot2::element_text(size = 10))
}

#' S figure (D2): per-PA change in edge extent vs edge intensity (response SD per record), one
#' panel for all PAs and one per biome, with Pearson r (95% CI) from slope_correlation().
#' `pa`: BIOME_NAME, change_extent_z, change_intensity_z. `cor_tab`: slope_correlation() output.
#' (Pearson r is the same on slopes and on standardised change: a linear rescaling.)
fig_slope_correlation <- function(pa, cor_tab, biome_order, pooled_label = "All PAs") {
  aes <- ggplot2::aes
  ok <- !is.na(pa$change_extent_z) & !is.na(pa$change_intensity_z)
  d <- pa[ok, c("BIOME_NAME", "change_extent_z", "change_intensity_z")]
  d <- rbind(transform(d, panel = pooled_label), transform(d, panel = BIOME_NAME))
  lv <- c(pooled_label, biome_order[biome_order %in% d$BIOME_NAME])
  d$panel <- factor(d$panel, levels = lv)
  lab <- cor_tab[cor_tab$stratum %in% lv, ]
  lab$panel <- factor(lab$stratum, levels = lv)
  lab$label <- sprintf("r = %.2f (%.2f, %.2f)\nn = %s", lab$r_pearson, lab$r_lo, lab$r_hi,
                       format(lab$n, big.mark = ",", trim = TRUE))
  d$col <- ifelse(d$panel == pooled_label, "grey30", BIOME_COLORS[as.character(d$BIOME_NAME)])
  ggplot2::ggplot(d, aes(change_extent_z, change_intensity_z)) +
    ggplot2::geom_hline(yintercept = 0, colour = "grey70", linewidth = 0.3) +
    ggplot2::geom_vline(xintercept = 0, colour = "grey70", linewidth = 0.3) +
    ggplot2::geom_point(aes(colour = col), alpha = 0.25, size = 0.6, shape = 16) +
    ggplot2::geom_smooth(method = "lm", formula = y ~ x, se = FALSE, colour = "black", linewidth = 0.6) +
    ggplot2::geom_text(data = lab, aes(x = -Inf, y = Inf, label = label), hjust = -0.05, vjust = 1.2,
                       size = 3, family = PLOT_FAMILY, inherit.aes = FALSE) +
    ggplot2::scale_colour_identity() +
    ggplot2::facet_wrap(~ panel, ncol = 4) +
    ggplot2::labs(x = "Change in edge extent, 2003–2025 (among-PA SD)",
                  y = "Change in edge intensity, 2003–2025 (among-PA SD)") +
    theme_pa(base_size = 11)
}

#' Diagnostic: Moran's I of model residuals by distance band (03:494-509). `bands`:
#' moran_residuals()$bands rows (model, mid_km, I, p), several models stacked.
fig_moran_correlogram <- function(bands, alpha = 0.05) {
  aes <- ggplot2::aes
  bands$sig <- !is.na(bands$p) & bands$p < alpha
  ggplot2::ggplot(bands[!is.na(bands$I), ], aes(mid_km, I)) +
    ggplot2::geom_hline(yintercept = 0, linetype = 2) +
    ggplot2::geom_line(colour = "grey40") +
    ggplot2::geom_point(aes(shape = sig), size = 2) +
    ggplot2::scale_x_log10() +
    ggplot2::scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 1), name = paste("p <", alpha)) +
    ggplot2::facet_wrap(~ model) +
    ggplot2::labs(x = "Distance band midpoint (km, log scale)", y = "Moran's I of residuals",
                  title = "Residual spatial correlogram: non-spatial vs spatial (normalised)") +
    theme_pa(base_size = 10)
}

#' Diagnostic: residual plots of a fit (03:272-296; loess instead of the old gam smoother). `res`: fitted, resid (normalised residuals),
#' log_area_z, BIOME_NAME — written as a table by the model script.
fig_residual_diagnostics <- function(res, title, biome_order) {
  aes <- ggplot2::aes
  res$BIOME_NAME <- factor(res$BIOME_NAME, levels = biome_order[biome_order %in% res$BIOME_NAME])
  # The old plots used a gam smoother (mgcv), which is not in the locked R environment; loess
  # shows the same thing for a diagnostic (unreported, D4).
  sm <- ggplot2::geom_smooth(method = "loess", formula = y ~ x, se = FALSE,
                             colour = "red", linewidth = 0.6)
  th <- theme_pa(base_size = 10)
  pA <- ggplot2::ggplot(res, aes(sample = resid)) + ggplot2::stat_qq(alpha = 0.3, size = 0.6) +
    ggplot2::stat_qq_line(colour = "red") +
    ggplot2::labs(title = "A. Normal Q-Q", x = "Theoretical quantiles", y = "Normalised residuals") + th
  pB <- ggplot2::ggplot(res, aes(fitted, resid)) + ggplot2::geom_point(alpha = 0.25, size = 0.6) +
    ggplot2::geom_hline(yintercept = 0, linetype = 2) + sm +
    ggplot2::labs(title = "B. Residuals vs fitted", x = "Fitted", y = "Normalised residuals") + th
  pC <- ggplot2::ggplot(res, aes(log_area_z, abs(resid))) + ggplot2::geom_point(alpha = 0.25, size = 0.6) + sm +
    ggplot2::labs(title = "C. |Residual| vs PA area", x = "log PA area (z)", y = "|Normalised residual|") + th
  pD <- ggplot2::ggplot(res, aes(BIOME_NAME, resid)) +
    ggplot2::geom_boxplot(outlier.size = 0.5, outlier.alpha = 0.4) +
    ggplot2::geom_hline(yintercept = 0, linetype = 2) +
    ggplot2::labs(title = "D. Residuals by biome", x = NULL, y = "Normalised residuals") + th +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 30, hjust = 1))
  patchwork::wrap_plots(pA, pB, pC, pD, ncol = 2) + patchwork::plot_annotation(title = title)
}

#' Residual table for fig_residual_diagnostics() (written by the model scripts, so the figure
#' needs no fitted model): normalised residuals and fitted values of `m`, with log area and biome.
residual_table <- function(m, data, label) {
  data.frame(response = label, WDPA_PID = data$WDPA_PID, fitted = as.numeric(stats::fitted(m)),
             resid = as.numeric(stats::resid(m, type = "normalized")), log_area_z = data$log_area_z,
             BIOME_NAME = as.character(data$BIOME_NAME), stringsAsFactors = FALSE)
}
