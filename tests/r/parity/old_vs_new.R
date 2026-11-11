# Old vs new: the collaborator's results (old pipeline, PA_New_Analysis/) beside the v2 results.
# Writes data/parity/old_vs_new.md. Run from the repo root (after 05 and 07; the model sections
# need 06 and 08 and say "pending" until their tables exist):
#   Rscript tests/r/parity/old_vs_new.R
# Part 1: PAs in both datasets — PA means, slopes, standardised change, trend classes.
# Part 2: headline results side by side, with the method changes that touch each.
source(file.path("tests", "r", "helper_load.R"))
stopifnot(legacy_available())
out_file <- file.path(CFG$paths$parity_dir, "old_vs_new.md")
read <- function(path) {
  pid <- "WDPA_PID" %in% names(readr::read_csv(path, n_max = 0, show_col_types = FALSE))
  ct <- if (pid) readr::cols(WDPA_PID = "c", .default = readr::col_guess()) else readr::cols(.default = readr::col_guess())
  as.data.frame(readr::read_csv(path, col_types = ct, guess_max = 1e5, show_col_types = FALSE, progress = FALSE))
}
tab <- function(name) file.path(CFG$paths$tables, name)
L <- c("# Old vs new results", "",
       sprintf("Generated %s by `tests/r/parity/old_vs_new.R`. Old: the collaborator's run on `wdpa_df_ndvi_trends.csv` (PA_New_Analysis/outputs). New: v2 pipeline + R scripts 05–08.", format(Sys.time(), "%Y-%m-%d %H:%M UTC")), "")
md_table <- function(d, digits = 3) {
  d[] <- lapply(d, function(x) {
    if (!is.numeric(x)) return(x)
    if (all(is.na(x) | x == round(x))) return(format(x, big.mark = ",", trim = TRUE))
    ifelse(!is.na(x) & x != 0 & abs(x) < 1e-3, formatC(x, digits = digits - 1, format = "e"),
           formatC(x, digits = digits, format = "fg", flag = "#"))
  })
  c(paste("|", paste(names(d), collapse = " | "), "|"), paste("|", paste(rep("---", ncol(d)), collapse = " | "), "|"),
    apply(d, 1, function(r) paste("|", paste(r, collapse = " | "), "|")))
}

# ── Part 1: PAs in both datasets ─────────────────────────────────────────────
old <- legacy_pa()
new <- merge(read(file.path(CFG$paths$analysis_dir, "pa_level.csv"))[c("WDPA_PID", "BIOME_NAME", "edge_extent_mean", "edge_intensity_mean")],
             read(file.path(CFG$paths$analysis_dir, "pa_trends.csv"))[c("WDPA_PID", "slope_extent", "slope_intensity",
                                                                         "change_extent_z", "change_intensity_z",
                                                                         "trend_extent", "trend_intensity")], by = "WDPA_PID")
both <- merge(old, new, by = "WDPA_PID", suffixes = c("_old", "_new"))
L <- c(L, "## 1. PAs in both datasets", "",
       sprintf("Old analysis PAs: %s; new: %s; in both: %s (old only: %s, new only: %s). Biome agrees for %.1f%% of the shared PAs.",
               format(nrow(old), big.mark = ","), format(nrow(new), big.mark = ","), format(nrow(both), big.mark = ","),
               format(sum(!old$WDPA_PID %in% new$WDPA_PID), big.mark = ","), format(sum(!new$WDPA_PID %in% old$WDPA_PID), big.mark = ","),
               100 * mean(both$BIOME_NAME_old == both$BIOME_NAME_new)), "")
rows <- do.call(rbind, lapply(c("edge_extent_mean", "edge_intensity_mean", "slope_extent", "slope_intensity",
                                "change_extent_z", "change_intensity_z"), function(v) {
  a <- both[[paste0(v, "_old")]]; b <- both[[paste0(v, "_new")]]; ok <- !is.na(a) & !is.na(b)
  data.frame(quantity = v, n = sum(ok), pearson_r = stats::cor(a[ok], b[ok]),
             spearman_rho = stats::cor(a[ok], b[ok], method = "spearman"),
             mean_old = mean(a[ok]), mean_new = mean(b[ok]), sd_old = stats::sd(a[ok]), sd_new = stats::sd(b[ok]))
}))
L <- c(L, md_table(rows), "")
cls <- do.call(rbind, lapply(c("extent", "intensity"), function(m) {
  a <- both[[paste0("trend_", m, "_old")]]; b <- both[[paste0("trend_", m, "_new")]]; ok <- !is.na(a) & !is.na(b)
  data.frame(metric = m, n = sum(ok), pct_same_class = 100 * mean(a[ok] == b[ok]),
             kappa = cohen_kappa(a[ok], b[ok], CFG$trend$levels),
             pct_increase_old = 100 * mean(a[ok] == "increase"), pct_increase_new = 100 * mean(b[ok] == "increase"),
             pct_decrease_old = 100 * mean(a[ok] == "decrease"), pct_decrease_new = 100 * mean(b[ok] == "decrease"))
}))
L <- c(L, "Trend classes of the shared PAs (old: BH over the old 4,025 PAs; new: BH over the 4,396):", "", md_table(cls), "",
       "Method changes behind these differences: new transects and PA selection (pipeline v2: other transects, MODIS sampling on the native grid, more PAs), the sea / lake / coverage rules, the standardised change scaled by the SD over the new analysis set.", "")

# ── Part 2: headline results ─────────────────────────────────────────────────
L <- c(L, "## 2. Headline results", "")
# Biome mean change (Table SV)
sv_old <- legacy_table("TableSV_biome_trend_summary.csv")
if (file.exists(tab("TableSV_biome_trend_summary.csv"))) {
  sv_new <- read(tab("TableSV_biome_trend_summary.csv"))
  sv_old$BIOME_NAME[sv_old$BIOME_NAME == "All biomes"] <- CFG$pooled_label
  x <- merge(sv_old[c("metric", "BIOME_NAME", "n_PAs", "change_mean", "change_lo", "change_hi", "pct_increasing", "pct_decreasing")],
             sv_new[c("metric", "BIOME_NAME", "n_PAs", "change_mean", "change_lo", "change_hi", "pct_increase", "pct_decrease")],
             by = c("metric", "BIOME_NAME"), suffixes = c("_old", "_new"), sort = FALSE)
  L <- c(L, "### Mean standardised change by biome (Table SV; Fig. 4A)", "", md_table(x), "",
         "Changes: the SD for standardising is over the new analysis set (old: the 3,914-PA model set); PA set rules.", "")
}
# Main-effects continuous coefficients and the interaction tests (06)
if (file.exists(tab("TableSX_mean_models_main_effects.csv"))) {
  cx_old <- legacy_table("model_coef_additive_gls.csv"); cx_old$term <- legacy_rename_terms(cx_old$term)
  cx_new <- read(tab("TableSX_mean_models_main_effects.csv"))
  keep <- c(MODEL_COVARIATES, MODEL_HMI)
  x <- merge(cx_old[cx_old$term %in% keep, c("response", "term", "estimate", "conf.low", "conf.high", "p.value")],
             cx_new[cx_new$term %in% keep, c("response", "term", "estimate", "conf.low", "conf.high", "p.value")],
             by = c("response", "term"), suffixes = c("_old", "_new"), sort = FALSE)
  L <- c(L, "### Mean models, main effects: continuous predictors (Fig. 3A, Table SX)", "", md_table(x), "",
         "Categorical rows are not comparable directly: the old Fig. 3A showed emmeans contrasts against non-strict / Boreal; the new one shows the coefficients of a fit with non-strict / Boreal as reference (D1), the same quantities.", "")
  w_old <- legacy_table("model_tests_wald_gls.csv"); w_new <- read(tab("TableSY_wald_interactions.csv"))
  x <- merge(w_old[c("hypothesis", "response", "numDF", "denDF", "F.value", "p.value")],
             w_new[c("hypothesis", "response", "numDF", "denDF", "F.value", "p.value")],
             by = c("hypothesis", "response"), suffixes = c("_old", "_new"), sort = FALSE)
  L <- c(L, "### HMI interactions (Wald F; Fig. 3B–D panels)", "", md_table(x), "")
} else L <- c(L, "### Mean models (06): pending", "")
# Change models (08): Table SZ, model A
if (file.exists(tab("TableSZ_change_models.csv"))) {
  sz_old <- legacy_table("TableSZ_trend_drivers_long.csv"); sz_new <- read(tab("TableSZ_change_models.csv"))
  a <- CHANGE_MODEL_LABELS[["A"]]
  x <- merge(sz_old[sz_old$model == a, c("response", "term_label", "estimate", "conf.low", "conf.high", "p.value")],
             sz_new[sz_new$model == a, c("response", "term_label", "estimate", "conf.low", "conf.high", "p.value")],
             by = c("response", "term_label"), suffixes = c("_old", "_new"), sort = FALSE)
  L <- c(L, "### Change models, model A (Table SZ)", "", md_table(x), "",
         "Same reference levels as the old contrasts (non-strict / Boreal), now fitted directly (same model, reparametrised).", "")
} else L <- c(L, "### Change models (08): pending", "")

writeLines(L, out_file)
cat("wrote", out_file, "\n")
