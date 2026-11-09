# 06_models_mean.R: models of mean edge extent and intensity (Stats ¶2–3).
#
# For each response (the PA mean of the yearly metric), spatial GLS with exponential correlation
# in great-circle distance (corHaversine, range optimised from config spatial$range_start_km,
# REML):
#   main-effects model   → Fig. 3A and Table SX;
#   interaction model    (+ HMI × IUCN, HMI × biome) → Table SY: coefficients, marginal Wald F
#                        tests of the two interactions, HMI slopes per IUCN group and biome with
#                        their contrasts, and the predictions across HMI for Fig. 3B–D (written
#                        for both factors and both responses; 10_figures.R shows a panel only
#                        where its interaction is supported, B3);
#   non-spatial interaction model → diagnostics only (AICc, Moran's I; D4).
# Reference levels: config ref$mean (strict / Grassland & Shrubland, D1), so the categorical
# coefficients of the main-effects model are the Fig. 3A contrasts.
#
# Run from the repository root:
#   Rscript scripts/06_models_mean.R [--pilot N] [--cores K] [--out DIR]
# --pilot N  fit on N randomly chosen PAs (seed config pilot_seed); outputs go to
#            data/05_analysis/pilot_N/ unless --out is given.
# --cores K  spatial fits run in parallel, K at a time (default 2; each needs ~1.5 GB at full size).
# --out DIR  write tables to DIR/tables and DIR/diagnostics instead of results/.
# At full size (4,590 PAs) each spatial fit takes about an hour: run in the background, e.g.
#   nohup Rscript scripts/06_models_mean.R > data/05_analysis/06_models_mean.log 2>&1 &
# Fits are cached in data/05_analysis/cache by content (data, formula, settings), so a rerun
# loads them, and a change in the data or settings refits.
#
# Inputs:  data/05_analysis/pa_level.csv, data/05_analysis/scaling.csv (05)
# Outputs: <tables>/TableSX_mean_models_main_effects.csv   main-effects coefficients
#          <tables>/TableSY_mean_models_interaction.csv    interaction-model coefficients
#          <tables>/TableSY_wald_interactions.csv          H2 (HMI × IUCN), H3 (HMI × biome)
#          <tables>/TableSY_hmi_slopes_{iucn,biome}.csv and _contrasts.csv
#          <tables>/fig3_pred_hmi_{iucn,biome}.csv         predictions across HMI (Fig. 3B–D)
#          <tables>/mean_models_response_sd.csv            response SD over the PAs (Fig. 3A units)
#          <diagnostics>/mean_models_{wald_all_terms,spatial_params,fit_stats,moran_knn,
#                         moran_bands,residuals}.csv
#          <diagnostics>/mean_models_ranges.csv           estimated range of each spatial fit (09
#                                                         fixes the sensitivity refits at these)

source(file.path("modules", "r", "config.R"))
for (f in setdiff(list.files(file.path("modules", "r"), "\\.R$", full.names = TRUE),
                  file.path("modules", "r", "config.R"))) {
  sys.source(f, envir = globalenv())   # corHaversine must live in globalenv()
}

args <- commandArgs(trailingOnly = TRUE)
arg <- function(name) if (any(args == name)) args[which(args == name) + 1] else NULL
pilot <- as.integer(arg("--pilot") %||% NA)
cores <- as.integer(arg("--cores") %||% 2L)
out_root <- arg("--out") %||% if (!is.na(pilot)) file.path(CFG$paths$analysis_dir, paste0("pilot_", pilot))
out <- if (is.null(out_root)) list(tables = CFG$paths$tables, diagnostics = CFG$paths$diagnostics) else
  list(tables = file.path(out_root, "tables"), diagnostics = file.path(out_root, "diagnostics"))
say <- function(...) cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"), sprintf(...)))

# ── 1. Model data ────────────────────────────────────────────────────────────
pa <- as.data.frame(readr::read_csv(file.path(CFG$paths$analysis_dir, "pa_level.csv"),
                                    col_types = readr::cols(WDPA_PID = "c", .default = readr::col_guess()),
                                    guess_max = 1e5, show_col_types = FALSE, progress = FALSE))
scaling <- as.data.frame(readr::read_csv(file.path(CFG$paths$analysis_dir, "scaling.csv"),
                                         show_col_types = FALSE, progress = FALSE))
hmi <- scaling[scaling$variable == "gHM_mean_z", ]
stopifnot(nrow(hmi) == 1)
resp_cols <- stats::setNames(paste0(CFG$responses, "_mean"), names(CFG$responses))
vars <- c("WDPA_PID", unname(resp_cols), names(CFG$covariates), "gHM_mean", "IUCN_group",
          "BIOME_NAME", "lon", "lat")
stopifnot(all(vars %in% names(pa)))
d <- pa[vars]
stopifnot(!anyNA(d))
if (!is.na(pilot)) {
  set.seed(CFG$pilot_seed)
  d <- d[sort(sample.int(nrow(d), min(pilot, nrow(d)))), ]
}
d <- set_model_references(d, CFG$ref$mean, names(CFG$iucn_groups), CFG$biome_order)
rownames(d) <- NULL
say("06: %d PAs%s; reference levels %s / %s; %d cores", nrow(d),
    if (!is.na(pilot)) sprintf(" (pilot, seed %d)", CFG$pilot_seed) else "",
    CFG$ref$mean$iucn, CFG$ref$mean$biome, cores)

# ── 2. Fits (spatial fits in parallel; each child writes its fit to the cache) ─
cs <- spatial_setup(d, value = CFG$spatial$range_start_km, nugget = CFG$spatial$nugget,
                    cache_dir = CFG$paths$cache_dir)
say("corHaversine set-up ready (start %g km)", CFG$spatial$range_start_km)
fit <- function(r, type, spatial) {
  fit_gls_cached(model_formula(resp_cols[[r]], type), d, if (spatial) cs, method = CFG$spatial$method,
                 max_iter = CFG$spatial$max_iter, cache_dir = CFG$paths$cache_dir,
                 label = sprintf("%s %s, %s", r, type, if (spatial) "spatial" else "non-spatial"))
}
jobs <- expand.grid(r = names(resp_cols), type = c("add", "full"), stringsAsFactors = FALSE)
done <- parallel::mclapply(seq_len(nrow(jobs)), function(i) {
  m <- fit(jobs$r[i], jobs$type[i], TRUE)
  sprintf("%s %s: range %.3f km, %.1f min", jobs$r[i], jobs$type[i],
          stats::coef(m$modelStruct$corStruct, unconstrained = FALSE)[["range"]],
          attr(m, "fit_minutes") %||% NA_real_)
}, mc.cores = cores, mc.preschedule = FALSE)
for (x in done) if (inherits(x, "try-error")) stop("spatial fit failed: ", x) else say("fitted: %s", x)

# ── 3. Tables ────────────────────────────────────────────────────────────────
w <- suppressWarnings(moran_weights(d, CFG$spatial$k_nn, CFG$spatial$bands_km))  # short bands may be sparse
tabs <- lapply(stats::setNames(names(resp_cols), names(resp_cols)), function(r) {
  fits <- fit_mean_models(d, resp_cols[[r]], cs, CFG$spatial$method, CFG$spatial$max_iter,
                          cache_dir = CFG$paths$cache_dir, label = r)
  say("%s: tables", r)
  tb <- mean_model_tables(fits, d, resp_cols[[r]], r, hmi$mean, hmi$sd, contrast_ref = NULL, moran = w,
                          moran_settings = list(nsim = CFG$spatial$moran_nsim, seed = CFG$spatial$moran_seed),
                          pred_by = c("IUCN_group", "BIOME_NAME"))
  rg <- function(m) stats::coef(m$modelStruct$corStruct, unconstrained = FALSE)[["range"]]
  tb$ranges <- data.frame(response = r, model = c("main effects", "interaction"),
                          range_km = c(rg(fits$add), rg(fits$full)), stringsAsFactors = FALSE)
  tb$residuals <- rbind(cbind(model = "interaction, spatial", residual_table(fits$full, d, r)),
                        cbind(model = "interaction, non-spatial", residual_table(fits$full_nosp, d, r)))
  tb
})
stack <- function(field) { x <- do.call(rbind, lapply(tabs, `[[`, field)); rownames(x) <- NULL; x }

# ── 4. Write ─────────────────────────────────────────────────────────────────
for (dir in out) dir.create(dir, recursive = TRUE, showWarnings = FALSE)
write <- function(field, dir, name) readr::write_csv(stack(field), file.path(dir, name), na = "")
write("coef_add",               out$tables, "TableSX_mean_models_main_effects.csv")
write("coef_full",              out$tables, "TableSY_mean_models_interaction.csv")
write("wald",                   out$tables, "TableSY_wald_interactions.csv")
write("slopes_iucn",            out$tables, "TableSY_hmi_slopes_iucn.csv")
write("slopes_iucn_contrasts",  out$tables, "TableSY_hmi_slopes_iucn_contrasts.csv")
write("slopes_biome",           out$tables, "TableSY_hmi_slopes_biome.csv")
write("slopes_biome_contrasts", out$tables, "TableSY_hmi_slopes_biome_contrasts.csv")
write("pred_IUCN_group",        out$tables, "fig3_pred_hmi_iucn.csv")
write("pred_BIOME_NAME",        out$tables, "fig3_pred_hmi_biome.csv")
write("response_sd",            out$tables, "mean_models_response_sd.csv")
write("wald_all",               out$diagnostics, "mean_models_wald_all_terms.csv")
write("spatial_params",         out$diagnostics, "mean_models_spatial_params.csv")
write("fit_stats",              out$diagnostics, "mean_models_fit_stats.csv")
write("moran_knn",              out$diagnostics, "mean_models_moran_knn.csv")
write("moran_bands",            out$diagnostics, "mean_models_moran_bands.csv")
write("residuals",              out$diagnostics, "mean_models_residuals.csv")
write("ranges",                 out$diagnostics, "mean_models_ranges.csv")   # fixed ranges of 09

wd <- stack("wald")
for (i in seq_len(nrow(wd))) say("%s %s: F(%d, %d) = %.2f, p = %.3g", wd$response[i], wd$hypothesis[i],
                                 wd$numDF[i], wd$denDF[i], wd$F.value[i], wd$p.value[i])
sp <- stack("spatial_params")
for (i in seq_len(nrow(sp))) say("%s interaction model: range %.2f km (%.2f–%.2f), ΔAICc (non-spatial − spatial) %.1f",
                                 sp$response[i], sp$range_km[i], sp$range_lo[i], sp$range_hi[i],
                                 sp$delta_AICc_nonspatial_minus_spatial[i])
say("wrote %s and %s", out$tables, out$diagnostics)
