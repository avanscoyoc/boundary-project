# 08_models_change.R: models of change in edge extent and intensity (Stats ¶6).
#
# For each metric, the standardised change of each PA (07) on the same predictors as the
# main-effects mean model, spatial GLS with the same correlation structure and fitting method:
#   A  primary: unweighted (Table SZ);
#   B  precision-weighted: varFixed(~ v), v = se² + τ², τ² from A by the method of moments
#      (Table SZ, sensitivity);
#   non-spatial analogue of A → diagnostics (AICc, Moran's I; D4).
# Reference levels: config ref$change (non-strict / Boreal Forest, D1), fitted directly, so the
# categorical rows of Table SZ are the coefficients of the fit.
#
# Run from the repository root (after 07):
#   Rscript scripts/08_models_change.R [--pilot N] [--cores K] [--out DIR]
# --pilot N  N randomly chosen PAs (the same PAs as 06 --pilot N); outputs go to
#            data/05_analysis/pilot_N/ unless --out is given.
# --cores K  the two metrics are fitted in parallel when K ≥ 2 (default 2; ~1.5 GB each).
# --out DIR  write tables to DIR/tables and DIR/diagnostics instead of results/.
# At full size A and B each take about an hour per metric: run in the background, not at the
# same time as 06 (memory). The spatial set-up is the same as 06's (same PAs and settings), so
# it is loaded from the cache.
#
# Inputs:  data/05_analysis/pa_level.csv (05), data/05_analysis/pa_trends.csv (07)
# Outputs: <tables>/TableSZ_change_models.csv   long: continuous terms (per 1 SD) and categorical
#                                               contrasts vs the reference, models A and B
#          <diagnostics>/change_models_{coef,wald,contrasts,tau2,spatial_params,moran,
#                         residuals}.csv

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
read <- function(name) {
  as.data.frame(readr::read_csv(file.path(CFG$paths$analysis_dir, name), guess_max = 1e5,
                                col_types = readr::cols(WDPA_PID = "c", .default = readr::col_guess()),
                                show_col_types = FALSE, progress = FALSE))
}

# ── 1. Model data ────────────────────────────────────────────────────────────
pa <- read("pa_level.csv")
tr <- read("pa_trends.csv")
metrics <- names(CFG$responses)
resp <- lapply(stats::setNames(metrics, metrics), function(m)
  c(response = paste0("change_", m, "_z"), se = paste0("se_change_", m, "_z")))
vars <- c("WDPA_PID", names(CFG$covariates), "IUCN_group", "BIOME_NAME", "lon", "lat")
stopifnot(all(vars %in% names(pa)), identical(sort(tr$WDPA_PID), sort(pa$WDPA_PID)))
d <- pa[vars]
i <- match(d$WDPA_PID, tr$WDPA_PID)
for (col in unlist(resp)) d[[col]] <- tr[[col]][i]
stopifnot(!anyNA(d))
if (!is.na(pilot)) {   # the same rows as 06 --pilot N (same seed and row count)
  set.seed(CFG$pilot_seed)
  d <- d[sort(sample.int(nrow(d), min(pilot, nrow(d)))), ]
}
d <- set_model_references(d, CFG$ref$change, names(CFG$iucn_groups), CFG$biome_order)
rownames(d) <- NULL
say("08: %d PAs%s; reference levels %s / %s; %d cores", nrow(d),
    if (!is.na(pilot)) sprintf(" (pilot, seed %d)", CFG$pilot_seed) else "",
    CFG$ref$change$iucn, CFG$ref$change$biome, cores)

# ── 2. Fits (A, non-spatial A, B per metric; the metrics in parallel) ────────
cs <- spatial_setup(d, value = CFG$spatial$range_start_km, nugget = CFG$spatial$nugget,
                    cache_dir = CFG$paths$cache_dir)
say("corHaversine set-up ready (start %g km)", CFG$spatial$range_start_km)
fit_one <- function(m) fit_change_models(d, resp[[m]][["response"]], resp[[m]][["se"]], cs,
                                         CFG$spatial$method, CFG$spatial$max_iter,
                                         cache_dir = CFG$paths$cache_dir, label = m)
rg <- function(x) stats::coef(x$modelStruct$corStruct, unconstrained = FALSE)[["range"]]
done <- parallel::mclapply(metrics, function(m) {
  cm <- fit_one(m)
  sprintf("%s: A range %.3f km, B range %.3f km", m, rg(cm$fits$A), rg(cm$fits$B))
}, mc.cores = min(cores, length(metrics)), mc.preschedule = FALSE)
for (x in done) if (inherits(x, "try-error")) stop("change-model fit failed: ", x) else say("fitted: %s", x)

# ── 3. Tables ────────────────────────────────────────────────────────────────
w <- suppressWarnings(moran_weights(d, CFG$spatial$k_nn, CFG$spatial$bands_km[1:2]))  # kNN only
tabs <- lapply(stats::setNames(metrics, metrics), function(m) {
  cm <- fit_one(m)   # loads the cached fits
  say("%s: tables", m)
  tb <- change_model_tables(cm, resp[[m]][["response"]], m, CFG$ref$change, moran = w,
                            moran_settings = list(nsim = CFG$spatial$moran_nsim, seed = CFG$spatial$moran_seed))
  tb$sz <- table_sz(tb$coef, tb$contrasts)
  tb$residuals <- cbind(model = CHANGE_MODEL_LABELS[["A"]], residual_table(cm$fits$A, cm$data, m))
  tb
})
stack <- function(field) { x <- do.call(rbind, lapply(tabs, `[[`, field)); rownames(x) <- NULL; x }

# ── 4. Write ─────────────────────────────────────────────────────────────────
for (dir in out) dir.create(dir, recursive = TRUE, showWarnings = FALSE)
write <- function(field, dir, name) readr::write_csv(stack(field), file.path(dir, name), na = "")
write("sz",             out$tables, "TableSZ_change_models.csv")
write("coef",           out$diagnostics, "change_models_coef.csv")
write("wald",           out$diagnostics, "change_models_wald.csv")
write("contrasts",      out$diagnostics, "change_models_contrasts.csv")
write("tau",            out$diagnostics, "change_models_tau2.csv")
write("spatial_params", out$diagnostics, "change_models_spatial_params.csv")
write("moran",          out$diagnostics, "change_models_moran.csv")
write("residuals",      out$diagnostics, "change_models_residuals.csv")

sp <- stack("spatial_params")
for (i in seq_len(nrow(sp))) say("%s model A: range %.2f km (%.2f–%.2f), ΔAICc (non-spatial − spatial) %.1f, pseudo-R² %.3f",
                                 sp$response[i], sp$range_km[i], sp$range_lo[i], sp$range_hi[i],
                                 sp$delta_AICc_nonspatial_minus_spatial[i], sp$pseudoR2_spatial[i])
say("wrote %s and %s", out$tables, out$diagnostics)
