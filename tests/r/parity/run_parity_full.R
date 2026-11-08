# Full parity run: the spatial GLS refitted with the range OPTIMISED (start 200 km, REML), as the
# old run did, on the old data, compared with every old model table at tolerance 1e-4
# (optimiser, BLAS, platform; the collaborator's own non-spatial GLS rerun differed by 2e-5).
#
# Slow (each optimiser step refactors an n × n matrix): run in the background from the repo root,
#   Rscript tests/r/parity/run_parity_full.R [mean] [change] [--cores 2]
# Fits are cached in data/parity/cache, so a rerun only refits what changed. Writes the rows of
# source "full_mean" / "full_change" to data/parity/parity_report.csv, the new tables to
# data/parity/full_<part>_tables.rds, and a log to data/parity/run_parity_full.log.
# Parts: "mean" (03: 4 spatial + 2 non-spatial fits); "change" (02b: per response A spatial,
# A non-spatial, B spatial precision-weighted; responses = standardised change from the step 2–3
# functions). No part given = both.

args  <- commandArgs(trailingOnly = TRUE)
cores <- if (any(args == "--cores")) as.integer(args[which(args == "--cores") + 1]) else 2L
parts <- intersect(args, c("mean", "change"))
if (!length(parts)) parts <- c("mean", "change")

local({
  root <- normalizePath(getwd())
  stopifnot(file.exists(file.path(root, "modules", "r", "config.R")))
})
source(file.path("tests", "r", "helper_load.R"))
source(file.path("tests", "r", "helper_compare.R"))
source(file.path("tests", "r", "parity", "mean_model_checks.R"))
source(file.path("tests", "r", "parity", "change_model_checks.R"))
stopifnot(legacy_available())

log_file <- file.path(CFG$paths$parity_dir, "run_parity_full.log")
dir.create(dirname(log_file), recursive = TRUE, showWarnings = FALSE)
say <- function(...) {
  msg <- sprintf("[%s] %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), sprintf(...))
  cat(msg, "\n", sep = ""); cat(msg, "\n", sep = "", file = log_file, append = TRUE)
}
say("run_parity_full.R: parts %s, %d cores", paste(parts, collapse = ","), cores)

pa <- legacy_pa(); s <- legacy_settings(pa); co <- legacy_coords(pa$WDPA_PID)
d <- legacy_model_data(pa, co, s)
cdir <- file.path(CFG$paths$parity_dir, "cache")
say("model data: %d PAs", nrow(d))

cs <- spatial_setup(d, value = CFG$spatial$range_start_km, nugget = CFG$spatial$nugget, cache_dir = cdir)
say("corHaversine set-up ready (start %g km)", CFG$spatial$range_start_km)

if ("mean" %in% parts) {
  jobs <- expand.grid(resp = names(LEGACY_RESPONSES), type = c("add", "full"), stringsAsFactors = FALSE)
  # Fit in parallel (each child writes its fit to the cache), then load them in this process.
  done <- parallel::mclapply(seq_len(nrow(jobs)), function(i) {
    r <- jobs$resp[i]; ty <- jobs$type[i]
    t0 <- Sys.time()
    m <- fit_gls_cached(model_formula(LEGACY_RESPONSES[[r]], ty), d, cs,
                        method = CFG$spatial$method, max_iter = CFG$spatial$max_iter, cache_dir = cdir)
    sprintf("%s %s: range %.4f km, %.1f min", r, ty,
            stats::coef(m$modelStruct$corStruct, unconstrained = FALSE)[["range"]],
            as.numeric(difftime(Sys.time(), t0, units = "mins")))
  }, mc.cores = cores, mc.preschedule = FALSE)
  for (x in done) say("fitted: %s", if (inherits(x, "try-error")) paste("ERROR", x) else x)

  w <- suppressWarnings(moran_weights(d, CFG$spatial$k_nn, CFG$spatial$bands_km))
  tabs <- lapply(stats::setNames(names(LEGACY_RESPONSES), names(LEGACY_RESPONSES)), function(r) {
    fits <- fit_mean_models(d, LEGACY_RESPONSES[[r]], cs, CFG$spatial$method, CFG$spatial$max_iter,
                            cache_dir = cdir, label = r)
    say("%s: building tables", r)
    mean_model_tables(fits, d, LEGACY_RESPONSES[[r]], r, s$hmi[["mean"]], s$hmi[["sd"]],
                      contrast_ref = s$ref_contrast, moran = w,
                      moran_settings = list(nsim = CFG$spatial$moran_nsim, seed = CFG$spatial$moran_seed),
                      pred_by = c(if (r == "extent") "IUCN_group", "BIOME_NAME"))
  })
  saveRDS(tabs, file.path(CFG$paths$parity_dir, "full_mean_tables.rds"))
  testthat::test_that("optimised mean models reproduce the old 03 tables", {
    check_mean_tables(tabs, what = c("full", "add", "optimised"), tol = 1e-4, source = "full_mean")
  })
}

if ("change" %in% parts) {
  # Responses: standardised change from the new trends (as the fast suite's step 3 block)
  an <- legacy_annual()
  for (m in c("extent", "intensity")) {
    tr <- ols_trends(an, paste0("edge_", m), s$min_years)
    ch <- standardised_change(tr$slope, tr$se, s$n_span, s$sd_resp[[m]])
    i <- match(d$WDPA_PID, tr$WDPA_PID)
    d[[paste0("change_", m, "_z")]] <- ch$change[i]
    d[[paste0("se_change_", m, "_z")]] <- ch$se_change[i]
  }
  stopifnot(!anyNA(d$change_extent_z), !anyNA(d$change_intensity_z))
  fit_one <- function(r) fit_change_models(d, LEGACY_CHANGE[[r]][["response"]], LEGACY_CHANGE[[r]][["se"]],
                                           cs, CFG$spatial$method, CFG$spatial$max_iter,
                                           cache_dir = cdir, label = r)
  done <- parallel::mclapply(names(LEGACY_CHANGE), function(r) {
    t0 <- Sys.time(); cm <- fit_one(r)
    rg <- function(m) stats::coef(m$modelStruct$corStruct, unconstrained = FALSE)[["range"]]
    sprintf("%s change: A range %.4f km, B range %.4f km, %.1f min", r, rg(cm$fits$A), rg(cm$fits$B),
            as.numeric(difftime(Sys.time(), t0, units = "mins")))
  }, mc.cores = cores, mc.preschedule = FALSE)
  for (x in done) say("fitted: %s", if (inherits(x, "try-error")) paste("ERROR", x) else x)

  w <- suppressWarnings(moran_weights(d, CFG$spatial$k_nn, CFG$spatial$bands_km[1:2]))
  tabs <- lapply(stats::setNames(names(LEGACY_CHANGE), names(LEGACY_CHANGE)), function(r) {
    say("%s change: building tables", r)
    change_model_tables(fit_one(r), LEGACY_CHANGE[[r]][["response"]], r, s$ref_contrast, moran = w,
                        moran_settings = list(nsim = CFG$spatial$moran_nsim, seed = CFG$spatial$moran_seed))
  })
  saveRDS(tabs, file.path(CFG$paths$parity_dir, "full_change_tables.rds"))
  testthat::test_that("optimised change models reproduce the old 02b tables", {
    check_change_tables(tabs, models = c("A", "B"), optimised = TRUE, tol = 1e-4, source = "full_change")
  })
}

rep <- parity_write_report(file.path(CFG$paths$parity_dir, "parity_report.csv"))
full <- rep[startsWith(rep$source, "full"), ]
say("parity rows (full_*): %d, failing: %d; worst rel diff %.3g", nrow(full), sum(!full$pass),
    max(full$max_rel_diff[full$kind == "rel"], na.rm = TRUE))
for (i in which(!full$pass)) say("FAIL %s / %s: %s", full$output[i], full$item[i], full$detail[i])
say("done")
