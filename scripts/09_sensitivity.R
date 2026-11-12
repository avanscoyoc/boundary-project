# 09_sensitivity.R: sensitivity of the results to transect and PA selection.
#
# Runs (each recomputes the PA means, trends and models from the PA-year data):
#   main         all transects, the analysis PAs (reproduces 06–08 at their ranges)
#   all_common   all transects, the PAs of the transect-subset comparison (set_transects)
#   nowater, nootherpa, noflags
#                each transect subset on the same PAs as all_common; a PA-year counts only
#                with ≥ config sensitivity$min_transects complete transects in the subset
#   established  all transects, PAs established by config sensitivity$established_by (analysed year)
#   reported     all transects, PAs with a reported establishment year
# Each run is compared with a reference: the transect subsets with all_common (same PAs, so a
# difference comes from the transects); all_common and the PA subsets with main.
#
# S1 (TableS_sensitivity_metrics): per subset and metric, against all_common: Pearson r and the
#   paired mean difference (response SDs, bootstrap CI) of the PA means and of the standardised
#   change; agreement of the trend classes (% and Cohen's kappa).
# S2 (TableS_sensitivity_models): per run, the headline results with the reference values beside
#   them: main-effects coefficients of the mean models, the Wald F of the two HMI interactions,
#   % of PAs increasing / decreasing, biome mean change (bootstrap CIs), and the change-model A
#   coefficients; same sign, same significance at 0.05.
# Spatial fits: range FIXED at the estimate of the main analysis (06, 08: *_ranges.csv). One
# optimised refit (noflags, extent, main effects) checks that shortcut (diagnostics). Standardised
# change uses the main analysis's response SDs; covariate z-scores are the main ones.
#
# Run from the repository root, after 06–08:
#   Rscript scripts/09_sensitivity.R [--pilot N] [--cores K] [--out DIR] [--no-refit]
# --pilot N   N randomly chosen analysis PAs (seed config pilot_seed); outputs go to
#             data/05_analysis/pilot_N/ unless --out is given; no refit check.
# --cores K   runs fitted in parallel, K at a time (default 2; ~1.5 GB each at full size).
# Full size: 7 runs × 6 fixed-range spatial fits (≈ 1–2 min each) plus the refit (≈ 1 h).
#
# Inputs:  data/05_analysis/pa_level.csv, pa_year.csv, scaling.csv (05);
#          results/diagnostics/mean_models_ranges.csv (06), change_models_ranges.csv (08)
# Outputs: <tables>/TableS_sensitivity_metrics.csv, TableS_sensitivity_models.csv;
#          <diagnostics>/sensitivity_runs.csv, sensitivity_main_check.csv, sensitivity_refit_check.csv

source(file.path("modules", "r", "config.R"))
for (f in setdiff(list.files(file.path("modules", "r"), "\\.R$", full.names = TRUE),
                  file.path("modules", "r", "config.R"))) {
  sys.source(f, envir = globalenv())   # corHaversine must live in globalenv()
}

args <- commandArgs(trailingOnly = TRUE)
arg <- function(name) if (any(args == name)) args[which(args == name) + 1] else NULL
pilot <- as.integer(arg("--pilot") %||% NA)
cores <- as.integer(arg("--cores") %||% 2L)
do_refit <- is.na(pilot) && !any(args == "--no-refit")
out_root <- arg("--out") %||% if (!is.na(pilot)) file.path(CFG$paths$analysis_dir, paste0("pilot_", pilot))
out <- if (is.null(out_root)) list(tables = CFG$paths$tables, diagnostics = CFG$paths$diagnostics) else
  list(tables = file.path(out_root, "tables"), diagnostics = file.path(out_root, "diagnostics"))
say <- function(...) cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"), sprintf(...)))
read <- function(path, ...) as.data.frame(readr::read_csv(path, guess_max = 1e5, show_col_types = FALSE,
                                                          progress = FALSE, ...))
ids <- readr::cols(WDPA_PID = "c", .default = readr::col_guess())

# ── 1. Inputs ────────────────────────────────────────────────────────────────
pa <- read(file.path(CFG$paths$analysis_dir, "pa_level.csv"), col_types = ids)
py <- read(file.path(CFG$paths$analysis_dir, "pa_year.csv"),
           col_types = readr::cols(WDPA_PID = "c", transects = "c", .default = readr::col_guess()))
scaling <- read(file.path(CFG$paths$analysis_dir, "scaling.csv"))
metrics <- names(CFG$responses)
labels  <- RESPONSE_LABELS[metrics]
sd_resp <- stats::setNames(vapply(metrics, function(m) scaling$sd[scaling$variable == paste0(CFG$responses[[m]], "_mean")],
                                  numeric(1)), metrics)
rng_file <- file.path(CFG$paths$diagnostics, c("mean_models_ranges.csv", "change_models_ranges.csv"))  # 06, 08
if (all(file.exists(rng_file))) {
  rng <- do.call(rbind, lapply(rng_file, read))
} else if (!is.na(pilot)) {
  say("pilot: no range tables from 06/08 yet, fixing every range at %g km", CFG$spatial$range_start_km)
  rng <- expand.grid(response = metrics, model = c("main effects", "interaction", CHANGE_MODEL_LABELS[["A"]]),
                     stringsAsFactors = FALSE)
  rng$range_km <- CFG$spatial$range_start_km
} else stop("Run 06 and 08 first: ", paste(rng_file[!file.exists(rng_file)], collapse = ", "))
range_of <- function(r, model) { x <- rng$range_km[rng$response == r & rng$model == model]; stopifnot(length(x) == 1); x }
if (!is.na(pilot)) {
  set.seed(CFG$pilot_seed)
  pa <- pa[sort(sample.int(nrow(pa), min(pilot, nrow(pa)))), ]
}

# ── 2. Runs ──────────────────────────────────────────────────────────────────
common <- pa$WDPA_PID[pa$set_transects]
run_def <- function(label, pids, version, min_tr, vs) list(label = label, pids = pids, version = version,
                                                            min_tr = min_tr, vs = vs)
mt <- CFG$sensitivity$min_transects
runs <- list(
  main        = run_def("Main analysis (all transects)", pa$WDPA_PID, "all", 0L, NA),
  all_common  = run_def("All transects, transect-subset PAs", common, "all", 0L, "main"),
  nowater     = run_def("Without water transects", common, "nowater", mt, "all_common"),
  nootherpa   = run_def("Without transects bordering other PAs", common, "nootherpa", mt, "all_common"),
  noflags     = run_def("Without water or other-PA transects", common, "noflags", mt, "all_common"),
  established = run_def(sprintf("PAs established by %d", CFG$sensitivity$established_by),
                        pa$WDPA_PID[pa$set_established], "all", 0L, "main"),
  reported    = run_def("PAs with a reported establishment year", pa$WDPA_PID[pa$set_reported], "all", 0L, "main"))
say("09: %d runs; %d PAs in the main set, %d in the transect-subset comparison%s", length(runs), nrow(pa),
    length(common), if (!is.na(pilot)) sprintf(" (pilot, seed %d)", CFG$pilot_seed) else "")

# Per-PA data of a run: PA means, OLS trends (BH over the run's PAs), standardised change.
run_data <- function(r) {
  x <- py[py$transects == r$version & py$WDPA_PID %in% r$pids & !is.na(py$n_trnst) & py$n_trnst >= r$min_tr, ]
  pm <- pa_means(x, unname(CFG$responses))
  d <- pa[match(r$pids, pa$WDPA_PID), c("WDPA_PID", names(CFG$covariates), "IUCN_group", "BIOME_NAME", "lon", "lat")]
  for (m in metrics) {
    col <- paste0(CFG$responses[[m]], "_mean")
    d[[col]] <- pm[[col]][match(d$WDPA_PID, pm$WDPA_PID)]
    tr <- ols_trends(x, CFG$responses[[m]], CFG$min_years)
    bh <- bh_classes(tr$slope, tr$p_value, CFG$trend$fdr, CFG$trend$levels)
    j <- match(d$WDPA_PID, tr$WDPA_PID)
    ch <- standardised_change(tr$slope, tr$se, CFG$n_span, sd_resp[[m]])
    d[[paste0("slope_", m)]] <- tr$slope[j]
    d[[paste0("trend_", m)]] <- bh$class[j]
    d[[paste0("change_", m, "_z")]] <- ch$change[j]
  }
  if (anyNA(d)) stop(r$label, ": PAs without both metrics for >= ", CFG$min_years, " years")
  rownames(d) <- NULL
  attr(d, "n_pa_years") <- nrow(x)
  d
}

# Headline results of one run (S2), in long format.
run_results <- function(key) {
  r <- runs[[key]]
  d <- run_data(r)
  say("%s: %d PAs, %d PA-years", key, nrow(d), attr(d, "n_pa_years"))
  fit_fixed <- function(response, type, range, ref) {
    dd <- set_model_references(d, ref, names(CFG$iucn_groups), CFG$biome_order)
    cs <- spatial_setup(dd, value = range, fixed = TRUE, cache_dir = CFG$paths$cache_dir)
    fit_gls_cached(model_formula(response, type), dd, cs, method = CFG$spatial$method,
                   max_iter = CFG$spatial$max_iter, cache_dir = CFG$paths$cache_dir)
  }
  row <- function(section, metric, term, estimate, lo = NA_real_, hi = NA_real_, p = NA_real_) {
    data.frame(run = key, section = section, metric = metric, term = term, estimate = estimate,
               conf.low = lo, conf.high = hi, p.value = p, stringsAsFactors = FALSE)
  }
  out <- list()
  for (m in metrics) {
    resp <- paste0(CFG$responses[[m]], "_mean")
    ct <- coef_table(fit_fixed(resp, "add", range_of(m, "main effects"), CFG$ref), m)
    ct <- ct[ct$term != "(Intercept)", ]
    out[[length(out) + 1]] <- row("Mean model, main effects", m, ct$term, ct$estimate, ct$conf.low, ct$conf.high, ct$p.value)
    w <- wald_tests(fit_fixed(resp, "full", range_of(m, "interaction"), CFG$ref), m, hypotheses = TRUE)
    out[[length(out) + 1]] <- row("Mean model, HMI interactions (Wald F)", m, w$hypothesis, w$F.value, p = w$p.value)
    ca <- coef_table(fit_fixed(paste0("change_", m, "_z"), "add", range_of(m, CHANGE_MODEL_LABELS[["A"]]),
                               CFG$ref), m)
    ca <- ca[ca$term != "(Intercept)", ]
    out[[length(out) + 1]] <- row("Change model A", m, ca$term, ca$estimate, ca$conf.low, ca$conf.high, ca$p.value)
  }
  cls <- stats::setNames(paste0("trend_", metrics), labels)
  cc <- class_counts(d, cls, "BIOME_NAME", CFG$biome_order, CFG$trend$levels, CFG$pooled_label)
  cc <- cc[cc$BIOME_NAME == CFG$pooled_label & cc$class != CFG$trend$levels[2], ]
  out[[length(out) + 1]] <- row("Trend classes (% of PAs)", names(labels)[match(cc$metric, labels)],
                                paste0("% ", cc$class), cc$pct)
  k <- stats::setNames(CFG$n_span / sd_resp, labels)
  bcols <- stats::setNames(paste0("slope_", metrics), labels)
  bm <- rbind(boot_means(d, bcols, "BIOME_NAME", CFG$biome_order, CFG$boot$R, CFG$boot$seeds[["biome"]],
                         pooled_label = CFG$pooled_label)$summary,
              boot_means(d, bcols, "BIOME_NAME", CFG$biome_order, CFG$boot$R, CFG$boot$seeds[["pooled"]],
                         pooled = "only", pooled_label = CFG$pooled_label)$summary)
  out[[length(out) + 1]] <- row("Mean change (response SDs)", names(labels)[match(bm$metric, labels)], bm$BIOME_NAME,
                                bm$mean * k[bm$metric], bm$ci_lo * k[bm$metric], bm$ci_hi * k[bm$metric])
  res <- do.call(rbind, out)
  list(results = res, data = d[c("WDPA_PID", paste0(CFG$responses, "_mean"), paste0("change_", metrics, "_z"),
                                 paste0("trend_", metrics))],
       info = data.frame(run = key, label = r$label, transects = r$version, min_transects = r$min_tr,
                         reference = r$vs, n_PAs = nrow(d), n_PA_years = attr(d, "n_pa_years"),
                         stringsAsFactors = FALSE))
}

done <- parallel::mclapply(names(runs), function(key) tryCatch(run_results(key), error = function(e) e),
                           mc.cores = cores, mc.preschedule = FALSE)
names(done) <- names(runs)
for (key in names(done)) if (inherits(done[[key]], "error") || inherits(done[[key]], "try-error")) {
  stop("run ", key, " failed: ", conditionMessage(done[[key]]))
}

# ── 3. S2: each run beside its reference ─────────────────────────────────────
res <- do.call(rbind, lapply(done, `[[`, "results"))
key_cols <- c("section", "metric", "term")
s2 <- do.call(rbind, lapply(names(runs), function(key) {
  x <- res[res$run == key, ]
  ref <- runs[[key]]$vs
  x$reference <- ref
  if (is.na(ref)) {
    x[c("ref_estimate", "ref_conf.low", "ref_conf.high", "ref_p.value")] <- NA_real_
  } else {
    y <- res[res$run == ref, ]
    i <- match(do.call(paste, x[key_cols]), do.call(paste, y[key_cols]))
    x$ref_estimate <- y$estimate[i]; x$ref_conf.low <- y$conf.low[i]
    x$ref_conf.high <- y$conf.high[i]; x$ref_p.value <- y$p.value[i]
  }
  x
}))
sig <- function(p, lo, hi) ifelse(!is.na(p), p < 0.05, ifelse(!is.na(lo), lo > 0 | hi < 0, NA))
has_sign <- !grepl("Wald|Trend classes", s2$section)
s2$same_sign <- ifelse(has_sign & !is.na(s2$ref_estimate), sign(s2$estimate) == sign(s2$ref_estimate), NA)
s2$same_significance <- ifelse(!is.na(s2$ref_estimate) & !grepl("Trend classes", s2$section),
                               sig(s2$p.value, s2$conf.low, s2$conf.high) ==
                                 sig(s2$ref_p.value, s2$ref_conf.low, s2$ref_conf.high), NA)
s2 <- cbind(s2[c("run", "reference")], s2[setdiff(names(s2), c("run", "reference"))])
rownames(s2) <- NULL

# ── 4. S1: per-PA agreement of each transect subset with all transects (same PAs) ──
base <- done[["all_common"]]$data
s1 <- do.call(rbind, lapply(c("nowater", "nootherpa", "noflags"), function(v) {
  x <- done[[v]]$data
  stopifnot(identical(x$WDPA_PID, base$WDPA_PID))
  do.call(rbind, lapply(metrics, function(m) {
    mn <- paste0(CFG$responses[[m]], "_mean"); ch <- paste0("change_", m, "_z"); tc <- paste0("trend_", m)
    a <- paired_agreement(base[[mn]], x[[mn]], sd_resp[[m]], CFG$boot$R, CFG$sensitivity$seed)
    b <- paired_agreement(base[[ch]], x[[ch]], 1, CFG$boot$R, CFG$sensitivity$seed)
    data.frame(subset = v, label = runs[[v]]$label, metric = m, n_PAs = a$n,
               r_mean = a$r, mean_diff_sd = a$mean_diff_sd, mean_diff_lo = a$diff_lo, mean_diff_hi = a$diff_hi,
               r_change = b$r, change_diff_sd = b$mean_diff_sd, change_diff_lo = b$diff_lo, change_diff_hi = b$diff_hi,
               pct_class_agreement = 100 * mean(base[[tc]] == x[[tc]]),
               kappa_class = cohen_kappa(base[[tc]], x[[tc]], CFG$trend$levels), stringsAsFactors = FALSE)
  }))
}))

# ── 5. Checks: the main run reproduces 06's main-effects coefficients; one optimised refit ─
checks <- list()
sx_file <- file.path(CFG$paths$tables, "TableSX_mean_models_main_effects.csv")
if (is.na(pilot) && file.exists(sx_file)) {
  sx <- read(sx_file)
  x <- res[res$run == "main" & res$section == "Mean model, main effects", ]
  i <- match(paste(x$metric, x$term), paste(sx$response, sx$term))
  checks$main <- data.frame(check = "main run (fixed range) vs TableSX (06, optimised range)",
                            n = sum(!is.na(i)),
                            max_rel_diff_estimate = max(abs(x$estimate - sx$estimate[i]) / pmax(abs(sx$estimate[i]), 1e-8)),
                            max_abs_diff_p = max(abs(x$p.value - sx$p.value[i])))
}
refit <- NULL
if (do_refit) {
  say("refit check: noflags, extent, main effects, range optimised from %g km", CFG$spatial$range_start_km)
  d <- set_model_references(run_data(runs$noflags), CFG$ref, names(CFG$iucn_groups), CFG$biome_order)
  f <- model_formula(paste0(CFG$responses[["extent"]], "_mean"), "add")
  cs_fix <- spatial_setup(d, value = range_of("extent", "main effects"), fixed = TRUE, cache_dir = CFG$paths$cache_dir)
  cs_opt <- spatial_setup(d, value = CFG$spatial$range_start_km, nugget = CFG$spatial$nugget, cache_dir = CFG$paths$cache_dir)
  m_fix <- fit_gls_cached(f, d, cs_fix, method = CFG$spatial$method, max_iter = CFG$spatial$max_iter,
                          cache_dir = CFG$paths$cache_dir)
  m_opt <- fit_gls_cached(f, d, cs_opt, method = CFG$spatial$method, max_iter = CFG$spatial$max_iter,
                          cache_dir = CFG$paths$cache_dir,
                          label = "noflags extent main effects, optimised")
  a <- coef_table(m_fix, "extent"); b <- coef_table(m_opt, "extent")
  refit <- data.frame(term = a$term, estimate_fixed = a$estimate, estimate_optimised = b$estimate,
                      se_fixed = a$std.error, se_optimised = b$std.error,
                      diff_in_se = (a$estimate - b$estimate) / b$std.error,
                      p_fixed = a$p.value, p_optimised = b$p.value,
                      range_fixed_km = range_of("extent", "main effects"),
                      range_optimised_km = stats::coef(m_opt$modelStruct$corStruct, unconstrained = FALSE)[["range"]])
  say("refit: range %.2f km (fixed at %.2f); max |coef difference| = %.3f SE", refit$range_optimised_km[1],
      refit$range_fixed_km[1], max(abs(refit$diff_in_se)))
}

# ── 6. Write ─────────────────────────────────────────────────────────────────
for (dir in out) dir.create(dir, recursive = TRUE, showWarnings = FALSE)
w <- function(x, dir, name) readr::write_csv(x, file.path(dir, name), na = "")
w(s1, out$tables, "TableS_sensitivity_metrics.csv")
w(s2, out$tables, "TableS_sensitivity_models.csv")
w(do.call(rbind, lapply(done, `[[`, "info")), out$diagnostics, "sensitivity_runs.csv")
if (length(checks)) w(do.call(rbind, checks), out$diagnostics, "sensitivity_main_check.csv")
if (!is.null(refit)) w(refit, out$diagnostics, "sensitivity_refit_check.csv")
for (i in seq_len(nrow(s1))) say("%s %s: r(means) %.3f, diff %.3f SD; r(change) %.3f; classes %.1f%% (kappa %.2f)",
                                 s1$subset[i], s1$metric[i], s1$r_mean[i], s1$mean_diff_sd[i], s1$r_change[i],
                                 s1$pct_class_agreement[i], s1$kappa_class[i])
x <- s2[!is.na(s2$same_significance), ]
say("S2: %d of %d compared results keep their significance; %d of %d keep their sign", sum(x$same_significance),
    nrow(x), sum(x$same_sign, na.rm = TRUE), sum(!is.na(x$same_sign)))
say("wrote %s and %s", out$tables, out$diagnostics)
