# Comparison helpers for the parity tests, and the parity report (data/parity/parity_report.csv).

# Largest elementwise relative difference, |new − old| / max(|old|, floor), where
# floor = rel_floor × the largest |old| in the column. Values a million times smaller than the
# column's largest are therefore compared on that scale: two algorithms that both compute an
# exact zero as ±1e-17 (e.g. a trend slope of a series uncorrelated with year: lm() gives
# 1.4e-17, the sums 6.6e-19) are not flagged. NA patterns must match; a mismatch returns Inf.
max_rel_diff <- function(new, old, rel_floor = 1e-6) {
  new <- as.numeric(new); old <- as.numeric(old)
  if (length(new) != length(old) || any(is.na(new) != is.na(old))) return(Inf)
  ok <- !is.na(old)
  if (!any(ok)) return(0)
  floor <- max(rel_floor * max(abs(old[ok])), .Machine$double.xmin)
  max(abs(new[ok] - old[ok]) / pmax(abs(old[ok]), floor))
}

.parity <- new.env()
.parity$rows <- list()

# Record one parity check and assert it. `kind` is
#   "rel"        numeric, max relative difference (max_rel_diff) ≤ tol;
#   "abs"        numeric, max absolute difference ≤ tol (rounded old tables: tol = half the last
#                digit; bootstrap CIs whose random stream changed: values divided by the old CI
#                width, tol = 0.05, i.e. within 5% of the CI width);
#   "identical"  exact equality, e.g. classes and letters (tol is ignored).
# The max_rel_diff column holds the max relative or absolute difference, by kind.
parity_check <- function(output, item, new, old, tol = 1e-8, kind = c("rel", "abs", "identical"),
                         source = "fast") {
  kind <- match.arg(kind)
  if (kind == "rel") {
    d <- max_rel_diff(new, old)
    pass <- is.finite(d) && d <= tol
    detail <- sprintf("max rel diff %.3g (tol %.0e)", d, tol)
  } else if (kind == "abs") {
    new <- as.numeric(new); old <- as.numeric(old)
    d <- if (length(new) != length(old) || any(is.na(new) != is.na(old))) Inf
         else if (all(is.na(old))) 0 else max(abs(new - old), na.rm = TRUE)
    pass <- is.finite(d) && d <= tol
    detail <- sprintf("max abs diff %.3g (tol %.3g)", d, tol)
  } else {
    n_diff <- if (length(new) != length(old)) NA else sum(xor(is.na(new), is.na(old)) |
                                                         (!is.na(new) & !is.na(old) & new != old))
    pass <- isTRUE(n_diff == 0)
    detail <- if (is.na(n_diff)) "lengths differ" else sprintf("%d values differ", n_diff)
    d <- if (pass) 0 else NA_real_
  }
  .parity$rows[[length(.parity$rows) + 1]] <- data.frame(
    source = source, output = output, item = item, kind = kind, tol = if (kind != "identical") tol else NA,
    max_rel_diff = d, pass = pass, detail = detail, stringsAsFactors = FALSE)
  testthat::expect_true(pass, label = sprintf("%s / %s: %s", output, item, detail))
  invisible(pass)
}

# Write the recorded rows, replacing earlier rows of the same source(s) in the report.
parity_write_report <- function(path) {
  if (!length(.parity$rows)) return(invisible(NULL))
  new <- do.call(rbind, .parity$rows)
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  if (file.exists(path)) {
    old <- utils::read.csv(path, stringsAsFactors = FALSE)
    new <- rbind(old[!old$source %in% unique(new$source), , drop = FALSE], new)
  }
  utils::write.csv(new, path, row.names = FALSE)
  invisible(new)
}
