# Sourced automatically by testthat::test_dir("tests/r") before the tests; can also be sourced
# by hand from anywhere inside the repository. Loads the settings and the analysis modules
# (modules/r/load.R, as the scripts do).
local({
  root <- normalizePath(getwd())
  while (!file.exists(file.path(root, "modules", "r", "load.R"))) {
    if (dirname(root) == root) stop("Run from inside the repository.")
    root <- dirname(root)
  }
  source(file.path(root, "modules", "r", "load.R"))
})
