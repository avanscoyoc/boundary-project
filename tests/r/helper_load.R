# Sourced by testthat::test_dir("tests/r") before the tests, or by hand from anywhere in the
# repository. Loads modules/r/load.R, as the scripts do.
local({
  root <- normalizePath(getwd())
  while (!file.exists(file.path(root, "modules", "r", "load.R"))) {
    if (dirname(root) == root) stop("Run from inside the repository.")
    root <- dirname(root)
  }
  source(file.path(root, "modules", "r", "load.R"))
})
