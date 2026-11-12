# Sourced automatically by testthat::test_dir("tests/r") before the tests; can also be sourced
# by hand from anywhere inside the repository.
# Loads the settings, the analysis modules and the conventions of the legacy analysis
# (tests/r/parity/legacy_inputs.R; parity tests only).
local({
  root <- normalizePath(getwd())
  while (!file.exists(file.path(root, "modules", "r", "config.R"))) {
    if (dirname(root) == root) stop("Run from inside the repository.")
    root <- dirname(root)
  }
  source(file.path(root, "modules", "r", "config.R"), local = globalenv())
  for (f in setdiff(list.files(file.path(root, "modules", "r"), pattern = "\\.R$", full.names = TRUE),
                    file.path(root, "modules", "r", "config.R"))) {
    sys.source(f, envir = globalenv())   # corHaversine must live in globalenv() (see its header)
  }
  source(file.path(root, "tests", "r", "parity", "legacy_inputs.R"), local = globalenv())
})
