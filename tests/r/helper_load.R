# Sourced automatically by testthat::test_dir("tests/r") before the tests.
# Loads the settings, the analysis modules and the old-run conventions (parity tests only).
local({
  root <- normalizePath(file.path(testthat::test_path(), "..", ".."))
  source(file.path(root, "modules", "r", "config.R"), local = globalenv())
  for (f in setdiff(list.files(file.path(root, "modules", "r"), pattern = "\\.R$", full.names = TRUE),
                    file.path(root, "modules", "r", "config.R"))) {
    sys.source(f, envir = globalenv())   # corHaversine must live in globalenv() (see its header)
  }
  source(file.path(root, "tests", "r", "parity", "legacy_inputs.R"), local = globalenv())
})
