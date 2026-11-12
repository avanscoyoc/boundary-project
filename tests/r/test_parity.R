# Fast parity tests against the results of the legacy analysis, the earlier R implementation
# (tests/r/parity/).
# Skipped when PA_New_Analysis/ is absent. The full spatial refits are run separately
# (tests/r/parity/run_parity_full.R, in the background).
source(testthat::test_path("parity", "test_parity_fast.R"), local = TRUE)

# One row per check; parity holds when every row passes.
if (legacy_available()) parity_write_report(file.path(CFG$paths$parity_dir, "parity_report.csv"))
