# Fast parity tests against the collaborator's old results (tests/r/parity/).
# Skipped when PA_New_Analysis/ is absent. The full spatial refits are run separately
# (tests/r/parity/run_parity_full.R, in the background).
source(testthat::test_path("parity", "test_parity_fast.R"), local = TRUE)

# One row per check; the gate for the r-port-parity tag is that every row passes.
if (legacy_available()) parity_write_report(file.path(CFG$paths$parity_dir, "parity_report.csv"))
