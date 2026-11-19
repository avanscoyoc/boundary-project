# Loads the R settings (config.R) and every analysis module of modules/r into globalenv(), and
# defines the command-line helpers of the scripts. The R scripts start with
# source(file.path("modules", "r", "load.R")), run from the repository root; tests/r/helper_load.R
# sources it as well. corHaversine must live in globalenv() (see its header), so the modules are
# sys.source()d there.
local({
  root <- normalizePath(getwd())
  while (!file.exists(file.path(root, "modules", "r", "config.R"))) {
    if (dirname(root) == root) stop("Run from inside the repository.")
    root <- dirname(root)
  }
  dir <- file.path(root, "modules", "r")
  source(file.path(dir, "config.R"), local = globalenv())
  for (f in setdiff(list.files(dir, pattern = "\\.R$", full.names = TRUE),
                    file.path(dir, c("config.R", "load.R")))) {
    sys.source(f, envir = globalenv())
  }
})

#' The value after flag `name` in the script's arguments (e.g. arg("--cores")), or NULL.
arg <- function(name, args = commandArgs(trailingOnly = TRUE)) {
  if (any(args == name)) args[which(args == name) + 1] else NULL
}

#' A timestamped log line: say("%d PAs", n).
say <- function(...) cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"), sprintf(...)))

#' Zip the CSV tables of `dir` into `zipfile` (results/tables.zip), for sharing. Rewritten at
#' the end of 10 and of 14, so it always matches the tables. Uses Python's zipfile module (the
#' container has no zip program); warns and skips without python3.
zip_tables <- function(dir = CFG$paths$tables, zipfile = file.path(dirname(dir), "tables.zip")) {
  files <- sort(list.files(dir, pattern = "\\.csv$"))
  py <- Sys.which("python3")
  if (!length(files)) return(invisible(NULL))
  if (!nzchar(py)) {
    warning("python3 not found: ", zipfile, " not written", call. = FALSE)
    return(invisible(NULL))
  }
  zipfile <- file.path(normalizePath(dirname(zipfile)), basename(zipfile))
  tmp <- paste0(zipfile, ".tmp")
  code <- paste("import sys, zipfile",
                "z = zipfile.ZipFile(sys.argv[1], 'w', zipfile.ZIP_DEFLATED)",
                "[z.write(f) for f in sys.argv[2:]]", "z.close()", sep = "; ")
  owd <- setwd(dir); on.exit(setwd(owd))
  if (system2(py, c("-c", shQuote(code), shQuote(tmp), shQuote(files))) != 0) stop("zipping ", dir, " failed")
  if (!file.rename(tmp, zipfile)) stop("could not write ", zipfile)
  invisible(zipfile)
}
