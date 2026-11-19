# Loads config.R and every module of modules/r into globalenv() and defines the script helpers
# arg() and say(). The R scripts (run from the repository root) and tests/r/helper_load.R source
# it. Modules are sys.source()d into globalenv() because corHaversine must live there.
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
