# Loads the R settings (config.R) and every analysis module of modules/r into globalenv(), and
# defines the scripts' helpers arg() and say(). The R scripts start with
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
