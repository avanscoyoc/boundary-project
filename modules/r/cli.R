# Command-line helpers shared by the R scripts (05–10, 14).

#' The value after flag `name` in the script's arguments (e.g. arg("--out")), or NULL.
arg <- function(name, args = commandArgs(trailingOnly = TRUE)) {
  if (any(args == name)) args[which(args == name) + 1] else NULL
}

#' TRUE if flag `name` is among the script's arguments (e.g. has_flag("--no-refit")).
has_flag <- function(name, args = commandArgs(trailingOnly = TRUE)) any(args == name)

#' A timestamped log line: say("%d PAs", n).
say <- function(...) cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"), sprintf(...)))
