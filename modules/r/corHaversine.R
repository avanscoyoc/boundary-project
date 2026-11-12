# corHaversine: great-circle (Haversine) spatial correlation structure for nlme::gls / lme.
#
# Vendored from HaversineLMEfunctions.R of the earlier R implementation of this analysis (not in
# this repository), which adapts a widely circulated community helper.
#
# ONE CHANGE: haversineDist() is vectorised. The original filled an n × n matrix with a double R
# loop over the scalar haversine() (≈ 1 min at n ≈ 3,900) and returned as.dist() of it. The
# vectorised version computes the same lower triangle in the same column-major (as.dist) order
# with the same arithmetic, elementwise: min(1, ·) became pmin(1, ·), R = 6371 km. The result is
# bit-identical (tests/r/test_spatial.R checks it against the original loop). haversine() is kept
# unchanged as the reference for that test. Everything below the distance functions is verbatim.
#
# MUST BE SOURCED INTO globalenv() (sys.source(..., envir = globalenv())). The constructor, Dim
# and getCovariate methods have their environment set to the nlme namespace, so they find
# haversineDist() and the S3 methods only through the namespace's fall-back to the global
# environment.
#
# Usage: Initialize(corHaversine(value = <range km>, form = ~ lon + lat, mimic = "corExp"),
#                   data = d)   — coordinates must be columns named lon and lat, in degrees.

# ---- Distance functions ----------------------------------------------------
# Great-circle distance between two points (in radians) using the Haversine
# formula. Returns distance in km, assuming Earth radius = 6371 km.
# (Unchanged; not used by haversineDist(), kept as the reference for its test.)
haversine <- function(x0, x1, y0, y1) {
  a <- sin((y1 - y0) / 2)^2 + cos(y0) * cos(y1) * sin((x1 - x0) / 2)^2
  v <- 2 * asin(min(1, sqrt(a)))
  6371 * v
}

# Pairwise great-circle distances (km) for a two-column matrix of longitude/latitude, as a "dist"
# object (lower triangle, column-major). Input in decimal degrees (default) or radians.
# Vectorised over pairs, in blocks of columns of at most `block` pairs to bound memory.
haversineDist <- function(xy, radians = FALSE, block = 2e6) {
  if (ncol(xy) > 2) stop("Input must have two columns (longitude and latitude)")
  xy <- as.matrix(xy)
  if (!radians) xy <- xy * pi / 180
  n <- nrow(xy)
  out <- numeric(n * (n - 1) / 2)
  if (n > 1) {
    x <- xy[, 1]; y <- xy[, 2]; cy <- cos(y)
    # column i of the lower triangle holds rows j = i+1..n; it starts at offset[i] + 1
    len <- (n - 1):1
    offset <- c(0, cumsum(as.numeric(len)))[seq_len(n - 1)]
    i0 <- 1L
    while (i0 <= n - 1) {
      i1 <- i0
      while (i1 < n - 1 && sum(len[i0:(i1 + 1)]) <= block) i1 <- i1 + 1L
      i <- rep.int(i0:i1, len[i0:i1])
      j <- sequence(len[i0:i1], from = (i0:i1) + 1L)
      # same expression as haversine(x0 = x[i], x1 = x[j], y0 = y[i], y1 = y[j])
      a <- sin((y[j] - y[i]) / 2)^2 + cy[i] * cy[j] * sin((x[j] - x[i]) / 2)^2
      out[offset[i0] + seq_along(i)] <- 6371 * (2 * asin(pmin(1, sqrt(a))))
      i0 <- i1 + 1L
    }
  }
  structure(out, Size = n, Labels = NULL, Diag = FALSE, Upper = FALSE,
            method = NULL, call = NULL, class = "dist")
}


# ---- Inherit standard methods from corSpatial ------------------------------
# Most of the machinery from nlme's corSpatial parent class works without
# modification; we simply alias these methods to corHaversine.
Initialize.corHaversine <- nlme:::Initialize.corSpatial
recalc.corHaversine     <- nlme:::recalc.corSpatial
Variogram.corHaversine  <- nlme:::Variogram.corSpatial
corFactor.corHaversine  <- nlme:::corFactor.corSpatial
corMatrix.corHaversine  <- nlme:::corMatrix.corSpatial
coef.corHaversine       <- nlme:::coef.corSpatial
"coef<-.corHaversine"   <- nlme:::"coef<-.corSpatial"


# ---- Constructor for the corHaversine class --------------------------------
corHaversine <- function(value  = numeric(0),
                         form   = ~ 1,
                         mimic  = "corExp",
                         nugget = FALSE,
                         fixed  = FALSE) {
  spClass <- "corHaversine"
  attr(value, "formula")  <- form
  attr(value, "nugget")   <- nugget
  attr(value, "fixed")    <- fixed
  attr(value, "function") <- mimic
  class(value) <- c(spClass, "corStruct")
  value
}
environment(corHaversine) <- asNamespace("nlme")


# ---- Dim method -------------------------------------------------------------
# Custom Dim method that uses the third element of the Dim list to store the
# spatial-class index, so that downstream methods know which built-in
# correlation shape to emulate.
Dim.corHaversine <- function(object, groups, ...) {
  if (missing(groups)) return(attr(object, "Dim"))
  val <- Dim.corStruct(object, groups)
  val[["start"]] <- c(0, cumsum(val[["len"]] * (val[["len"]] - 1) / 2)[-val[["M"]]])
  names(val)[3] <- "spClass"
  val[[3]] <- match(attr(object, "function"),
                    c("corSpher", "corExp", "corGaus", "corLin", "corRatio"),
                    0)
  val
}
environment(Dim.corHaversine) <- asNamespace("nlme")


# ---- getCovariate method ---------------------------------------------------
# Extracts the lon/lat covariate from the model frame and converts it to a
# pairwise Haversine-distance representation (grouped or ungrouped, depending
# on whether the formula includes a grouping factor).
getCovariate.corHaversine <- function(object, form = formula(object), data) {
  
  if (is.null(covar <- attr(object, "covariate"))) {
    
    if (missing(data)) stop("need data to calculate covariate")
    
    covForm <- getCovariateFormula(form)
    
    if (length(all.vars(covForm)) > 0) {
      
      # Drop intercept if present
      if (attr(terms(covForm), "intercept") == 1) {
        covForm <- eval(parse(text = paste0("~", deparse(covForm[[2]]), "-1")))
      }
      
      # Validate that the covariates are exactly "lon" and "lat"
      if (length(all.vars(covForm)) > 2) {
        stop("corHaversine can only take two covariates, 'lon' and 'lat'")
      }
      if (!all(all.vars(covForm) %in% c("lon", "lat"))) {
        stop("covariates must be named 'lon' and 'lat'")
      }
      
      covar <- as.data.frame(unclass(
        model.matrix(covForm,
                     model.frame(covForm, data, drop.unused.levels = TRUE))
      ))
      covar <- covar[, order(colnames(covar), decreasing = TRUE)]  # ensure lon, lat order
      
    } else {
      covar <- NULL
    }
    
    # Compute pairwise distances, optionally within groups
    if (!is.null(getGroupsFormula(form))) {
      
      grps <- getGroups(object, data = data)
      
      if (is.null(covar)) {
        covar <- lapply(split(grps, grps),
                        function(x) as.vector(dist(1:length(x))))
      } else {
        giveDist <- function(el) {
          el <- as.matrix(el)
          if (nrow(el) > 1) as.vector(haversineDist(el)) else numeric(0)
        }
        covar <- lapply(split(covar, grps), giveDist)
      }
      covar <- covar[sapply(covar, length) > 0]  # drop singleton groups
      
    } else {
      
      if (is.null(covar)) {
        covar <- as.vector(dist(1:nrow(data)))
      } else {
        covar <- as.vector(haversineDist(as.matrix(covar)))
      }
    }
    
    if (any(unlist(covar) == 0)) {
      stop("cannot have zero distances in \"corHaversine\"")
    }
  }
  
  covar
}
environment(getCovariate.corHaversine) <- asNamespace("nlme")
