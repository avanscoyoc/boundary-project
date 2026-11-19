# corHaversine: great-circle (Haversine) spatial correlation structure for nlme::gls / lme.
#
# Source: Nate Pope's answer on Stack Overflow,
#   https://stackoverflow.com/questions/18857443/specifying-a-correlation-structure-for-a-linear-mixed-model-using-the-ramps-pack
# (also distributed as R/corHaversine.R in https://github.com/biometry/FReibier). Stack Overflow
# content is licensed under CC BY-SA (https://stackoverflow.com/help/licensing), so this file is
# shared under CC BY-SA with that attribution, not under the repository's MIT licence.
#
# One change: haversineDist() is vectorised. The original filled an n × n matrix with a double
# loop over the scalar haversine() (about a minute for a few thousand PAs). This version computes
# the same lower triangle, in the same as.dist order, with the same arithmetic (min became pmin;
# R = 6371 km), so the result is bit-identical; tests/r/test_spatial.R checks it against the
# original loop. The code below the distance functions is as in the source.
#
# Must be sourced into globalenv() (sys.source(..., envir = globalenv())): the constructor, Dim
# and getCovariate methods have the nlme namespace as environment, so they find haversineDist()
# and the S3 methods only through its fall-back to the global environment.
#
# Usage: Initialize(corHaversine(value = <range km>, form = ~ lon + lat, mimic = "corExp"),
#                   data = d)   (columns lon and lat, in degrees).

# ---- Distance functions ----------------------------------------------------
# Pairwise great-circle distances (km) between the lon/lat rows of `xy` (degrees, or radians), as
# a "dist" object. Pairs are computed in column blocks of at most `block` pairs to bound memory.
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


# ---- Methods inherited unchanged from corSpatial ----------------------------
Initialize.corHaversine <- nlme:::Initialize.corSpatial
recalc.corHaversine     <- nlme:::recalc.corSpatial
Variogram.corHaversine  <- nlme:::Variogram.corSpatial
corFactor.corHaversine  <- nlme:::corFactor.corSpatial
corMatrix.corHaversine  <- nlme:::corMatrix.corSpatial
coef.corHaversine       <- nlme:::coef.corSpatial
"coef<-.corHaversine"   <- nlme:::"coef<-.corSpatial"


# ---- Constructor ------------------------------------------------------------
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
# The third element of Dim ("spClass") holds the index of the built-in shape to mimic.
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
# Pairwise Haversine distances of the lon/lat covariate, within groups if the formula has a
# grouping factor.
getCovariate.corHaversine <- function(object, form = formula(object), data) {
  
  if (is.null(covar <- attr(object, "covariate"))) {
    
    if (missing(data)) stop("need data to calculate covariate")
    
    covForm <- getCovariateFormula(form)
    
    if (length(all.vars(covForm)) > 0) {
      
      # Drop intercept if present
      if (attr(terms(covForm), "intercept") == 1) {
        covForm <- eval(parse(text = paste0("~", deparse(covForm[[2]]), "-1")))
      }

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
