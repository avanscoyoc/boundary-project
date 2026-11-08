# Unit tests for modules/r/prep.R (simulated data, seconds).

donor_pool <- function() {
  # Country AAA: 6 known years (1950–1975); BBB: 2 known (2000, 2001); CCC: 5 known (1900–1904)
  data.frame(WDPA_PID = sprintf("d%02d", 1:13),
             ISO3 = c(rep("AAA", 6), rep("BBB", 2), rep("CCC", 5)),
             STATUS_YR = c(seq(1950, 1975, 5), 2000, 2001, 1900:1904), stringsAsFactors = FALSE)
}

test_that("impute_year: a country with >= min_country known years draws only from that country", {
  tg <- data.frame(WDPA_PID = "t1", ISO3 = "AAA", STATUS_YR = 0)
  r <- impute_year(tg, rbind(donor_pool(), tg), draws = 1000, min_country = 5, seed = 1)
  expect_true(r$STATUS_YR_imputed); expect_equal(r$impute_pool, "country"); expect_equal(r$n_donors, 6L)
  expect_gte(r$STATUS_YR, 1950); expect_lte(r$STATUS_YR, 1975)
  # Every draw comes from the country: with one huge-year donor elsewhere the median cannot move
  d <- donor_pool(); d$STATUS_YR[d$ISO3 == "BBB"] <- 3000
  expect_equal(impute_year(tg, d, 1000, 5, 1)$STATUS_YR, r$STATUS_YR)
})

test_that("impute_year: a country with < min_country known years uses the global pool", {
  tg <- data.frame(WDPA_PID = "t1", ISO3 = "BBB", STATUS_YR = 0)
  r <- impute_year(tg, donor_pool(), draws = 1000, min_country = 5, seed = 1)
  expect_equal(r$impute_pool, "global"); expect_equal(r$n_donors, 13L)
  # A country with no donors at all also goes global
  r2 <- impute_year(data.frame(WDPA_PID = "t2", ISO3 = "ZZZ", STATUS_YR = 0), donor_pool(), 100, 5, 1)
  expect_equal(r2$impute_pool, "global")
  # Year-0 donors are not donors: 4 known + 3 unknown in DDD stays below 5
  d <- rbind(donor_pool(), data.frame(WDPA_PID = sprintf("e%d", 1:7), ISO3 = "DDD",
                                      STATUS_YR = c(1990:1993, 0, 0, 0)))
  expect_equal(impute_year(data.frame(WDPA_PID = "t3", ISO3 = "DDD", STATUS_YR = 0), d, 100, 5, 1)$impute_pool, "global")
})

test_that("impute_year: known years are unchanged", {
  tg <- data.frame(WDPA_PID = c("k1", "t1", "k2"), ISO3 = "AAA", STATUS_YR = c(1987, 0, 2010))
  r <- impute_year(tg, donor_pool(), 1000, 5, 1)
  expect_equal(r$STATUS_YR[c(1, 3)], c(1987, 2010)); expect_equal(r$STATUS_YR_imputed, c(FALSE, TRUE, FALSE))
  expect_equal(r$STATUS_YR_reported, c(1987, 0, 2010)); expect_true(all(is.na(r$impute_pool[c(1, 3)])))
  expect_equal(r$WDPA_PID, tg$WDPA_PID)
})

test_that("impute_year: transboundary PAs pool all listed countries", {
  # BBB alone has 2 donors (global); BBB;CCC pools 2 + 5 = 7 donors (country)
  tg <- data.frame(WDPA_PID = "t1", ISO3 = "BBB;CCC", STATUS_YR = 0)
  r <- impute_year(tg, donor_pool(), 1000, 5, 1)
  expect_equal(r$impute_pool, "country"); expect_equal(r$n_donors, 7L)
  expect_true(r$STATUS_YR %in% c(1900:1904, 2000, 2001) || r$STATUS_YR == floor(r$STATUS_YR) + 0.5)
  # A transboundary donor counts for each of its countries, once
  d <- rbind(donor_pool(), data.frame(WDPA_PID = "x1", ISO3 = "AAA;BBB", STATUS_YR = 1960))
  expect_equal(impute_year(data.frame(WDPA_PID = "t2", ISO3 = "AAA", STATUS_YR = 0), d, 10, 5, 1)$n_donors, 7L)
  expect_equal(impute_year(data.frame(WDPA_PID = "t3", ISO3 = "AAA;BBB", STATUS_YR = 0), d, 10, 5, 1)$n_donors, 9L)
})

test_that("impute_year: the same seed gives the same result; a single-donor pool is drawn by index", {
  tg <- data.frame(WDPA_PID = sprintf("t%d", 1:5), ISO3 = c("AAA", "BBB", "CCC", "AAA", "ZZZ"), STATUS_YR = 0)
  a <- impute_year(tg, donor_pool(), 1000, 5, 42); b <- impute_year(tg, donor_pool(), 1000, 5, 42)
  expect_identical(a, b)
  c <- impute_year(tg, donor_pool(), 1000, 5, 43)
  expect_false(identical(a$STATUS_YR, c$STATUS_YR))
  # One known donor and min_country = 1: every draw is that year (sample(1975) would draw 1:1975)
  one <- data.frame(WDPA_PID = "d1", ISO3 = "AAA", STATUS_YR = 1975)
  expect_equal(impute_year(tg[1, ], one, 1000, 1, 1)$STATUS_YR, 1975)
})

test_that("pa_year_long reshapes the versions and pa_means averages over non-NA years", {
  m <- data.frame(WDPA_PID = c("2", "2", "10", "10"), year = c(2003, 2004, 2003, 2004),
                  n_trnst = c(5, 6, 7, 8), edge_extent = c(1, 3, NA, 2), edge_intensity = c(0.5, 1.5, 1, NA),
                  n_trnst_nowater = c(4, 4, 6, 6), edge_extent_nowater = c(2, 2, 1, 1),
                  edge_intensity_nowater = c(0, 1, 1, 1))
  py <- pa_year_long(m, c("all", "nowater"))
  expect_equal(nrow(py), 8); expect_equal(levels(py$transects), c("all", "nowater"))
  expect_equal(py$WDPA_PID, c(rep("10", 4), rep("2", 4)))         # character order
  expect_equal(py$edge_extent[py$WDPA_PID == "2" & py$transects == "nowater"], c(2, 2))
  expect_error(pa_year_long(m, c("all", "noflags")), "not found")
  pm <- pa_means(py)
  expect_equal(nrow(pm), 4)
  r <- pm[pm$WDPA_PID == "10" & pm$transects == "all", ]
  expect_equal(r$edge_extent_mean, 2); expect_equal(r$n_years_edge_extent, 1)
  expect_equal(r$edge_intensity_mean, 1); expect_equal(r$n_years_edge_intensity, 1)
  r <- pm[pm$WDPA_PID == "2" & pm$transects == "all", ]
  expect_equal(r$edge_extent_mean, 2); expect_equal(r$edge_intensity_mean, 1)
  py$edge_extent[py$WDPA_PID == "10" & py$transects == "all"] <- NA
  expect_true(is.na(pa_means(py)$edge_extent_mean[1]))
})

test_that("analysis_set applies the steps in order and records the counts", {
  pa <- data.frame(WDPA_PID = c("c", "a", "b", "d", "e", "f"),
                   BIOME_NAME = c("Desert", "Mangrove", NA, "Tundra", "Desert", "Tundra"),
                   gHM_mean = c(0.1, 0.2, 0.3, NA, 0.5, 0.6),
                   n_years_edge_extent = c(10, 10, 10, 10, 4, 10),
                   n_years_edge_intensity = c(5, 10, 10, 10, 10, 10))
  r <- analysis_set(pa, c("Mangrove", "Rock & Ice"), "gHM_mean",
                    c("n_years_edge_extent", "n_years_edge_intensity"), 5)
  expect_equal(r$set$WDPA_PID, c("c", "f"))
  expect_equal(r$steps$n_left, c(6, 4, 3, 2)); expect_equal(r$steps$n_removed, c(NA, 2, 1, 1))
})

test_that("add_zscores standardises over the rows given, logging where asked", {
  d <- data.frame(a = c(1, 2, 3, 4), area = c(10, 100, 1000, 10000))
  z <- add_zscores(d, c(a_z = "a", log_area_z = "area"), log_vars = "log_area_z")
  expect_equal(mean(z$a_z), 0); expect_equal(stats::sd(z$a_z), 1)
  expect_equal(z$log_area_z, as.numeric(scale(log(d$area))))
  sc <- attr(z, "scaling")
  expect_equal(sc$mean[sc$variable == "log_area_z"], mean(log(d$area)))
  expect_equal(sc$log, c(FALSE, TRUE))
  expect_error(add_zscores(data.frame(a = c(1, NA)), c(a_z = "a")), "non-finite")
})

test_that("pa_coordinates returns a point inside each polygon, in the order asked", {
  sq <- function(x0, y0, w) sf::st_polygon(list(rbind(c(x0, y0), c(x0 + w, y0), c(x0 + w, y0 + w),
                                                      c(x0, y0 + w), c(x0, y0))))
  # A U shape: its centroid lies in the notch (outside), the point on surface inside
  u <- sf::st_polygon(list(rbind(c(10, 40), c(12, 40), c(12, 42), c(11.5, 42), c(11.5, 40.5),
                                 c(10.5, 40.5), c(10.5, 42), c(10, 42), c(10, 40))))
  p <- sf::st_sf(WDPA_PID = c("u", "s"), geometry = sf::st_sfc(u, sq(-60, -10, 1), crs = 4326))
  cen <- sf::st_centroid(sf::st_geometry(p)[1])
  expect_false(sf::st_within(cen, p[1, ], sparse = FALSE)[1, 1])
  xy <- pa_coordinates(p, c("s", "u"), CFG$coords_crs)
  expect_equal(xy$WDPA_PID, c("s", "u"))
  pts <- sf::st_as_sf(xy, coords = c("lon", "lat"), crs = 4326)
  inside <- sf::st_within(pts, p[match(xy$WDPA_PID, p$WDPA_PID), ], sparse = FALSE)
  expect_true(all(diag(inside)))
  expect_error(pa_coordinates(p, c("u", "zz"), CFG$coords_crs), "no polygon")
})

test_that("iucn_group maps categories to the config groups", {
  g <- iucn_group(c("Ia", "V", "Not Reported", "II", "xx"), CFG$iucn_groups)
  expect_equal(as.character(g), c("strict", "non-strict", "unknown", "strict", NA))
  expect_equal(levels(g), c("strict", "non-strict", "unknown"))
})

test_that("config covariates match the model formula", {
  expect_setequal(names(CFG$covariates), c(MODEL_COVARIATES, MODEL_HMI))
  expect_true(all(CFG$log_covariates %in% names(CFG$covariates)))
})
