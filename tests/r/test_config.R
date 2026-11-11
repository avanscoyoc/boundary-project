test_that("config settings are consistent", {
  expect_equal(CFG$n_span, diff(range(CFG$years)))
  expect_true(file.exists(file.path(CFG$root, "modules", "r", "config.R")))

  expect_true(CFG$ref$biome %in% CFG$biome_order)
  expect_true(CFG$ref$iucn %in% names(CFG$iucn_groups))
  expect_false(any(CFG$biome_order %in% CFG$exclude_biomes))

  cats <- unlist(CFG$iucn_groups, use.names = FALSE)
  expect_false(anyDuplicated(cats) > 0)

  expect_identical(CFG$transects[1], "all")
  expect_setequal(names(CFG$boot$seeds), c("biome", "weighted", "sen", "pooled"))
  expect_length(CFG$trend$levels, 3)

  cs <- CFG$case_studies
  expect_identical(names(cs), c("WDPA_PID", "label", "expect"))
  expect_false(anyDuplicated(cs$WDPA_PID) > 0)
  ok <- c(outer(CFG$trend$levels, CFG$trend$levels, paste, sep = "/"))
  expect_true(all(cs$expect %in% ok))
  expect_false(anyNA(cs))
})
