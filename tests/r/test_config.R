test_that("config settings are consistent", {
  expect_equal(CFG$n_span, diff(range(CFG$years)))
  expect_true(file.exists(file.path(CFG$root, "modules", "r", "config.R")))

  for (fam in names(CFG$ref)) {
    expect_true(CFG$ref[[fam]]$biome %in% CFG$biome_order, info = fam)
    expect_true(CFG$ref[[fam]]$iucn %in% names(CFG$iucn_groups), info = fam)
  }
  expect_false(any(CFG$biome_order %in% CFG$exclude_biomes))

  cats <- unlist(CFG$iucn_groups, use.names = FALSE)
  expect_false(anyDuplicated(cats) > 0)

  expect_identical(CFG$transects[1], "all")
  expect_setequal(names(CFG$boot$seeds), c("biome", "weighted", "sen", "pooled"))
  expect_length(CFG$trend$levels, 3)
})
