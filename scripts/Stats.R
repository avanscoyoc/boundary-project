library(tidyverse)

# ── 1. Load raw data ──────────────────────────────────────────
df_raw <- read_csv("./data/wdpa_df_ndvi_trends.csv")

# ── 2. Keep only columns needed ───────────────────────────────
# Drop: admin/metadata cols not used in analysis
# Keep: identifiers, metrics, predictors, biome, python trends (for comparison)

df <- df_raw |>
  select(
    # Identifiers
    WDPA_PID, NAME, ISO3, BIOME_NAME,
    
    # Time
    year,
    
    # Core metrics (annual, per PA)
    n_trnst, edge_extent, edge_intensity,
    
    # Raw distance columns (in case needed later)
    D02, D01, D0m1, D0m2,
    
    # Predictors
    IUCN_CAT, STATUS_YR, AREA_DISSO,
    gHM_mean, elevation_mean, slope_mean, water_extent_pct,
    GOV_TYPE,
    
    # Python-derived trends (keep for comparison only)
    trend_extent, slope_extent, p_value_extent,
    trend_intensity, slope_intensity, p_value_intensity
  )

# ── 3. Basic cleaning ─────────────────────────────────────────

# Collapse IUCN to strict / non-strict / unknown. Commonly done in other PA papers.
df <- df |>
  mutate(
    IUCN_strict = case_when(
      IUCN_CAT %in% c("Ia", "Ib", "II")                          ~ "strict",
      IUCN_CAT %in% c("III", "IV", "V", "VI")                    ~ "non-strict",
      IUCN_CAT %in% c("Not Applicable", "Not Assigned", 
                      "Not Reported")                             ~ "unknown",
      TRUE ~ NA_character_  # catches anything unexpected
    ),
    IUCN_strict = factor(IUCN_strict, 
                         levels = c("non-strict", "strict", "unknown"))
  ) 
print(table(df$IUCN_strict, useNA = "ifany"))

# Drop biomes with too few PAs or non-terrestrial
exclude_biomes <- c("Mangrove", "Rock & Ice")
df <- df |> 
  filter(
    !BIOME_NAME %in% exclude_biomes,
    !is.na(BIOME_NAME)
  )

# Confirm structure
glimpse(df)
cat("Years:", sort(unique(df$year)), "\n")
cat("PAs:", n_distinct(df$WDPA_PID), "\n")
cat("Biomes:", unique(df$BIOME_NAME), "\n")
cat("IUCN breakdown:\n")
print(table(df$IUCN_strict, useNA = "ifany"))

# set "non-strict" as the reference level since it's the most common and interpretable baseline
df <- df |>
  mutate(
    IUCN_strict = factor(IUCN_strict, 
                         levels = c("non-strict", "strict", "unknown"))
  )

# ── 3. Trend model ───────────────────────────────
library(broom)

# ── Per-PA OLS trend model ────────────────────────────────────
# edge_extent ~ year, separately for each WDPA_PID

pa_trends <- df |>
  group_by(WDPA_PID) |>
  filter(sum(!is.na(edge_extent)) >= 5) |>  # need enough years to fit a line. only PA 166904 had only 1 year of data.
  group_modify(~tidy(lm(edge_extent ~ year, data = .x))) |>
  ungroup() |>
  filter(term == "year") |>
  select(WDPA_PID, slope = estimate, se = std.error, 
         statistic, p_value = p.value)

# ── FDR correction ────────────────────────────────────────────
# FDR (False Discovery Rate) correction addresses the multiple testing problem. 
# When you run thousands of simultaneous hypothesis tests — one OLS per PA — you 
# expect many false positives by chance alone.
# At FDR < 0.05, you're saying: among all PAs I classify as having a significant 
# trend, I accept that up to 5% of them are wrong. This is a much more reasonable 
# trade-off when you have thousands of tests and genuinely expect many true signals.
pa_trends <- pa_trends |>
  mutate(
    p_fdr = p.adjust(p_value, method = "BH"),
    trend = case_when(
      slope > 0 & p_fdr < 0.05 ~ "increasing",
      slope < 0 & p_fdr < 0.05 ~ "decreasing",
      TRUE                      ~ "stable"
    ),
    trend = factor(trend, levels = c("increasing", "stable", "decreasing"))
  )

# ── Quick summary ─────────────────────────────────────────────
cat("Trend classification (R, FDR-corrected):\n")
print(table(pa_trends$trend))
cat("\nPython comparison:\n")
py_compare <- df |> 
  distinct(WDPA_PID, trend_extent) |>
  left_join(pa_trends |> select(WDPA_PID, trend), by = "WDPA_PID")
print(table(py = py_compare$trend_extent, r = py_compare$trend, useNA = "ifany"))

# prep data for mix model 
# ── PA-level dataset for cross-sectional model ────────────────
# Average time-varying columns across years, keep static columns

df_pa <- df |>
  group_by(WDPA_PID) |>
  summarise(
    # Static identifiers - use first non-NA value
    NAME             = first(na.omit(NAME)),
    ISO3             = first(na.omit(ISO3)),
    BIOME_NAME       = first(na.omit(BIOME_NAME)),
    IUCN_strict      = first(na.omit(IUCN_strict)),
    STATUS_YR        = first(na.omit(STATUS_YR)),
    AREA_DISSO       = first(na.omit(AREA_DISSO)),
    gHM_mean         = first(na.omit(gHM_mean)),
    elevation_mean   = first(na.omit(elevation_mean)),
    slope_mean       = first(na.omit(slope_mean)),
    water_extent_pct = first(na.omit(water_extent_pct)),
    GOV_TYPE         = first(na.omit(GOV_TYPE)),
    
    # Average metrics across years
    edge_extent_mean    = mean(edge_extent, na.rm = TRUE),
    edge_intensity_mean = mean(edge_intensity, na.rm = TRUE),
    n_years             = sum(!is.na(edge_extent))
  ) |>
  ungroup() |>
  left_join(pa_trends |> select(WDPA_PID, slope, se, p_value, p_fdr, trend),
            by = "WDPA_PID") |>
  filter(
    !BIOME_NAME %in% c("Mangrove", "Rock & Ice"),
    !is.na(BIOME_NAME)
  )

df_pa |>
  select(edge_extent_mean, STATUS_YR, AREA_DISSO, gHM_mean,
         elevation_mean, slope_mean, water_extent_pct, IUCN_strict) |>
  summarise(across(everything(), ~sum(is.na(.))))

cat("PAs remaining:", nrow(df_pa), "\n")
print(table(df_pa$BIOME_NAME, useNA = "ifany"))

cat("PA-level dataset rows:", nrow(df_pa), "\n")
glimpse(df_pa)

# ── 4. PA-level mixed model. Which variable correlates with edge extent?  ───────────────────────────────
library(car)  # for VIF

# ── Z-score continuous predictors ─────────────────────────────
df_pa <- df_pa |>
  mutate(
    STATUS_YR_z        = scale(STATUS_YR),
    AREA_DISSO_z       = scale(log(AREA_DISSO)),
    gHM_mean_z         = scale(gHM_mean),
    elevation_mean_z   = scale(elevation_mean),
    slope_mean_z       = scale(slope_mean),
    water_extent_pct_z = scale(water_extent_pct)
  )

m_extent <- lm(
  edge_extent_mean ~ STATUS_YR_z + AREA_DISSO_z + gHM_mean_z +
    elevation_mean_z + slope_mean_z + water_extent_pct_z + IUCN_strict,
  data = df_pa
)

summary(m_extent)

# ── 5. biome aggregation. mean slope per biome (with bootstrap CI) and the % increasing/decreasing per biome: ───────────────────────────────
library(boot)

# ── 5.1. % increasing / stable / decreasing per biome ──────────
biome_trend_pct <- df_pa |>
  filter(!is.na(trend)) |>
  group_by(BIOME_NAME, trend) |>
  summarise(n = n(), .groups = "drop") |>
  group_by(BIOME_NAME) |>
  mutate(pct = n / sum(n) * 100) |>
  ungroup()

print(biome_trend_pct)

# ── 5.2. Mean slope per biome with bootstrap CI ─────────────────
boot_mean <- function(data, i) mean(data[i], na.rm = TRUE)

biome_slope <- df_pa |>
  filter(!is.na(slope), !is.na(BIOME_NAME)) |>
  group_by(BIOME_NAME) |>
  summarise(
    n_pa       = n(),
    mean_slope = mean(slope, na.rm = TRUE),
    ci         = list(boot.ci(boot(slope, boot_mean, R = 1000), 
                              type = "perc")$percent[4:5]),
    .groups    = "drop"
  ) |>
  mutate(
    ci_lo = map_dbl(ci, 1),
    ci_hi = map_dbl(ci, 2)
  ) |>
  select(-ci) |>
  arrange(desc(mean_slope))

print(biome_slope)


# ═══════════════════════════════════════════════════════════════
# FIGURE: Biome-level edge extent and slope (still developing)
# ═══════════════════════════════════════════════════════════════
library(tidyverse)
library(patchwork)
library(dunn.test)

# ── Constants ─────────────────────────────────────────────────
biome_order <- biome_slope |>
  arrange(desc(mean_slope)) |>
  pull(BIOME_NAME)

biome_colors <- c(
  "Grassland & Shrubland" = "#E07B39",
  "Tropical Forest"       = "#2E8B57",
  "Temperate Forest"      = "#5B8DB8",
  "Boreal Forest"         = "#7B68AA",
  "Desert"                = "#C4A35A",
  "Tundra"                = "#6AACB8"
)

dodge_width <- 0.6  # shared dodge width across layers

# ── Pairwise tests ────────────────────────────────────────────
kruskal.test(edge_extent_mean ~ BIOME_NAME, data = df_pa)
dunn_extent <- dunn.test(
  df_pa$edge_extent_mean,
  df_pa$BIOME_NAME,
  method = "bh", altp = TRUE
)

kruskal.test(slope ~ BIOME_NAME, data = df_pa |> filter(!is.na(slope)))
dunn_slope <- dunn.test(
  df_pa$slope[!is.na(df_pa$slope)],
  df_pa$BIOME_NAME[!is.na(df_pa$slope)],
  method = "bh", altp = TRUE
)

# ── CLD labels (manual from Dunn results) ─────────────────────
cld_labels_extent <- tibble(
  BIOME_NAME = c("Grassland & Shrubland", "Tropical Forest",
                 "Temperate Forest", "Boreal Forest",
                 "Desert", "Tundra"),
  cld_extent = c("a", "a", "ab", "b", "b", "b")
)

cld_labels_slope <- tibble(
  BIOME_NAME = c("Grassland & Shrubland", "Tropical Forest",
                 "Temperate Forest", "Boreal Forest",
                 "Desert", "Tundra"),
  cld_slope  = c("a", "ab", "ab", "ab", "ab", "b")
)

# ── Biome summary ─────────────────────────────────────────────
fig_biome <- df_pa |>
  filter(!is.na(BIOME_NAME), !is.na(edge_extent_mean)) |>
  group_by(BIOME_NAME) |>
  summarise(
    n_pa       = n(),
    biome_mean = mean(edge_extent_mean, na.rm = TRUE),
    ci_lo      = biome_mean - 1.96 * sd(edge_extent_mean, na.rm = TRUE) / sqrt(n()),
    ci_hi      = biome_mean + 1.96 * sd(edge_extent_mean, na.rm = TRUE) / sqrt(n()),
    .groups    = "drop"
  ) |>
  left_join(biome_slope |> select(BIOME_NAME, mean_slope,
                                  ci_lo_slope = ci_lo,
                                  ci_hi_slope = ci_hi),
            by = "BIOME_NAME") |>
  left_join(cld_labels_extent, by = "BIOME_NAME") |>
  left_join(cld_labels_slope,  by = "BIOME_NAME") |>
  mutate(
    BIOME_NAME  = factor(BIOME_NAME, levels = biome_order),
    biome_label = paste0(BIOME_NAME, "\n(n=", n_pa, ")")
  ) |>
  arrange(BIOME_NAME)

# ── PA-level data ─────────────────────────────────────────────
# Clean rebuild of fig_pa
fig_pa <- df_pa |>
  filter(!is.na(BIOME_NAME), !is.na(edge_extent_mean), !is.na(AREA_DISSO)) |>
  mutate(
    area_km2   = AREA_DISSO / 1e6,
    BIOME_NAME = factor(BIOME_NAME, levels = biome_order)
  ) |>
  left_join(
    fig_biome |> select(BIOME_NAME, biome_label),
    by = "BIOME_NAME"
  ) |>
  mutate(biome_label = factor(biome_label, levels = label_order))

# Verify
cat("biome_label in fig_pa:", "biome_label" %in% names(fig_pa), "\n")
cat("rows:", nrow(fig_pa), "\n")
head(fig_pa$biome_label)

# These two are still needed
label_order <- fig_biome |> arrange(BIOME_NAME) |> pull(biome_label)
fig_biome$biome_label <- factor(fig_biome$biome_label, levels = label_order)

# This is still needed
global_mean_extent <- mean(df_pa$edge_extent_mean, na.rm = TRUE)

# ── Panel A ───────────────────────────────────────────────────
pA <- ggplot() +
  geom_hline(yintercept = global_mean_extent,
             linetype = "dashed", color = "darkgreen", linewidth = 0.7) +
  geom_point(
    data = fig_pa,
    aes(x = biome_label, y = edge_extent_mean, 
        size = area_km2, color = BIOME_NAME),
    alpha = 0.2, shape = 16,
    position = position_jitter(width = 0.2, height = 0, seed = 42)
  ) +
  geom_pointrange(
    data = fig_biome,
    aes(x = biome_label, y = biome_mean, ymin = ci_lo, ymax = ci_hi),
    color = "grey20", size = 0.8, linewidth = 1.2,
    shape = 18, fatten = 6,
    position = position_nudge(x = 0.2)
  ) +
  geom_text(
    data = fig_biome,
    aes(x = biome_label, y = 0.97, label = cld_extent),
    size = 4, fontface = "bold", color = "grey20"
  ) +
  scale_color_manual(values = biome_colors) +
  scale_size_continuous(
    name   = "PA area (km²)",
    range  = c(0.5, 10),
    breaks = c(100, 10000, 100000),
    labels = c("100", "10,000", "100,000")
  ) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent_format()) +
  labs(x = NULL, y = "Edge extent (mean)") +
  theme_classic(base_size = 13) +
  theme(
    axis.text.x        = element_text(size = 9),
    panel.grid.major.y = element_line(color = "grey92"),
    legend.position    = "right"
  ) +
  guides(color = "none")

# ── Panel B ───────────────────────────────────────────────────
pB <- ggplot() +
  geom_hline(yintercept = 0, linetype = "dashed",
             color = "darkgreen", linewidth = 0.7) +
  geom_point(
    data = fig_pa |> filter(trend == "stable"),
    aes(x = biome_label, y = slope * 1000, size = area_km2),
    color = "grey75", alpha = 0.2, shape = 16,
    position = position_jitter(width = 0.2, height = 0, seed = 42)
  ) +
  geom_point(
    data = fig_pa |> filter(trend != "stable"),
    aes(x = biome_label, y = slope * 1000, 
        size = area_km2, color = BIOME_NAME),
    alpha = 0.6, shape = 16,
    position = position_jitter(width = 0.2, height = 0, seed = 42)
  ) +
  geom_pointrange(
    data = fig_biome,
    aes(x = biome_label, y = mean_slope * 1000,
        ymin = ci_lo_slope * 1000, ymax = ci_hi_slope * 1000,
        color = BIOME_NAME),
    size = 0.8, linewidth = 1.2, shape = 18, fatten = 6,
    position = position_nudge(x = 0.2)
  ) +
  geom_text(
    data = fig_biome,
    aes(x = biome_label,
        y = max(fig_pa$slope * 1000, na.rm = TRUE) * 1.05,
        label = cld_slope),
    size = 4, fontface = "bold", color = "grey20"
  ) +
  scale_color_manual(values = biome_colors) +
  scale_size_continuous(
    name   = "PA area (km²)",
    range  = c(0.5, 8),
    breaks = c(100, 10000, 100000),
    labels = c("100", "10,000", "100,000")
  ) +
  labs(
    x       = NULL,
    y       = "Slope (×10⁻³ per year)",
    caption = paste0(
      "Bubbles = PA-level estimates; diamond = biome mean ± 95% bootstrap CI\n",
      "Panel B: colored = significant trend (FDR q<0.05), grey = stable\n",
      "Letters = Dunn test pairwise (BH corrected); shared letter = not significantly different"
    )
  ) +
  theme_classic(base_size = 13) +
  theme(
    axis.text.x        = element_text(size = 9),
    panel.grid.major.y = element_line(color = "grey92"),
    plot.caption       = element_text(size = 8, color = "grey50"),
    legend.position    = "none"
  )

# ── Combine ───────────────────────────────────────────────────
p_combined <- pA / pB +
  plot_layout(heights = c(2, 1.5), guides = "collect") +
  plot_annotation(tag_levels = "A") &
  theme(legend.position = "right")

print(p_combined)
# ggsave("./outputs/fig_biome_combined.pdf", p_combined,
#        width = 11, height = 11, dpi = 300)