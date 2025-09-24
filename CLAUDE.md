# Protected Area Edge Effect Analysis with Google Earth Engine

## Overview

This codebase performs large-scale edge effect analysis on protected areas using Google Earth Engine (GEE). It creates buffer zones around protected areas and analyzes vegetation gradients to understand edge effects in conservation areas.

## Core Methodology

### Buffer Zone Creation
- **Inner Zone**: 1km buffer ring around protected area boundary
- **Outer Zone**: 5km buffer ring around protected area boundary MINUS the inner zone
- Each zone is tagged with `zone: "inner"` or `zone: "outer"` property

### Data Sources
- **Protected Areas**: `WCMC/WDPA/current/polygons` (World Database on Protected Areas)
- **Human Modification**: `CSP/HM/GlobalHumanModification` (static baseline)
- **Vegetation**: `MODIS/006/MOD13Q1` NDVI (annual composites 2001-2023)
- **Water Mask**: `JRC/GSW1_0/GlobalSurfaceWater` max_extent (excluded from analysis)
- **Ecoregions**: `RESOLVE/ECOREGIONS/2017` (for biome classification)

### Analysis Pipeline
1. **Static Analysis**: Extract protected area properties, biome classification, and human modification baseline for all wdpa_ids
2. **Temporal Analysis**: Calculate annual NDVI gradient statistics (mean/std) for each zone for all wdpa_ids and then another function to iterate this for each year
3. **Integration**: Combine static and temporal data into final dataset

## GEE Best Practices Implementation

### Efficient Batching Strategy
- **Batch Size**: 500 parks per batch to stay within GEE computation limits
- **Parallel Processing**: Separate static and temporal analyses to maximize throughput
- **Memory Management**: Use `tileScale=4` for large computations

### Optimized Operations
- **Single Property Setting**: Use `ee.Dictionary()` and batch property assignment instead of multiple `.set()` calls
- **Combined Reducers**: Use `.combine()` to calculate mean and standard deviation in one operation with `.setOutputs()`
- **Efficient Filtering**: Use `ee.Filter.inList()` for batch processing multiple WDPA_PIDs
- **Geometry Optimization**: Apply `maxError` parameters to balance precision and performance


## Scaling Considerations

### Current Scale
- **6,500 protected areas** across global dataset
- **23 years** of temporal analysis (2001-2023)
- **Total operations**: ~300,000 individual zone-year combinations

### Performance Optimizations
- **Reducer Efficiency**: Combined mean/stdDev calculations minimize computation
- **Export Strategy**: Direct-to-Cloud Storage exports avoid memory bottlenecks
- **Task Management**: Automated monitoring prevents manual oversight requirements

## Output Structure

### Final Dataset Columns
- **Identifiers**: `WDPA_PID`, `zone`, `year`
- **Protected Area Properties**: `ORIG_NAME`, `GOV_TYPE`, `OWN_TYPE`, `STATUS_YR`, `IUCN_CAT`, `GIS_AREA`
- **Environmental Context**: `BIOME_NAME`
- **Human Modification**: `gHM_mean`, `gHM_stdDev`
- **Edge Effects**: `gradient_mean`, `gradient_stdDev`

This architecture ensures efficient processing of large-scale conservation datasets while maintaining scientific rigor and computational efficiency within Google Earth Engine's constraints.