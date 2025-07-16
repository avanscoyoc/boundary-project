---
title: Islandization of terrestrial protected areas
authors:
  - name: Amy Van Scoyoc
    affiliation: University of California, Berkeley
    orcid: 0000-0001-8638-935X
    email: avanscoyoc@berkeley.edu
    equal_contributor: true
    corresponding: true

  - name: Wenjing Xu
    affiliation: Senckenberg Biodiversity and Climate Research Centre
    orcid: 0000-0001-5657-2364
    email: wenjing.xu@senckenberg.de 
    equal_contributor: true
    corresponding: true
  - name: Carl Boettiger
    affiliation: University of California, Berkeley
  - name: Justin Brashares
    affiliation: University of California, Berkeley
abstract: |
  Recent global commitments to biodiversity conservation focus on safeguarding habitat connectivity to preserve landscape-scale ecological processes and the capacity for adaptation to rapid global change. While many key studies have quantified changes in forest cover to highlight the potential isolation of protected areas via their edges, biome-wide assessments of edges at protected areas boundaries have been few to none. Here, we quantified the rate of change in edges along XXXX protected area boundaries over 23-year period (2001 - 2023). We achieved a comparative biome-wide global assessment by using 500m MODIS satellite imagery and a pixel-based approach to compute the contrast in spectral values along the 10km boundary of XXXX protected areas representing all terrestrial biomes for all visible bands and several remote sensing indices. Nearly half the world’s protected areas showed accelerated islandization over a 23-year period (2001-2023). Surprisingly, protected areas in grassland and shrubland biomes showed the greatest rates of islandization over time. These findings highlight the challenges and opportunities for utilizing protected areas as the backbone of post-2020 initiatives for large-landscape conservation.
keywords: [protected areas, islandization, habitat continuity, MODIS, geospatial analysis]
bibliography: references.bib
acknowledgements: We thank M.W. Brunson, C.E. Aslan, W. Ji, I. Dronova, A. Merenlender, and A. Middleton for their comments on this study and manuscript. Special thanks to the Middleton and Brashares lab groups at UC Berkeley, and anonymous reviewers for helpful feedback and edits. 
exports:
  - format: docx
  - format: pdf
    template: springer
    output: manuscript.pdf
    show_date: false
  - format: md
---

### Introduction

Differences in adjacent land cover or land use—referred to as ‘edges’—can fragment habitat and reduce landscape connectivity for plants and animals. Edge effects are known to disrupt ecological processes such as migration and dispersal, with consequences for population demography, gene flow, and long-term persistence. As a result, maintaining landscape connectivity is a central goal of global biodiversity conservation strategies. However, our understanding of where edges occur—and how they change over time—remains limited, particularly at the boundaries of protected areas.

Outside protected area borders, processes such as human settlement, land conversion, or resource extraction can fragment landscapes and disrupt ecological continuity. Within protected areas, management interventions like prescribed burns or ecological restoration can create edges. Together, these processes may lead to the 'islandization' of protected areas—where protected areas become functionally disconnected from the broader land system despite their formal designation. Most analyses of edge dynamics around protected areas focus on identifying edges along a single land cover type (e.g., forests) or are limited in geographic or temporal scale. Biome-wide assessments that systematically quantify changes in edge presence or intensity over time remain rare. 

Here, we quantified the rate of change in edges along XXXX protected area boundaries across the world’s biomes for a 23-year period (2001–2023). We conducted a comparative global assessment using 500 m MODIS satellite imagery and a pixel-based approach to measure gradient magnitude across multiple spectral bands and remote sensing indices. Gradient magnitude represents the intensity of spectral value differences among neighboring pixels within a 3×3 kernel, with higher values indicating greater local heterogeneity. To standardize edge detection, we computed an ‘edge index’ for each protected area as the ratio of the median gradient magnitude of a 1 km diameter buffer to that of a 10 km diameter buffer, for all protected areas larger than 200 km² (n = XXXX). By tracking changes in the edge index over time, our approach isolated active edge dynamics from static landscape features (e.g., mountain ranges, elevation zones), offering new insight into the pace and extent of islandization across the world’s protected landscapes.

### Methods

We conducted a global assessment of edge dynamics in terrestrial protected areas using 500 m MODIS satellite imagery and a pixel-based approach to quantify spatial heterogeneity across protected area boundaries. Specifically, we calculated the gradient magnitude, a measure of spectral contrast among neighboring pixels, for multiple spectral bands and remote sensing indices. Gradient magnitude was computed within a 3×3 kernel using:

$$
\text{Magnitude} = \sqrt{(\nabla_x I)^2 + (\nabla_y I)^2}
$$

Higher gradient values indicated greater local heterogeneity. Because abrupt transitions in land cover or land use create contrast in spectral values—“the gradient of [a] characteristic is steeper in the boundary than in either of the neighboring patches” [8]. Using this logic, when a protected area boundary aligns with an edge, we expect higher gradient magnitude on the boundary than in a larger reference area. Thus, to detect edges at protected area boundaries, we defined an ‘edge index’ for each protected area as the ratio of the median gradient magnitude within a 1 km buffer to that within a concentric 10 km buffer. 

$$
\text{Edge Index} = \frac{\overline{X}_{\text{boundary}}}{\overline{X}_{\text{buffer}}}
$$


##### *Protected Area Data*

Protected area geometries were obtained from the June 2021 release of the World Database on Protected Areas (WDPA) [@Hanson2022]. Consistent with prior global studies [@Jones2018,@Butchart2015], we excluded marine protected areas, protected areas lacking reported area or detailed geometry (i.e., points only), and those designated as “UNESCO-MAB Biosphere Reserves.” Only terrestrial protected areas classified as “designated,” “established,” or “inscribed” were retained, following WDPA best practices.

We limited our analysis to protected areas larger than 200 km² to ensure compatibility with the 500 m spatial resolution of MODIS data and to reduce the likelihood that a 10 km buffer would overlap itself in smaller areas. We also excluded protected areas in the upper quartile of the perimeter-to-area ratio to avoid long, narrow shapes where buffers might intersect within the same protected area. These filters resulted in XXXX protected areas for analysis, removing 225,353—primarily small protected areas in Europe—while reducing the total protected area analyzed by only 7.66%.


##### *Geometric Operations*

For each protected area, we generated concentric buffers of 1 km and 10 km diameter centered on the protected area boundary. The 1 km buffer captured fine-scale heterogeneity aligned with the administrative boundary, while the 10 km buffer captured the background landscape variability. We selected these distances to balance spatial precision with ecological relevance: 1 km was deemed suitable for detecting land cover transitions at the administrative line while accounting for any slight spatial imprecision in the WDPA dataset [@Hanson2022] and 10 km reflects an arbitrary but reasonable distance for monitoring protected area isolation and ecological differences. 

Since we were interested in terrestrial land surface dynamics, we excluded water features by removing buffered areas that overlapped with the maximum extent of surface water using the Global Surface Water dataset (1984–2021) [@Pekel2016]. For each year and each band/index, we calculated the median gradient magnitude within the 1 km and 10 km buffers and derived the edge index as their ratio.

To examine how edge dynamics varied by biome, we overlaid each protected area with the global terrestrial ecoregions map [@Dinerstein2017], and assigned each protected area the largest biome by area. We also extracted the mean human modification score from the global Human Modification dataset (gHM), which quantifies cumulative human impact at 1 km² resolution (CITE). These variables were ultimately used to analyze relationships between edge dynamics, ecological context, and anthropogenic modification.


##### *Satellite Imagery*

We used annual global composites from the MODIS/Terra Surface Reflectance 8-Day L3 product (MOD09A1) at 500 m resolution, spanning 2001 to 2023. For each year, we generated annual median composites of bands 1–4 and computed two spectral indices: the Normalized Difference Vegetation Index (NDVI) to represent vegetation greenness, and the Bare Soil Index (BSI) to reflect exposed soil. These bands and indices were selected to capture key land cover properties across biomes, including vegetation structure, soil exposure, and anthropogenic features.

$$
\mathrm{NDVI} = \frac{\mathrm{NIR} - \mathrm{RED}}{\mathrm{NIR} + \mathrm{RED}}
$$

$$
\mathrm{BSI} = \frac{(\mathrm{SWIR2} + \mathrm{RED}) - (\mathrm{NIR} + \mathrm{BLUE})}{(\mathrm{SWIR2} + \mathrm{RED}) + (\mathrm{NIR} + \mathrm{BLUE})}
$$

To validate the suitability of 500 m imagery, we visually compared MODIS-derived gradients from 2023 with those from 30 m Landsat-8 imagery (fig. S1). MODIS better captured broad-scale land cover transitions and was less sensitive to fine-scale spectral noise (e.g., individual trees, buildings, paddocks), while providing consistent coverage over the 23-year study period. Landsat data, by contrast, suffered from data gaps due to the ETM+ Scan Line Corrector failure [@Arvidson2006]. For the spatial scale of this study, which focused on protected areas greater than 200 km2, MODIS was most appropriate, though finer-resolution imagery (e.g., Landsat 30-meter, Sentinel 10 – 60-meter, Planet 0.5-meter) may be preferable for analyses that include small protected areas.

```{figure} ../images/resolutions.png
Fig. S1. Original protected area satellite imagery and spectral gradient calculation of a protected area. (A) Google satellite image; (B) 2020 annual composite MODIS image (RGB band combination 1-4-3); (C) 2020 annual composite Landsat-8 image (RGB band combination 4-3-2); (D) gradient image calculated from MODIS (b), (E) gradient image calculated from Landsat-8 (C). As shown in (D), MODIS 500-meter pixel size was most adept at reducing fine-scale heterogeneity while retaining broad-scale patterns. 
```


##### *Statistical analyses*

We used linear regression to estimate temporal trends in each protected area’s edge index from 2001 to 2023. Significance was determined using a threshold of p < 0.05, and we summarized the proportion of protected areas with significant trends by biome. Because landscape change is not always linear, we also calculated a 10-year rolling mean of the edge index to assess long-term dynamics. To quantify temporal variability, we computed a 10-year rolling standard deviation. Additionally, we conducted breakpoint analysis (CITE) to identify structural changes in edge dynamics and compared the number of breakpoints across biomes.

To evaluate underlying environmental correlates of observed edge patterns, we calculated the annual rate of change in NDVI and BSI for each protected area and assessed correlations with 2021 gHM values.

Last, to assess the contemporary state of edge dynamics, we calculated the proportion of protected areas with an edge index greater than 1 in the year 2023. All spatial analyses were performed using the `geemap` Python package (CITE), and all statistical analyses were conducted in Python (CITE).


### Results


### Discussion


### References 