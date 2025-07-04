---
title: Islandization of terrestrial protected areas
abstract: Recent global commitments to biodiversity conservation focus on reversing the loss of habitat surrounding protected areas (PAs) to safeguard landscape-scale ecological processes and the capacity for adaptation to rapid global change. Yet our ability to recognize and mitigate the ‘islandization’ of PAs is limited by a weak understanding of where and at what rate this process occurs. Here, we used satellite imagery to measure change in landscape pattern along the boundaries of 4,471 PAs representing all terrestrial biomes. Nearly half the world’s PAs showed accelerated islandization over a 23-year period (2001-2023). Surprisingly, PAs in grassland and shrubland biomes showed the greatest rates of islandization over time. These findings highlight the challenges and opportunities for utilizing PAs as the backbone of post-2020 initiatives for large-landscape conservation.
authors:
  - name: Amy Van Scoyoc
    affiliation: University of California, Berkeley
    orcid: 0000-0001-8638-935X
    email: avanscoyoc@berkeley.edu
  - name: Wenjing Xu
    affiliation: Senckenberg Biodiversity and Climate Research Centre
    orcid: 0000-0001-5657-2364
    email: wenjing.xu@senckenberg.de 
  - name: Carl Boettiger
    affiliation: Unversity of California, Berkeley
  - name: Justin Brashares
    affiliation: Unversity of California, Berkeley    
description: >
  Analysis of habitat continuity and islandization in protected areas using MODIS and geospatial methods.
keywords: protected area, boundary, habitat edge, earth engine, gradient
bibliography: references.bib
---

### Introduction

For more than half a century, conservation science has cautioned that the continued expansion of human activities will ultimately reduce our terrestrial protected areas (PAs) to habitat islands in a sea of development [@Gilpin1980; @Wilson1967]. Human settlement, land conversion, and resource extraction outside of PA boundaries (3–6), as well as habitat preservation, ecosystem recovery, and management practices within PAs (7, 8), may all serve to disrupt habitat continuity between PAs and their surrounding landscapes. This process of PA ‘islandization’ (1, 9), that is, the encirclement of PAs by dissimilar habitat, has been observed to reduce habitat connectivity, affecting the demography, genetics, and survival of isolated populations, rendering species, and large-landscape processes such as migrations, vulnerable to extinction (10–13).

Though well-trod in conservation science and planning, these concerns surrounding PA islandization have received new life in recent conservation initiatives such as the Convention on Biodiversity Aichi Target 11 which prioritized the integration of PAs into larger land and seascapes (14). More recently, the Kunming-Montreal Global Biodiversity Framework committed to restoring and supporting sustainable use of land surrounding PAs to retain wider ecosystem function and improve the adaptive potential of species facing rapid climate change (15). Yet, while several landmark studies have quantified forest loss surrounding PAs (16–20), there has been no general assessment of PA islandization across biomes, limiting our ability to measure progress toward the conservation of diverse habitat types beyond PA boundaries (15, 21, 22).

We quantified the islandization of terrestrial PAs across the world’s biomes by measuring change in landscape pattern encircling PA boundaries over a 20-year period (2001-2021). Our analysis included terrestrial PAs with an area greater than 200 km2 (n = 4,471 PAs). For each PA, we generated 10 km and 1 km diameter buffers around the boundary and removed regions covered by water using the maximum water extent from 1984 to 2021 of the Global Surface Water dataset. Annual median composites were created from MODIS/Terra Surface Reflectance imagery (MOD09A1, 500 m) for each year from 2001 to 2023. For each MODIS composite, we analyzed bands (1–4), as well as two widely used remote sensing indicies, the Normalized Difference Vegetation Index (NDVI, a measure of vegetation greenness based on red and near-infrared reflectance) and the Bare Soil Index (BSI, which distinguishes bare soil using blue, red, near-infrared, and shortwave infrared reflectance). To identify habitat edges, we computed the magnitude of the spectral gradient for each band and index using a 3×3 kernel. The magnitude gradient measures the difference in spectral values among neighboring pixels of a kernel, whereby higher values indicate greater local heterogeneity. In order to provide a standardized measure of habitat discontinuity at the boundary, we computed an 'edge index' as the ratio of the mean gradient value in the 1 km buffer to that of the 10 km buffer.


### Results


### Discussion


### References 


### Methods


##### Protected area data

We obtained data for protected area geometries using the June 2021 World Database on Protected Areas (58). Following other global protected area studies (4, 59), we removed PAs that were marine, lacked a reported area, did not include detailed geographic information (i.e. those represented as a single point), or with a “UNESCO-MAB Biosphere Reserve” designation. Following the WDPA best practice guidelines, we only included PAs with status of “designated”, “established”, or “inscribed”. 

From this terrestrial protected area geometry dataset, we selected PAs with an area larger than 200 km2. This enabled us to match the resolution of the satellite imagery to the PA size, and to include PAs that were wide enough to sample with transects. Restricting PA area to 200 km2 excluded 225,353 individuals PAs, most of which were in Europe, but only reduced the total area PA analyzed by 7.66%. We assigned biomes to each PA using the global ecoregion layer (61). When multiple biomes were present we retained the biome label with the largest area.

##### Geometry operations

To quantify habitat discontinuity at PA boundaries, we create a 10 km diameter buffer across each protected area boundary. We chose 10 km for the sampling transects to capture sharp habitat transitions at park boundaries (23) and to reflect a common standard for animal dispersal distances (24). 

To prevent temportal Each buffer was masked by  

We removed transects that that bisected other areas of the PA boundary in curved or narrow sections, and that fell within neighboring protected areas.

##### Satellite imagery

To quantify the habitat discontinuity along the 10 km boundary buffer, we created global annual median composites from MODIS/Terra Surface Reflectance 8-Day L3 Global 500-m SIN Grid (MOD09A1) imagery for each year, from 2001 to 2023. For each composite, we retained bands 1-4 that are primarily used for land surface monitoring. We calculated the mean value of the spectral gradient of each image, that is the difference in spectral value among neighboring pixels across a 3x3 kernel, where the higher gradient value indicates higher spectral heterogeneity across a kernel. represented a greater spectral difference among pixels in the 3x3 kernel (i.e., greater spectral heterogeneity), indicating higher landcover heterogeneity. We extracted the spectral gradient value of every transect point for each of the 20 MODIS composite images. 

We created global annual median composites from MODIS/Terra Surface Reflectance 8-Day L3 Global 500-m SIN Grid (MOD09A1) imagery for each year, from 2001 to 2023.  

Different spectral bands measure the intensity of different wavelengths of light, hence are suitable for detecting various landcover characteristics. For example, the red and near infrared bands (band 1 and 2 of MOD09A1) are used to compute the widely used Normalized Difference Vegetation Index (NDVI) as a proxy for vegetation greenness or coverage We also analzyed two indicies, Normalized Vegetation Difference Index (NDVI) and Bare Soil Index (BSI). 

Land cover spectral gradients were characterized using annual median composites of Moderate Resolution Imaging Spectroradiometer (MODIS) Terra MOD09A1 satellite imagery. Larger spectral gradient values represented a greater spectral difference among pixels in the 3x3 kernel (i.e., greater spectral heterogeneity), indicating higher landcover heterogeneity. We extracted the spectral gradient value of every transect point for each of the 20 MODIS composite images. 

Recognizing that habitat discontinuity is scale dependent (8), we carried out identical analyses for distances of 1 km.

Specifically, we combined the seven land bands (i.e., band 1-7) to derive a single-band spectral gradient image for each year (fig. S2) to summarize habitat characteristics important to all biomes (e.g., vegetation type and water content, anthropogenic features and soil features). 

We conducted a case-by-case identification, using a single-band summary to identify landscape pattern along PA boundaries worldwide (Fig. 1). 

##### Change detection and validation

To confirm whether 500 m was an appropriate spatial resolution to analyze spectral gradient at the PA level, we visually compared the gradient output from the 2020 annual composite 500 m MODIS data with 30 m Landsat-8 data – the two most widely used satellite imagery series with global coverage over our study period. The 500 m MODIS imagery had the most consistent data quality and coverage for this 20-year analysis compared to Landsat, which suffered the ETM+ Scan line Corrector failure that introduced data gaps (68). The 500 m MODIS imagery was also more adept than 30 m Landsat imagery at capturing the broad landcover patterns while reducing noise from fine-scale spectral heterogeneity, such as individual trees or grazing paddocks (fig. S2). We acknowledge that habitat continuity is highly scale-dependent (8). MODIS satellite imagery was appropriate for our study goal, which was to quantify global-scale islandization processes along the boundaries of the world’s largest PAs. However, for smaller PAs, associated with fine-scale habitat heterogeneity, analyses might be better suited to using remote sensing imagery with finer resolution, such as Landsat (30-meter), Sentinel (10 – 60-meter), or Planet (0.5-meter).