import ee


def load_protected_area(name):
    """Load a protected area by name from the WDPA dataset"""
    protected_areas = ee.FeatureCollection("WCMC/WDPA/202106/polygons")

    # Define filters
    marine_filter = ee.Filter.eq("MARINE", "0")
    not_mpa_filter = ee.Filter.neq("DESIG_ENG", "Marine Protected Area")
    status_filter = ee.Filter.inList("STATUS", ["Designated", "Established", "Inscribed"])
    designation_filter = ee.Filter.neq("DESIG_ENG", "UNESCO-MAB Biosphere Reserve")
    area_filter = ee.Filter.gte("GIS_AREA", 200)

    excluded_pids = ["555655917", "555656005", "555656013", "555665477", "555656021",
                    "555665485", "555556142", "187", "555703455", "555563456", "15894"]
    pids_filter = ee.Filter.inList("WDPA_PID", excluded_pids).Not()

    # Combine all filters
    combined_filter = ee.Filter.And(
        marine_filter,
        not_mpa_filter,
        status_filter,
        designation_filter,
        pids_filter,
        area_filter
    )

    # Filter and get specific protected area
    data = protected_areas.filter(combined_filter)
    pa = data.filter(ee.Filter.eq('ORIG_NAME', name)).first()
    
    return pa