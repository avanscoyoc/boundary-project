"""Unit tests for modules/edge_metrics.py (script 04 metrics, biome groups, CSV names)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from modules.edge_metrics import _grouped_cohens_d, PT_COLS, recategorize_biome, find_asset_csvs


# Reference implementation of Cohen's d; pa_year_metrics computes the same values with
# grouped statistics (_grouped_cohens_d).
def cohens_d(x, y):
    """
    Cohen's d between two samples, with pooled standard deviation ((n-1) df).

    Returns NaN if either sample has < 2 values or the pooled SD is 0.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan

    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    if pooled_std == 0 or np.isnan(pooled_std):
        return np.nan

    return (np.mean(x) - np.mean(y)) / pooled_std


def test_grouped_cohens_d_matches_reference():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(60, 5)), columns=PT_COLS)
    df['pt_0'] += 1.0
    df['WDPA_PID'], df['year'] = 'A', 2010
    d = _grouped_cohens_d(df, ['WDPA_PID', 'year']).iloc[0]
    for name, col in {'D02': 'pt_2', 'D01': 'pt_1', 'D0m1': 'pt_m1', 'D0m2': 'pt_m2'}.items():
        assert np.isclose(d[name], cohens_d(df['pt_0'], df[col]))



def test_missing_transects():
    """Transects with a missing value in any year, as build_transect_years drops them (script 04 → 05)."""
    from modules.edge_metrics import missing_transects
    years = [2003, 2004, 2005]

    def pts(pid, tid, missing=(), water=(), n=5):
        rows = []
        for p in (-2, -1, 0, 1, 2)[:n]:
            rows.append({'WDPA_PID': pid, 'transectID': tid, 'pointID': p, 'max_extent': int(p in water),
                         **{str(y): (np.nan if (p, y) in missing else 1.0) for y in years}})
        return rows
    df = pd.DataFrame(pts('a', 1)                                                     # complete
                      + pts('a', 2, missing={(2, y) for y in years}, water={1, 2})     # +5 km at water, every year
                      + pts('b', 3, missing={(0, 2004)})                               # boundary point, one year
                      + pts('b', 4, missing={(-2, 2005), (1, 2005)})                   # both sides, one year
                      + pts('b', 5, missing={(0, 2003)}, n=4))                         # 4 points: skipped
    out = missing_transects(df.sample(frac=1, random_state=1), years)
    assert out[['WDPA_PID', 'transectID']].values.tolist() == [['a', 2], ['b', 3], ['b', 4]]
    assert out['n_missing_years'].tolist() == [3, 1, 1]
    assert out['water_any'].tolist() == [1, 0, 0]
    assert out['n_missing_point_years'].tolist() == [3, 1, 2]
    assert out['n_missing_point_years_water'].tolist() == [3, 0, 0]
    assert out[[f'miss_{c}' for c in PT_COLS]].values.tolist() == [[0, 0, 0, 0, 3], [0, 0, 1, 0, 0], [1, 0, 0, 1, 0]]


def test_recategorize_biome_all_resolve_biomes():
    groups = {
        "Tropical & Subtropical Moist Broadleaf Forests": "Tropical Forest",
        "Tropical & Subtropical Dry Broadleaf Forests": "Tropical Forest",
        "Tropical & Subtropical Coniferous Forests": "Tropical Forest",
        "Temperate Broadleaf & Mixed Forests": "Temperate Forest",
        "Temperate Conifer Forests": "Temperate Forest",
        "Boreal Forests/Taiga": "Boreal Forest",
        "Tropical & Subtropical Grasslands, Savannas & Shrublands": "Tropical Grassland",
        "Flooded Grasslands & Savannas": "Tropical Grassland",
        "Temperate Grasslands, Savannas & Shrublands": "Temperate & Montane Grassland",
        "Montane Grasslands & Shrublands": "Temperate & Montane Grassland",
        "Tundra": "Tundra",
        "Mediterranean Forests, Woodlands & Scrub": "Mediterranean",
        "Deserts & Xeric Shrublands": "Desert",
        "Mangroves": "Mangrove",
        "N/A": "Rock & Ice",
    }
    for resolve, group in groups.items():
        assert recategorize_biome(resolve) == group, resolve
    assert len(set(groups.values()) - {"Mangrove", "Rock & Ice"}) == 8
    for missing in ["Unknown", "None", "", None, np.nan]:
        assert recategorize_biome(missing) is None


def test_find_asset_csvs_only_this_prefix(tmp_path):
    names = ["ndvi_gradient_000_part_0.csv", "ndvi_gradient_000_part_1_01.csv",
             "ndvi_gradient_001_part_0.csv", "ndvi_gradient_000_part_0 (1).csv",
             "ndvi_gradient_displaced_000_part_0.csv", "other_000_part_0.csv"]
    for n in names:
        (tmp_path / n).write_text("x")
    groups, other = find_asset_csvs(tmp_path, "ndvi_gradient_")
    assert {a: [f.name for f in fs] for a, fs in groups.items()} == {
        "000": ["ndvi_gradient_000_part_0.csv", "ndvi_gradient_000_part_1_01.csv"],
        "001": ["ndvi_gradient_001_part_0.csv"]}
    assert sorted(other) == sorted(names[3:])
    groups, other = find_asset_csvs(tmp_path, "ndvi_gradient_displaced_")
    assert list(groups) == ["000"] and len(other) == 5
