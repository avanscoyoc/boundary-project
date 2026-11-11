"""
Edge metric functions (script 04).

- find_asset_csvs: group the Earth Engine CSV parts by asset number
- read_asset_csvs: read one asset's CSV parts
- build_transect_years: one row per complete transect-year, with edge flag and
  the water / other-PA flags
- pa_year_metrics: PA-year edge extent, Cohen's d values, and edge intensity, for
  four transect subsets (all, no water, no other-PA, neither)
- add_trends: per-PA OLS trends of every extent and intensity metric, with
  Benjamini-Hochberg adjusted p-values
- cohens_d, recategorize_biome, fit_trend, bh_adjust

Every PA lies within a single asset (script 02 writes whole PAs per upload file),
so each asset is processed on its own; memory is bounded by one asset.
"""

import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

POINT_IDS = [-2, -1, 0, 1, 2]
PT_COLS = ['pt_m2', 'pt_m1', 'pt_0', 'pt_1', 'pt_2']

# Cohen's d columns: boundary point (pt_0) vs each other position
D_PAIRS = {'D02': 'pt_2', 'D01': 'pt_1', 'D0m1': 'pt_m1', 'D0m2': 'pt_m2'}

# Metric versions: column suffix -> flags whose transects are excluded
VERSIONS = {
    '': (),
    '_nowater': ('water_extent',),
    '_nootherpa': ('in_other_pa',),
    '_noflags': ('water_extent', 'in_other_pa'),
}

# Earth Engine CSV part names: {prefix}_{asset}_part_{i}[_{retry}].csv
# (old runs: ..._{asset}_chunk_{i}.csv)
_CSV_NAME = re.compile(r'_(\d{3})_(?:part|chunk)_(\d+)(?:_(\d{2}))?\.csv$')


def cohens_d(x, y):
    """
    Cohen's d between two samples, with pooled standard deviation ((n-1) df).

    Reference implementation; pa_year_metrics computes the same values with
    grouped statistics. Returns NaN if either sample has < 2 values or the
    pooled SD is 0.
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


def find_asset_csvs(csv_dir):
    """
    Group the Earth Engine CSV parts in ``csv_dir`` by asset number.

    Returns
    -------
    tuple of (dict, list)
        asset number ('000', ...) -> sorted list of CSV paths (including retry
        halves), and the names of CSVs whose names don't match the pattern
        (e.g. Drive duplicates like 'name (1).csv').
    """
    groups, unrecognized = defaultdict(list), []
    for f in sorted(Path(csv_dir).glob('*.csv')):
        m = _CSV_NAME.search(f.name)
        if m:
            groups[m.group(1)].append(f)
        else:
            unrecognized.append(f.name)
    return dict(sorted(groups.items())), unrecognized


def read_asset_csvs(files, years):
    """
    Read and stack the CSV parts of one asset.

    Gradient values of -9999 (masked pixels) become NaN. Values are float32.
    """
    year_cols = [str(y) for y in years]
    dtypes = {'WDPA_PID': str, 'transectID': 'float64', 'pointID': 'float64',
              'max_extent': 'float32', 'gHM': 'float32', 'elevation': 'float32',
              'terrain_slope': 'float32', **{y: 'float32' for y in year_cols}}
    df = pd.concat([pd.read_csv(f, dtype=dtypes, usecols=list(dtypes)) for f in files],
                   ignore_index=True)
    df['transectID'] = df['transectID'].astype('int32')
    df['pointID'] = df['pointID'].astype('int8')
    df[year_cols] = df[year_cols].mask(df[year_cols] == -9999)
    return df


def build_transect_years(df, years, flags=None):
    """
    Reshape one asset's points into one row per complete transect-year.

    A transect needs all 5 points in the export, and a transect-year needs a
    gradient value at all 5 points; incomplete ones are dropped (counted).

    Parameters
    ----------
    df : DataFrame
        Output of read_asset_csvs.
    years : sequence of int
        Years in the order of the CSV columns.
    flags : DataFrame, optional
        WDPA_PID, transectID, in_other_pa (from transect_points.parquet).
        Missing -> in_other_pa = 0.

    Returns
    -------
    tuple of (DataFrame, dict)
        - transect-years: WDPA_PID, transectID, year, pt_m2 … pt_2, edge,
          water_extent, in_other_pa, gHM_mean_outer, elevation_pt0, terrain_slope_pt0
        - counts: transects_received, transects_missing_points,
          complete_per_year {year: n}

    Notes
    -----
    edge = 1 when pt_0 > min(pt_1, pt_2) and pt_0 > min(pt_m1, pt_m2).
    water_extent = 1 when any of the 5 points has JRC max_extent == 1.
    """
    key = ['WDPA_PID', 'transectID', 'pointID']
    n_dup = int(df.duplicated(key).sum())
    if n_dup:
        raise ValueError(f"{n_dup:,} duplicate points in the CSVs (was a part downloaded twice?)")

    df = df.sort_values(key, kind='stable')
    n_points = df.groupby(['WDPA_PID', 'transectID'], sort=False)['pointID'].transform('size')
    transects_received = int(df.groupby(['WDPA_PID', 'transectID'], sort=False).ngroups)
    df = df[n_points.to_numpy() == len(POINT_IDS)]
    n_tr = len(df) // len(POINT_IDS)
    year_cols = [str(y) for y in years]
    years = np.asarray(years, dtype='int16')

    # Per-transect values (static layers) from the reshaped points
    def per_point(col):
        return df[col].to_numpy().reshape(n_tr, len(POINT_IDS))

    ids = df.iloc[::len(POINT_IDS)][['WDPA_PID', 'transectID']].reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN outer gHM -> NaN
        transects = ids.assign(
            water_extent=(per_point('max_extent') == 1).any(axis=1).astype('int8'),
            gHM_mean_outer=np.nanmean(per_point('gHM')[:, 3:5], axis=1),
            elevation_pt0=per_point('elevation')[:, 2],
            terrain_slope_pt0=per_point('terrain_slope')[:, 2],
        )
    if flags is not None and len(flags):
        transects = transects.merge(flags[['WDPA_PID', 'transectID', 'in_other_pa']],
                                    on=['WDPA_PID', 'transectID'], how='left')
        transects['in_other_pa'] = transects['in_other_pa'].fillna(0).astype('int8')
    else:
        transects['in_other_pa'] = np.int8(0)

    # Transect-years: (n_tr, 5, n_years) -> (n_tr * n_years, 5)
    vals = df[year_cols].to_numpy(dtype='float32').reshape(n_tr, len(POINT_IDS), len(years))
    vals = vals.transpose(0, 2, 1).reshape(-1, len(POINT_IDS))
    complete = ~np.isnan(vals).any(axis=1)
    t_idx = np.repeat(np.arange(n_tr), len(years))[complete]

    ty = transects.iloc[t_idx][['WDPA_PID', 'transectID']].reset_index(drop=True)
    ty['year'] = np.tile(years, n_tr)[complete]
    ty[PT_COLS] = vals[complete]
    p = ty[PT_COLS]
    ty['edge'] = ((p['pt_0'] > p[['pt_1', 'pt_2']].min(axis=1)) &
                  (p['pt_0'] > p[['pt_m1', 'pt_m2']].min(axis=1))).astype('int8')
    for col in ['water_extent', 'in_other_pa', 'gHM_mean_outer', 'elevation_pt0', 'terrain_slope_pt0']:
        ty[col] = transects[col].to_numpy()[t_idx]

    counts = {
        'transects_received': transects_received,
        'transects_missing_points': transects_received - n_tr,
        'complete_per_year': ty['year'].value_counts().sort_index().to_dict(),
    }
    return ty, counts


def missing_transects(df, years):
    """
    Transects with no value at one or more of their 5 points in at least one year: the
    transect-years that build_transect_years drops. Written by script 04 for the missing-data
    counts of R script 05 (which keeps the PAs of the analysis set).

    Parameters
    ----------
    df : DataFrame
        Output of read_asset_csvs. Transects without all 5 points in the export are skipped
        (build_transect_years counts them).
    years : sequence of int
        Years in the order of the CSV columns.

    Returns
    -------
    DataFrame
        One row per such transect: WDPA_PID, transectID, n_missing_years (years with a missing
        value), water_any (1 if any point lies within the JRC maximum water extent),
        n_missing_point_years, n_missing_point_years_water (at a point with water), and the
        missing point-years per position: miss_pt_m2, miss_pt_m1, miss_pt_0, miss_pt_1, miss_pt_2.
    """
    key = ['WDPA_PID', 'transectID', 'pointID']
    df = df.sort_values(key, kind='stable')
    n_points = df.groupby(['WDPA_PID', 'transectID'], sort=False)['pointID'].transform('size')
    df = df[n_points.to_numpy() == len(POINT_IDS)]
    n_tr = len(df) // len(POINT_IDS)
    year_cols = [str(y) for y in years]
    miss = np.isnan(df[year_cols].to_numpy(dtype='float32')).reshape(n_tr, len(POINT_IDS), len(years))
    water = (df['max_extent'].to_numpy() == 1).reshape(n_tr, len(POINT_IDS))
    n_years = miss.any(axis=1).sum(axis=1)
    sel = n_years > 0
    out = df.iloc[::len(POINT_IDS)][['WDPA_PID', 'transectID']].reset_index(drop=True)[sel].reset_index(drop=True)
    out['n_missing_years'] = n_years[sel].astype('int16')
    out['water_any'] = water.any(axis=1)[sel].astype('int8')
    out['n_missing_point_years'] = miss.sum(axis=(1, 2))[sel].astype('int32')
    out['n_missing_point_years_water'] = (miss & water[:, :, None]).sum(axis=(1, 2))[sel].astype('int32')
    by_pos = miss.sum(axis=2)[sel]
    for j, col in enumerate(PT_COLS):
        out[f'miss_{col}'] = by_pos[:, j].astype('int16')
    return out


def _grouped_cohens_d(edge_rows, keys):
    """Cohen's d (pt_0 vs each position) per group, same formula as cohens_d()."""
    g = edge_rows.groupby(keys, sort=False)
    n = g.size()
    mean = g[PT_COLS].mean()
    var = g[PT_COLS].var(ddof=1)
    out = {}
    for name, col in D_PAIRS.items():
        pooled = np.sqrt(((n - 1) * var['pt_0'] + (n - 1) * var[col]) / (2 * n - 2))
        d = (mean['pt_0'] - mean[col]) / pooled
        d[(n < 2) | (pooled == 0) | pooled.isna()] = np.nan
        out[name] = d
    return pd.DataFrame(out)


def pa_year_metrics(ty):
    """
    PA-year edge metrics for each version (all transects, and without water /
    other-PA / either flagged transects).

    For each version suffix v ('', '_nowater', '_nootherpa', '_noflags'):
    - n_trnst{v}: complete transects
    - edge_extent{v}: share of transects with edge == 1
    - D02{v}, D01{v}, D0m1{v}, D0m2{v}: Cohen's d of pt_0 vs each position,
      on edge transects only
    - edge_intensity{v}: mean of the four D values (NaN if any is NaN)

    Covariates (all transects): gHM_mean, elevation_mean, terrain_slope_mean,
    water_extent_pct, in_other_pa_pct.

    Returns
    -------
    DataFrame
        One row per PA-year.
    """
    keys = ['WDPA_PID', 'year']
    ty = ty.astype({c: 'float64' for c in PT_COLS})
    g_all = ty.groupby(keys, sort=False)
    out = pd.DataFrame({
        'gHM_mean': g_all['gHM_mean_outer'].mean(),
        'elevation_mean': g_all['elevation_pt0'].mean(),
        'terrain_slope_mean': g_all['terrain_slope_pt0'].mean(),
        'water_extent_pct': g_all['water_extent'].mean() * 100,
        'in_other_pa_pct': g_all['in_other_pa'].mean() * 100,
    })

    parts = []
    for v, excluded in VERSIONS.items():
        sub = ty
        for flag in excluded:
            sub = sub[sub[flag] == 0]
        g = sub.groupby(keys, sort=False)
        m = pd.DataFrame({f'n_trnst{v}': g.size(), f'edge_extent{v}': g['edge'].mean()})
        d = _grouped_cohens_d(sub[sub['edge'] == 1], keys).add_suffix(v)
        m = m.join(d, how='left')
        m[f'edge_intensity{v}'] = m[[f'{k}{v}' for k in D_PAIRS]].mean(axis=1, skipna=False)
        parts.append(m)

    metrics = pd.concat(parts, axis=1)
    return metrics.join(out, how='left').reset_index()


def recategorize_biome(biome):
    """
    Recategorize biomes into 8 major groups.

    RESOLVE's "N/A" biome (its Rock & Ice areas, e.g. Greenland, the Himalaya)
    becomes "Rock & Ice". "Unknown" (no intersecting ecoregion) and missing
    values become None.

    Parameters
    ----------
    biome : str
        Original RESOLVE Ecoregions biome name

    Returns
    -------
    str
        Recategorized biome name
    """
    if biome == "N/A":
        return "Rock & Ice"
    if pd.isna(biome) or biome in ("Unknown", "None", ""):
        return None
    if biome == "Mangroves":
        return "Mangrove"
    elif biome in ["Tundra"]:
        return "Tundra"
    elif biome in ["Deserts & Xeric Shrublands"]:
        return "Desert"
    elif biome in ["Tropical & Subtropical Coniferous Forests",
                   "Tropical & Subtropical Moist Broadleaf Forests",
                   "Tropical & Subtropical Dry Broadleaf Forests"]:
        return "Tropical Forest"
    elif biome in ["Mediterranean Forests, Woodlands & Scrub",
                   "Temperate Conifer Forests",
                   "Temperate Broadleaf & Mixed Forests"]:
        return "Temperate Forest"
    elif biome in ["Boreal Forests/Taiga"]:
        return "Boreal Forest"
    elif biome in ["Tropical & Subtropical Grasslands, Savannas & Shrublands",
                   "Temperate Grasslands, Savannas & Shrublands",
                   "Montane Grasslands & Shrublands",
                   "Flooded Grasslands & Savannas"]:
        return "Grassland & Shrubland"
    else:
        return biome


def fit_trend(group, metric='edge_extent', min_years=5):
    """
    OLS trend of one PA's metric on year.

    Years where the metric is NaN are left out. With fewer than ``min_years``
    observations, slope and p-value are NaN.

    Returns a Series with 'slope_{suffix}' and 'p_value_{suffix}', where suffix
    is the metric name without 'edge_' (e.g. 'extent_nowater').
    """
    suffix = metric.removeprefix('edge_')
    ok = group[metric].notna().to_numpy()
    if ok.sum() < min_years:
        return pd.Series({f'slope_{suffix}': np.nan, f'p_value_{suffix}': np.nan})
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        group['year'].to_numpy()[ok], group[metric].to_numpy()[ok]
    )
    return pd.Series({f'slope_{suffix}': slope, f'p_value_{suffix}': p_value})


def bh_adjust(p_values):
    """
    Benjamini-Hochberg adjusted p-values (false discovery rate); NaN stays NaN.

    The number of tests is the number of non-NaN p-values.
    """
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return out
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / np.arange(1, m + 1)
    adjusted = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    vals = np.empty(m)
    vals[order] = adjusted
    out[ok] = vals
    return out


def add_trends(pa_years, alpha=0.05, min_years=5):
    """
    Add per-PA trend columns for every edge_extent* and edge_intensity* metric.

    For each metric (suffix = name without 'edge_', e.g. 'extent_nowater'):
    - slope_{suffix}: OLS slope on year (per year), from >= min_years years
    - p_value_{suffix}: unadjusted p-value of the slope
    - p_adj_{suffix}: Benjamini-Hochberg adjusted p-value, across PAs, separately
      for each metric
    - trend_{suffix}: 'increase' / 'decrease' when p_adj < alpha (by the sign of
      the slope), otherwise 'no_trend'; NaN when there are too few years

    Parameters
    ----------
    pa_years : DataFrame
        One row per PA-year (output of pa_year_metrics, plus attributes).

    Returns
    -------
    DataFrame
        ``pa_years`` with the trend columns merged onto every row.
    """
    metrics = [f'edge_{m}{v}' for v in VERSIONS for m in ('extent', 'intensity')]
    out = pa_years
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for metric in metrics:
            suffix = metric.removeprefix('edge_')
            trends = (pa_years.groupby('WDPA_PID')[['year', metric]]
                      .apply(fit_trend, metric=metric, min_years=min_years)
                      .reset_index())
            slope, p_adj = trends[f'slope_{suffix}'], bh_adjust(trends[f'p_value_{suffix}'])
            trends[f'p_adj_{suffix}'] = p_adj
            trends[f'trend_{suffix}'] = np.where(
                np.isnan(p_adj), None,
                np.where(p_adj < alpha, np.where(slope > 0, 'increase', 'decrease'), 'no_trend'))
            out = out.merge(trends, on='WDPA_PID', how='left')
    return out
