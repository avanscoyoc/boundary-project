"""
Edge metric functions (script 04).

- find_asset_csvs: group the Earth Engine CSV parts by asset number
- read_asset_csvs: read one asset's CSV parts
- build_transect_years: one row per complete transect-year, with edge flag and
  the water / other-PA flags
- pa_year_metrics: PA-year edge extent, Cohen's d values, and edge intensity, for
  four transect subsets (all, no water, no other-PA, neither)
- add_trends: per-PA OLS trend of every extent and intensity metric
- cohens_d, recategorize_biome, classify_trend

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
              'slope': 'float32', **{y: 'float32' for y in year_cols}}
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
          water_extent, in_other_pa, gHM_mean_outer, elevation_pt0, slope_pt0
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
            slope_pt0=per_point('slope')[:, 2],
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
    for col in ['water_extent', 'in_other_pa', 'gHM_mean_outer', 'elevation_pt0', 'slope_pt0']:
        ty[col] = transects[col].to_numpy()[t_idx]

    counts = {
        'transects_received': transects_received,
        'transects_missing_points': transects_received - n_tr,
        'complete_per_year': ty['year'].value_counts().sort_index().to_dict(),
    }
    return ty, counts


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

    Covariates (all transects): gHM_mean, elevation_mean, slope_mean,
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
        'slope_mean': g_all['slope_pt0'].mean(),
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


def classify_trend(group, metric='edge_extent', alpha=0.05, min_years=5):
    """
    Classify the temporal trend of one PA's metric with OLS on year.

    Years where the metric is NaN are left out. With fewer than ``min_years``
    observations, slope, p-value and trend are NaN.

    Returns a Series with 'trend_{suffix}', 'slope_{suffix}', 'p_value_{suffix}',
    where suffix is the metric name without 'edge_' (e.g. 'extent_nowater').
    Trend classes: sig_increase / sig_decrease (p < alpha), otherwise
    increase / decrease / no_change by the sign of the slope.
    """
    suffix = metric.removeprefix('edge_')
    ok = group[metric].notna().to_numpy()
    if ok.sum() < min_years:
        return pd.Series({f'trend_{suffix}': None, f'slope_{suffix}': np.nan, f'p_value_{suffix}': np.nan})

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        group['year'].to_numpy()[ok], group[metric].to_numpy()[ok]
    )
    if p_value < alpha:
        trend = 'sig_increase' if slope > 0 else 'sig_decrease'
    elif slope > 0:
        trend = 'increase'
    elif slope < 0:
        trend = 'decrease'
    else:
        trend = 'no_change'

    return pd.Series({f'trend_{suffix}': trend, f'slope_{suffix}': slope, f'p_value_{suffix}': p_value})


def add_trends(pa_years):
    """
    Add per-PA trend columns for every edge_extent* and edge_intensity* metric.

    Parameters
    ----------
    pa_years : DataFrame
        One row per PA-year (output of pa_year_metrics, plus attributes).

    Returns
    -------
    DataFrame
        ``pa_years`` with trend_/slope_/p_value_ columns merged onto every row.
    """
    metrics = [f'edge_{m}{v}' for v in VERSIONS for m in ('extent', 'intensity')]
    out = pa_years
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for metric in metrics:
            trends = (pa_years.groupby('WDPA_PID')[['year', metric]]
                      .apply(classify_trend, metric=metric)
                      .reset_index())
            out = out.merge(trends, on='WDPA_PID', how='left')
    return out
