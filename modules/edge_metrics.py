"""
Edge metric functions (script 04).

- find_asset_csvs / read_asset_csvs: the Earth Engine CSV parts of each asset
- build_transect_years: one row per complete transect-year, with the edge, water and
  other-PA flags
- pa_year_metrics: PA-year edge extent, Cohen's d values and edge intensity, for four
  transect subsets (all, no water, no other-PA, neither)
- missing_transects, recategorize_biome

Per-PA trends are computed in R (script 07). Script 02 writes whole PAs per upload file, so
each asset is processed on its own and memory is bounded by one asset.
"""

import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

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

# Earth Engine CSV part names: {prefix}{asset}_part_{i}[_{retry}].csv, e.g.
# ndvi_gradient_003_part_7.csv or ndvi_gradient_003_part_7_01.csv (a retried half)
_CSV_SUFFIX = r'(\d{3})_part_(\d+)(?:_(\d{2}))?\.csv'


def find_asset_csvs(csv_dir, prefix):
    """Group the CSV parts in `csv_dir` named `{prefix}NNN_part_N[_RR].csv` by asset number NNN
    (`prefix`: config.GEE_EXPORT_PREFIX for script 04, config.VALIDATION_EXPORT_PREFIX for 13).
    Returns ({asset: sorted paths, retry halves included}, names of the other CSVs, e.g. Drive
    duplicates such as 'name (1).csv' or another run's exports).
    """
    pattern = re.compile('^' + re.escape(prefix) + _CSV_SUFFIX + '$')
    groups, unrecognized = defaultdict(list), []
    for f in sorted(Path(csv_dir).glob('*.csv')):
        m = pattern.match(f.name)
        if m:
            groups[m.group(1)].append(f)
        else:
            unrecognized.append(f.name)
    return dict(sorted(groups.items())), unrecognized


def read_asset_csvs(files, years):
    """Read and stack the CSV parts of one asset (values float32; -9999, masked pixels, -> NaN)."""
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
    """One row per complete transect-year of one asset (`df` from read_asset_csvs, `years` in CSV
    column order). A transect needs all 5 points in the export, a transect-year a value at all 5
    points; the others are dropped and counted. `flags`: WDPA_PID, transectID, in_other_pa
    (transect_points.parquet; missing -> 0).

    edge: pt_0 > min(pt_1, pt_2) and pt_0 > min(pt_m1, pt_m2). water_extent: any point in the JRC
    maximum water extent. Also the point values pt_m2 ... pt_2, in_other_pa and the covariates
    gHM_mean_outer, elevation_pt0, terrain_slope_pt0. Returns (transect-years, counts).
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
    """Transects missing a value at one or more of their 5 points in at least one year (the
    transect-years build_transect_years drops), for R script 05's missing-data counts. Transects
    without all 5 points in the export are skipped (build_transect_years counts them).

    One row per such transect, with water_any (a point in the JRC maximum water extent) and
    counts of missing years and point-years, in total, at water and per position (miss_pt_*).
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
    """Cohen's d of pt_0 vs each other position, per group: pooled SD with (n-1) df; NaN when
    n < 2 or the pooled SD is 0."""
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
    """PA-year metrics (one row per PA-year) for each transect version v ('', '_nowater',
    '_nootherpa', '_noflags'): n_trnst{v} (complete transects), edge_extent{v} (share with
    edge == 1), D02{v}, D01{v}, D0m1{v}, D0m2{v} (Cohen's d of pt_0 vs each position, on edge
    transects) and edge_intensity{v} (their mean; NaN if any is NaN); plus the covariates over all
    transects: gHM_mean, elevation_mean, terrain_slope_mean, water_extent_pct, in_other_pa_pct.
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


def biome_groups():
    """The biome table (config.BIOME_GROUPS_CSV): BIOME_NUM (empty for "N/A"), BIOME_NAME (RESOLVE
    name), group. Read as text, so "N/A" stays a name."""
    from modules import config
    return pd.read_csv(config.BIOME_GROUPS_CSV, dtype=str, keep_default_na=False)


_BIOME_GROUP = dict(zip(biome_groups()['BIOME_NAME'], biome_groups()['group']))


def recategorize_biome(biome):
    """
    Group a RESOLVE Ecoregions 2017 biome name (modules/biome_groups.csv): 8 groups, by
    vegetation structure and then temperature (Tropical Grassland, with flooded grasslands;
    Temperate & Montane Grassland; Tundra; Desert; Mediterranean; Tropical Forest; Temperate
    Forest; Boreal Forest), plus Mangrove and Rock & Ice (RESOLVE's "N/A"), which the analysis
    excludes. "Unknown" (no intersecting ecoregion) and missing values give None; a name not in
    the table is returned unchanged.
    """
    if not isinstance(biome, str) and pd.isna(biome):
        return None
    if biome in ("Unknown", "None", ""):
        return None
    return _BIOME_GROUP.get(biome, biome)
