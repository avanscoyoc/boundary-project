import pathlib
import pandas as pd
import geopandas as gpd

PARQUET_DIR = pathlib.Path("results/raw/ndvi_raw/transect_chunks")
SHP_DIR = pathlib.Path("data/intermediate/transect_chunks")
OUTPUT_DIR = pathlib.Path("results")

# --- Set the WDPA_PID you want to extract ---
WDPA_PID = "916" #serengeti

# ── 1. Extract pt_0 time series from parquet chunks ──────────────────────────
chunks = sorted(PARQUET_DIR.glob("*.parquet"))
found = []
for chunk_path in chunks:
    df = pd.read_parquet(chunk_path, columns=["pt_0"])
    if WDPA_PID in df.index.get_level_values("WDPA_PID"):
        subset = df.xs(WDPA_PID, level="WDPA_PID")["pt_0"]
        found.append(subset)
        print(f"  Found {WDPA_PID} in {chunk_path.name}")

series = pd.concat(found)  # index: (year, transectID)

# Pivot to wide format: rows = transectID, columns = year
wide = series.unstack(level="year")
wide.index.name = "transectID"
wide.columns.name = None
wide = wide.sort_index(axis=1)
wide.insert(0, "WDPA_PID", WDPA_PID)
wide = wide.reset_index()

# ── 2. Load shapefiles and filter to edge point (pointID == 0) ────────────────
shp_chunks = [gpd.read_file(p) for p in sorted(SHP_DIR.glob("*.shp"))]
pts = pd.concat(shp_chunks)
pts = pts[(pts["WDPA_PID"] == WDPA_PID) & (pts["pointID"] == 0)]

# ── 3. Join time series to geometry ──────────────────────────────────────────
pts = pts.copy()
pts["transectID"] = pts["transectID"].astype(str)
wide["transectID"] = wide["transectID"].astype(str)
gdf = pts[["WDPA_PID", "transectID", "geometry"]].merge(wide, on=["WDPA_PID", "transectID"])

# ── 4. Save as zipped shapefile for Earth Engine upload ──────────────────────
# Shapefile field names max 10 chars — shorten year columns (e.g. 2003 -> y2003)
gdf.columns = [f"y{c}" if str(c).isdigit() else c for c in gdf.columns]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
shp_dir = OUTPUT_DIR / f"ndvi_timeseries_{WDPA_PID}"
shp_dir.mkdir(exist_ok=True)
gdf.to_file(shp_dir / f"ndvi_timeseries_{WDPA_PID}.shp")

import zipfile
zip_path = OUTPUT_DIR / f"ndvi_timeseries_{WDPA_PID}.zip"
with zipfile.ZipFile(zip_path, "w") as zf:
    for f in shp_dir.iterdir():
        zf.write(f, f.name)
print(f"Saved {len(gdf)} transect edge points → {zip_path}")
