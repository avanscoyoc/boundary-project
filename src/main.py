import ee
import time
from analysis import run_all
from utils import *
from config import *

ee.Authenticate()
ee.Initialize(project='dse-staff')

def main():
    start = time.time()

    shp_path = '/workspace/data/global_wdpa_June2021/Global_wdpa_wInfo_June2021.shp'
    gdf = gpd.read_file(shp_path)
    wdpaids = gdf['WDPA_PID'].head(50).tolist()

    run_all(wdpaids, start_year=2001, n_years=23, max_concurrent=15)

    end = time.time()
    print(f"Total elapsed time: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()