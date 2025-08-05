import ee
import time
from analysis import run_all
from utils import *
from config import *

ee.Authenticate()
ee.Initialize(project='dse-staff')


def main():
    start = time.time()
    
    sites = ee.FeatureCollection("projects/dse-staff/assets/movement_metadata").geometry().buffer(50000) #50km
    wdpaids = pas_at_movement_sites(sites)[:5]

    run_all(wdpaids, start_year=2016, n_years=1, max_concurrent=15)

    end = time.time()
    print(f"Total elapsed time: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()