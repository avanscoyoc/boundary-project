import ee
import time
from analysis import run_all

ee.Authenticate()
ee.Initialize(project='dse-staff')

def main():
    start = time.time()
    run_all(wdpaids = ["7","916"], start_year=2010, n_years=10, max_workers=4)
    end = time.time()
    print(f"Total elapsed time: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()