Angles-only Orbit Determination (standalone modules)
----------------------------------------------------
Structure (each file is independent; run `python main.py` after installing dependencies):

Files:
- main.py                 # entry point
- io_utils.py             # parsing observations and TLE input (UTC, HHMMSSss/DDMMSSss)
- geometry.py             # LOS, station positions (astropy)
- methods_gauss.py        # Gauss method implementation
- methods_laplace.py      # Laplace method implementation
- methods_gibbs.py        # Gibbs helper
- refine_orbit.py         # nonlinear refinement and residuals
- tle_compare.py          # propagate TLE and compute per-frame RA/DEC residuals (detailed CSV)
- constants.py            # physical constants

Notes:
- Time in observations must be UTC (as requested).
- RA/DEC are parsed from HHMMSSss / DDMMSSss format (examples in io_utils.py).
- TLE comparison outputs CSV with columns: datetime, obs_RA, obs_DEC, tle_RA, tle_DEC, dra_arcsec, ddec_arcsec.

Dependencies (install locally):

  `pip install -r requirements.txt`

To use Orekit orbit fit JAVA JDK 17+ Should be installer

  `sudo apt update`
  
  `sudo apt install -y openjdk-17-jdk`

After install make sure you got JAVA_HOME = 'path/to/java_jdk'
  

Usage example:

  `python main.py obs.txt --lat 48.63 --lon 22.33 --h 242 --method all --tle-line1 "TLE_LINE1" --tle-line2 "TLE_LINE2" --out-csv residuals.csv`
