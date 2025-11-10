import csv
import numpy as np
from sgp4.api import Satrec, jday
from astropy.coordinates import TEME, GCRS
import astropy.units as u

def propagate_tle_to_radec(tle1, tle2, times):
    sat = Satrec.twoline2rv(tle1, tle2)
    ras = []
    decs = []
    for t in times:
        jd, fr = jday(t.datetime.year, t.datetime.month, t.datetime.day,
                      t.datetime.hour, t.datetime.minute, t.datetime.second + t.datetime.microsecond*1e-6)
        e, r, v = sat.sgp4(jd, fr)
        if e != 0:
            ras.append(np.nan); decs.append(np.nan)
        else:
            try:
                teme = TEME(x=r[0]*u.km, y=r[1]*u.km, z=r[2]*u.km, obstime=t)
                g = teme.transform_to(GCRS(obstime=t))
                ic = g.transform_to('icrs')
                ras.append(ic.ra.deg); decs.append(ic.dec.deg)
            except Exception:
                ras.append(np.nan); decs.append(np.nan)
    return np.array(ras), np.array(decs)

def compare_with_tle(obs, state, tle_line1, tle_line2, out_csv=None):
    times, ras_obs, decs_obs, errs, mags = obs
    ras_tle, decs_tle = propagate_tle_to_radec(tle_line1, tle_line2, times)
    dra = (ras_tle - ras_obs) * 3600.0 * np.cos(np.deg2rad(decs_obs))
    ddec = (decs_tle - decs_obs) * 3600.0
    rms = np.sqrt(np.nanmean(np.hstack((dra, ddec))**2))
    if out_csv:
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['time_iso','obs_ra_deg','obs_dec_deg','tle_ra_deg','tle_dec_deg','dra_arcsec','ddec_arcsec'])
            for i in range(len(times)):
                w.writerow([times[i].iso, ras_obs[i], decs_obs[i], ras_tle[i], decs_tle[i], dra[i], ddec[i]])
    print(f"TLE comparison RMS (arcsec): {rms:.3f}")
    return rms
