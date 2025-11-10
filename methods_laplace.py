import numpy as np
from geometry import ra_dec_to_unitvec_icrs, station_positions_gcrs
from scipy.interpolate import UnivariateSpline
from constants import R_GEO

def wrap_ra_deg(ra_deg):
    return np.rad2deg(np.unwrap(np.deg2rad(ra_deg)))

def laplace_od(obs, lat, lon, h, polydeg=3):
    times, ras, decs, _, _, _ = obs
    n = len(times)
    if n < 5:
        raise ValueError('Laplace: at least 5 observations recommended')

    idx = np.argsort(times)
    times = times[idx]
    ras = ras[idx]
    decs = decs[idx]


    mid = n//2
    t0 = times[mid]
    dt = np.array([(t - t0).to('s').value for t in times])
    ra_unw = wrap_ra_deg(ras)
    s_ra = UnivariateSpline(dt, ra_unw, k=min(polydeg,5), s=0)
    s_dec = UnivariateSpline(dt, decs, k=min(polydeg,5), s=0)
    dra_deg_s = s_ra.derivative(1)(0.0)
    ddec_deg_s = s_dec.derivative(1)(0.0)
    dra = np.deg2rad(dra_deg_s)
    ddec = np.deg2rad(ddec_deg_s)
    los_mid = ra_dec_to_unitvec_icrs(np.array([ras[mid]]), np.array([decs[mid]]))[0]
    R_sites = station_positions_gcrs(lat, lon, h, times)
    R0 = R_sites[mid]
    def resid(rho):
        r = R0 + rho * los_mid
        return np.linalg.norm(r) - R_GEO
    from scipy.optimize import brentq
    try:
        rho = brentq(resid, 1000.0, 100000.0)
    except Exception:
        rho = R_GEO
    r2 = R0 + rho * los_mid
    v2 = rho * np.array([
        -np.cos(np.deg2rad(decs[mid]))*np.sin(np.deg2rad(ras[mid]))*dra - np.sin(np.deg2rad(decs[mid]))*np.cos(np.deg2rad(ras[mid]))*ddec,
        np.cos(np.deg2rad(decs[mid]))*np.cos(np.deg2rad(ras[mid]))*dra - np.sin(np.deg2rad(decs[mid]))*np.sin(np.deg2rad(ras[mid]))*ddec,
        np.cos(np.deg2rad(decs[mid]))*ddec
    ])
    return r2, v2
