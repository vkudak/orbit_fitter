import numpy as np
from geometry import ra_dec_to_unitvec_icrs, station_positions_gcrs
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq
from poliastro.core.elements import rv2coe
from tle_compare import elements_to_tle_manual

from constants import R_GEO, MU_EARTH


def wrap_ra_deg(ra_deg):
    return np.rad2deg(np.unwrap(np.deg2rad(ra_deg)))


def laplace_od(obs, lat, lon, h, polydeg=3, make_tle=False, dbg=False, norad=99999):
    times, ras, decs, _, _, _ = obs
    n = len(times)
    if n < 5:
        raise ValueError('Laplace: at least 5 observations recommended')

    idx = np.argsort(times)
    times = times[idx]
    ras = ras[idx]
    decs = decs[idx]

    mid = n // 2
    t0 = times[mid]
    dt = np.array([(t - t0).to('s').value for t in times])

    ra_unw = wrap_ra_deg(ras)
    s_ra = UnivariateSpline(dt, ra_unw, k=min(polydeg, 5), s=0)
    s_dec = UnivariateSpline(dt, decs, k=min(polydeg, 5), s=0)

    dra_deg_s = s_ra.derivative(1)(0.0)
    ddec_deg_s = s_dec.derivative(1)(0.0)
    dra = np.deg2rad(dra_deg_s)
    ddec = np.deg2rad(ddec_deg_s)

    los_mid = ra_dec_to_unitvec_icrs(np.array([ras[mid]]), np.array([decs[mid]]))[0]
    R_sites = station_positions_gcrs(lat, lon, h, times)
    R0 = R_sites[mid]

    # Обчислення rho
    def resid(rho):
        r = R0 + rho * los_mid
        return np.linalg.norm(r) - R_GEO

    try:
        rho = brentq(resid, 1000.0, 100000.0)
    except Exception:
        rho = R_GEO

    r2 = R0 + rho * los_mid

    # Вектор швидкості
    v2 = rho * np.array([
        -np.cos(np.deg2rad(decs[mid])) * np.sin(np.deg2rad(ras[mid])) * dra
        - np.sin(np.deg2rad(decs[mid])) * np.cos(np.deg2rad(ras[mid])) * ddec,
        np.cos(np.deg2rad(decs[mid])) * np.cos(np.deg2rad(ras[mid])) * dra
        - np.sin(np.deg2rad(decs[mid])) * np.sin(np.deg2rad(ras[mid])) * ddec,
        np.cos(np.deg2rad(decs[mid])) * ddec
    ])

    if dbg:
        print("=== LAPACE DEBUG ===")
        print("r2 norm:", np.linalg.norm(r2))
        print("v2 norm:", np.linalg.norm(v2))
        print("===================")

    # Орбітальні елементи
    mu = MU_EARTH
    try:
        h_vec, e_vec, inc, raan, argp, nu = rv2coe(mu, r2, v2)
    except Exception as e:
        if dbg:
            print("rv2coe failed:", e)
        raise RuntimeError("rv2coe failed in laplace_od") from e

    e = np.linalg.norm(e_vec)
    rnorm = np.linalg.norm(r2)
    vnorm = np.linalg.norm(v2)
    energy = 0.5 * vnorm**2 - mu / rnorm
    a = -mu / (2.0 * energy) if abs(energy) > 1e-12 else np.inf

    elements = {
        "a": a,
        "e": e,
        "i": np.degrees(inc),
        "raan": np.degrees(raan),
        "argp": np.degrees(argp),
        "nu": np.degrees(nu)
    }

    tle = None
    if make_tle:
        try:
            tle = elements_to_tle_manual(elements, satnum=norad, epoch_jd=times[mid].jd)
        except Exception:
            tle = None
    else:
        tle = None

    return {
        "r": r2,
        "v": v2,
        "elements": elements,
        "tle": tle
    }
