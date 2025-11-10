import numpy as np
from geometry import ra_dec_to_unitvec_icrs, station_positions_gcrs
from methods_gibbs import gibbs_velocity
from constants import R_GEO

def solve_range_quadratic(R_site, los, R_target=R_GEO):
    s = R_site
    l = los
    A = 1.0
    B = 2.0 * np.dot(s, l)
    C = np.dot(s, s) - R_target**2
    disc = B*B - 4*A*C
    if disc < 0:
        return None
    r1 = (-B + np.sqrt(disc)) / 2.0
    r2 = (-B - np.sqrt(disc)) / 2.0
    candidates = [r for r in (r1, r2) if r > 0]
    if not candidates:
        return max(r1, r2)
    return min(candidates)

def gauss_od(obs, lat, lon, h):
    times, ras, decs, errs, mags, site_n = obs
    n = len(times)
    i1, i2, i3 = 0, n//2, n-1
    los = ra_dec_to_unitvec_icrs(ras, decs)
    R_sites = station_positions_gcrs(lat, lon, h, times)
    rho1 = solve_range_quadratic(R_sites[i1], los[i1]) or 42000.0
    rho2 = solve_range_quadratic(R_sites[i2], los[i2]) or 42000.0
    rho3 = solve_range_quadratic(R_sites[i3], los[i3]) or 42000.0
    r1 = R_sites[i1] + rho1 * los[i1]
    r2 = R_sites[i2] + rho2 * los[i2]
    r3 = R_sites[i3] + rho3 * los[i3]
    v2 = gibbs_velocity(r1, r2, r3)
    return r2, v2
