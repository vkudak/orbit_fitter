import numpy as np
from geometry import ra_dec_to_unitvec_icrs, station_positions_gcrs
from methods_gibbs import gibbs_velocity
from poliastro.core.elements import rv2coe
from tle_compare import elements_to_tle_manual

from constants import R_GEO, MU_EARTH


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


def _debug_print_state(r1, r2, r3, v2, times, i1, i2, i3):
    def info(name, vec):
        return f"{name} norm={np.linalg.norm(vec):.6e} vec={vec}"
    print("=== GAUSS DEBUG ===")
    print("i1,i2,i3:", i1, i2, i3)
    print("t1,t2,t3 (JD):", times[i1].jd, times[i2].jd, times[i3].jd)
    print(info("r1", r1))
    print(info("r2", r2))
    print(info("r3", r3))
    print(info("v2", v2))
    # angles between radius vectors
    def angle(u,v):
        nu = np.linalg.norm(u); nv = np.linalg.norm(v)
        if nu==0 or nv==0: return None
        cos = np.dot(u,v)/(nu*nv)
        cos = np.clip(cos, -1.0, 1.0)
        return np.degrees(np.arccos(cos))
    print("angles (deg) r1-r2, r2-r3, r1-r3:",
          angle(r1,r2), angle(r2,r3), angle(r1,r3))
    print("===================")


def gauss_od(obs, lat, lon, h, make_tle=False, dbg=False, norad=99999):
    times, ras, decs, errs, mags, site_n = obs

    idx = np.argsort(times)
    times = times[idx]
    ras = ras[idx]
    decs = decs[idx]
    n = len(times)
    mid = n // 2

    n = len(times)
    i1, i2, i3 = 0, n // 2, n - 1

    los = ra_dec_to_unitvec_icrs(ras, decs)
    R_sites = station_positions_gcrs(lat, lon, h, times)

    rho1 = solve_range_quadratic(R_sites[i1], los[i1]) or 42000.0
    rho2 = solve_range_quadratic(R_sites[i2], los[i2]) or 42000.0
    rho3 = solve_range_quadratic(R_sites[i3], los[i3]) or 42000.0

    r1 = R_sites[i1] + rho1 * los[i1]
    r2 = R_sites[i2] + rho2 * los[i2]
    r3 = R_sites[i3] + rho3 * los[i3]

    v2 = gibbs_velocity(r1, r2, r3)

    if dbg:
        _debug_print_state(r1, r2, r3, v2, times, i1, i2, i3)

    mu = MU_EARTH  # m^3/s^2

    # Безпечний виклик rv2coe з резервними шляхами
    try:
        h_vec, e_vec, inc, raan, argp, nu = rv2coe(mu, r2, v2)
    except Exception as e:
        # Зазвичай тут ZeroDivisionError; друкуємо діагностику
        if dbg:
            print("rv2coe failed with:", repr(e))
            print("Attempting finite-difference fallback for v2...")

        # FALLBACK: центральна різниця для швидкості
        # times можуть бути astropy.time.Time -> перевести в секунди
        dt_seconds = (times[i3].jd - times[i1].jd) * 86400.0
        if dt_seconds == 0:
            raise RuntimeError("Fallback failed: zero time interval between r1 and r3.")

        v2_fd = (r3 - r1) / dt_seconds

        if dbg:
            print("v2 (gibbs) norm:", np.linalg.norm(v2), "v2_fd norm:", np.linalg.norm(v2_fd))

        try:
            h_vec, e_vec, inc, raan, argp, nu = rv2coe(mu, r2, v2_fd)
            v2 = v2_fd  # оновлюємо до резервного
            if dbg:
                print("rv2coe succeeded with fallback v2_fd.")
        except Exception as e2:
            # якщо знову впав — зібрати діагностику і підняти помилку
            _debug_print_state(r1, r2, r3, v2_fd, times, i1, i2, i3)
            raise RuntimeError(
                "rv2coe failed even with fallback velocity. Possible causes:\n"
                "- r and v are (nearly) colinear -> h ≈ 0 (radial motion)\n"
                "- gibbs_velocity returned invalid vector (zero or NaN)\n"
                "- LOS / station geometry is degenerate (angles between r vectors ≈ 0)\n"
                f"Original error: {repr(e)}\nFallback error: {repr(e2)}"
            ) from e2

    # далі нормальні обчислення елементів
    e = np.linalg.norm(e_vec)
    # energy:
    rnorm = np.linalg.norm(r2)
    vnorm = np.linalg.norm(v2)
    energy = 0.5 * vnorm**2 - mu / rnorm
    if abs(energy) < 1e-12:
        a = np.inf
    else:
        a = -mu / (2.0 * energy)

    elements = {
        "a": a,
        "e": e,
        "i": np.degrees(inc),
        "raan": np.degrees(raan),
        "argp": np.degrees(argp),
        "nu": np.degrees(nu)
    }

    if make_tle:
        try:
            tle = elements_to_tle_manual(elements, satnum=norad, epoch=times[mid].jd)
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