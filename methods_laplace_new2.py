# orbit_fitter/methods_laplace.py
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from geometry import ra_dec_to_unitvec_icrs, station_positions_gcrs
from astropy.constants import c as speed_of_light
from astropy import units as u

MU = 398600.4418  # km^3/s^2

# Простий rv2coe без poliastro
def simple_rv2coe(mu, r, v):
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    h_vec = np.cross(r, v)
    h = np.linalg.norm(h_vec)
    inc = np.arccos(h_vec[2] / h)
    energy = v_norm**2 / 2 - mu / r_norm
    a = -mu / (2 * energy) if energy < 0 else np.inf
    e_vec = np.cross(v, h_vec) / mu - r / r_norm
    e = np.linalg.norm(e_vec)
    # RAAN
    n_vec = np.cross([0, 0, 1], h_vec)
    n = np.linalg.norm(n_vec)
    raan = np.arccos(n_vec[0] / n) if n > 0 else 0.0
    if n_vec[1] < 0:
        raan = 2 * np.pi - raan
    # argp and nu (спрощено, достатньо для тесту)
    argp = 0.0
    nu = 0.0
    if e > 1e-6:
        argp = np.arccos(np.dot(n_vec, e_vec) / (n * e))
        if e_vec[2] < 0:
            argp = 2 * np.pi - argp
        nu = np.arccos(np.dot(e_vec, r) / (e * r_norm))
        if np.dot(r, v) < 0:
            nu = 2 * np.pi - nu
    return a, e, np.degrees(inc), np.degrees(raan) % 360, np.degrees(argp) % 360, np.degrees(nu) % 360


def _two_body_ode(t, y, mu):
    r = y[:3]
    v = y[3:]
    r_norm = np.linalg.norm(r) + 1e-20
    a = -mu * r / r_norm**3
    return np.hstack((v, a))


def laplace_od(obs, lat, lon, h, dbg=False):
    print("Runing laplace_new2_3000 times")
    times, ra_deg, dec_deg, _, _, _ = obs
    n = len(times)
    if n < 3:
        raise ValueError("At least 3 observations required")

    idx = np.argsort(times)
    times = times[idx]
    ra_deg = np.asarray(ra_deg)[idx]
    dec_deg = np.asarray(dec_deg)[idx]

    mid = n // 2
    t0 = times[mid]
    dt_s = np.array([(t - t0).to("s").value for t in times])

    los_all = ra_dec_to_unitvec_icrs(ra_deg, dec_deg)

    R_sites = station_positions_gcrs(lat, lon, h, times)


    mid = n // 2
    los_mid = los_all[mid]
    R_mid = R_sites[mid]

    rho0 = 2000.0

    r2_0 = R_mid + rho0 * los_mid
    r_norm = np.linalg.norm(r2_0)

    v_mag = np.sqrt(MU / r_norm) * 1.08

    z = np.array([0.0, 0.0, 1.0])
    tang = np.cross(r2_0, z)
    if np.linalg.norm(tang) < 0.1:
        tang = np.cross(r2_0, [0.0, 1.0, 0.0])
    tang /= np.linalg.norm(tang)

    v2_0 = v_mag * tang

    def residuals(x):
        r2 = x[:3]
        v2 = x[3:]

        sol = solve_ivp(
            _two_body_ode,
            (dt_s.min() - 200, dt_s.max() + 200),
            np.hstack((r2, v2)),
            args=(MU,),
            method="DOP853",
            rtol=1e-12,
            atol=1e-12,
            dense_output=True
        )

        res = np.empty(n)
        for j in range(n):
            r_sat = sol.sol(dt_s[j])[:3]
            rho_vec = r_sat - R_sites[j]
            dot = np.dot(rho_vec, los_all[j])
            norm2 = np.dot(rho_vec, rho_vec)  # <-- квадрат норми, 100% точно
            if norm2 < 1e-6:
                res[j] = np.pi
                continue
            cos_ang = dot / np.sqrt(norm2)
            cos_ang = np.clip(cos_ang, -1.0, 1.0)
            res[j] = np.arccos(cos_ang)
        return res

    x0 = np.hstack((r2_0, v2_0))
    opt = least_squares(
        residuals,
        x0,
        bounds=(-1e9, 1e9),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=2000,
        verbose=2 if dbg else 0
    )

    r_opt = opt.x[:3]
    v_opt = opt.x[3:]

    # a, e, i, raan, argp, nu = simple_rv2coe(mu, r_opt, v_opt)

    # прості елементи без poliastro
    h_vec = np.cross(r_opt, v_opt)
    h = np.linalg.norm(h_vec)
    inc = np.degrees(np.arccos(h_vec[2] / h))
    energy = np.dot(v_opt, v_opt) / 2 - MU / np.linalg.norm(r_opt)
    a = -MU / (2 * energy) if energy < 0 else np.inf
    e_vec = np.cross(v_opt, h_vec) / MU - r_opt / np.linalg.norm(r_opt)
    e = np.linalg.norm(e_vec)

    # elements = {
    #     "a": a,
    #     "e": e,
    #     "i": i,
    #     "raan": raan,
    #     "argp": argp,
    #     "nu": nu
    # }

    elements = {"a": float(a), "e": float(e), "i": float(inc)}

    if dbg:
        final_res_as = residuals(opt.x) * 206265
        print(f"\nLaplace OD success! nfev = {opt.nfev}")
        print(f"a = {a:.3f} km, e = {e:.6f}, i = {inc:.3f}°")
        print(f"Residuals: mean = {final_res_as.mean():.2f}″, max = {final_res_as.max():.2f}″")

    return {"r": r_opt, "v": v_opt, "elements": elements, "ls": opt}