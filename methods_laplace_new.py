import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

from geometry import ra_dec_to_unitvec_icrs, station_positions_gcrs
from poliastro.core.elements import rv2coe
from poliastro.bodies import Earth
from astropy import units as u

from tle_compare import create_tle_manual


# =============== 2-body ODE ======================

def _two_body_ode(t, y, mu):
    r = y[0:3]
    v = y[3:6]
    rnorm = np.linalg.norm(r)
    a = -mu * r / (rnorm**3 + 1e-12)  # захист
    return np.hstack((v, a))

def propagate_rv_batch(mu, r0, v0, dt_array, method="DOP853",
                       rtol=1e-9, atol=1e-12):
    """
    Пропагує (r0,v0) для всіх dt_array за одне інтегрування.
    dt_array — масив секунд (може містити і негативні dt).
    """
    dt_array = np.asarray(dt_array, dtype=float)
    # Якщо є негативні dt — інтегруємо мін до макс, а потім вибираємо значення
    t_min = float(np.min(dt_array))
    t_max = float(np.max(dt_array))
    y0 = np.hstack((r0, v0))

    # Інтегруємо з t_min -> t_max
    sol = solve_ivp(
        _two_body_ode,
        (t_min, t_max),
        y0,
        args=(mu,),
        method=method,
        rtol=rtol,
        atol=atol,
        dense_output=True
    )

    r_out = np.empty((len(dt_array), 3))
    v_out = np.empty((len(dt_array), 3))

    for i, dt in enumerate(dt_array):
        yf = sol.sol(float(dt))
        r_out[i] = yf[0:3]
        v_out[i] = yf[3:6]

    return r_out, v_out

# ==================================================

def wrap_ra_deg(ra_deg):
    return np.rad2deg(np.unwrap(np.deg2rad(ra_deg)))

# ==================================================

def laplace_od(
        obs,
        lat,
        lon,
        h,
        polydeg=3,
        make_tle=False,
        dbg=False,
        norad=99999,
        cospar="25001A"
    ):

    times, ras, decs, _, _, _ = obs
    n = len(times)
    if n < 3:
        raise ValueError("Laplace requires ≥3 observations")

    # ===== sort by time =====
    idx = np.argsort(times)
    times = times[idx]
    ras = ras[idx]
    decs = decs[idx]

    mid = n // 2
    t0 = times[mid]
    dt = np.array([(t - t0).to("s").value for t in times])  # seconds

    # ===== LOS splines =====
    ra_unw = wrap_ra_deg(ras)
    s_ra = UnivariateSpline(dt, ra_unw, k=min(polydeg, 5), s=0)
    s_dec = UnivariateSpline(dt, decs, k=min(polydeg, 5), s=0)

    dra_deg_s = s_ra.derivative(1)(0.0)
    ddec_deg_s = s_dec.derivative(1)(0.0)
    dra = np.deg2rad(dra_deg_s)
    ddec = np.deg2rad(ddec_deg_s)

    los_all = ra_dec_to_unitvec_icrs(ras, decs)
    los_mid = los_all[mid]

    # ===== Station positions GCRS =====
    R_sites = station_positions_gcrs(lat, lon, h, times)
    R0 = R_sites[mid]

    # ===== los_dot у ICRS =====
    ra_mid = np.deg2rad(ras[mid])
    dec_mid = np.deg2rad(decs[mid])
    cosd = np.cos(dec_mid)
    sind = np.sin(dec_mid)
    cosr = np.cos(ra_mid)
    sinr = np.sin(ra_mid)

    los_dot_mid = np.array([
        -cosd * sinr * dra - sind * cosr * ddec,
         cosd * cosr * dra - sind * sinr * ddec,
         cosd * ddec
    ])

    mu = Earth.k.to_value(u.km ** 3 / u.s ** 2)
    # ===== Початковий rho0 (мінімальний паралакс) =====
    # Дуже проста робоча оцінка:
    # Беремо напрямок на середині — шукаємо rho мінімізуючи відхилення LOS
    # Тобто шукаємо пересічення з "наближеною" сферою великого радіусу ~ 40000 km
    # Але не фіксуємо GEO.
    # rho0 = 40000.0  # стартове припущення
    #
    # r2_0 = R0 + rho0 * los_mid
    # v2_0 = rho0 * los_dot_mid  # init

    # Гарне початкове наближення: припустимо кругова орбіта на висоті GEO
    alt_guess = 35786 * u.km
    a_guess = (Earth.R.to(u.km).value + alt_guess.value)
    v_circ = np.sqrt(mu / a_guess)

    # Напрямок швидкості — перпендикулярний до r і los
    r_guess = R0 + 40000 * los_mid
    r_guess /= np.linalg.norm(r_guess)

    # Приблизно тангенціальна швидкість
    los_cross_z = np.cross(los_mid, [0, 0, 1])
    if np.linalg.norm(los_cross_z) < 0.1:
        tang_dir = np.cross(los_mid, [0, 1, 0])
    else:
        tang_dir = los_cross_z
    tang_dir /= np.linalg.norm(tang_dir)

    v2_0 = v_circ * tang_dir
    r2_0 = R0 + 40000 * los_mid

    if dbg:
        print("[INIT] rho0=", rho0, " r2_0=", r2_0, " v2_0=", v2_0)

    # ===== Підготовка для least-squares =====
    los_obs = los_all
    R_sites_all = R_sites


    def residuals(x):
        r2 = x[0:3]
        v2 = x[3:6]

        # propagate for all dt
        r_preds, v_preds = propagate_rv_batch(mu, r2, v2, dt)

        res = np.empty(len(dt))
        for j in range(len(dt)):
            rho_vec = r_preds[j] - R_sites_all[j]
            rho_norm = np.linalg.norm(rho_vec)
            if rho_norm == 0:
                res[j] = 1.0
                continue
            los_pred = rho_vec / rho_norm
            dot = np.dot(los_pred, los_obs[j])
            dot = np.clip(dot, -1.0, 1.0)
            res[j] = np.arccos(dot)
        return res

    x0 = np.hstack((r2_0, v2_0))
    big = 1e8
    lb = np.array([-big] * 6)
    ub = np.array([ big] * 6)

    opt = least_squares(
        residuals,
        x0,
        bounds=(lb, ub),
        xtol=1e-10, ftol=1e-10, gtol=1e-10,
        max_nfev=200
    )

    if dbg:
        print("LS success:", opt.success, opt.message)

    x_opt = opt.x
    r2 = x_opt[0:3]
    v2 = x_opt[3:6]

    if dbg:
        print("Optimized r2 =", r2, " |v2|=", np.linalg.norm(v2))
        res_f = residuals(opt.x)
        print("Residuals (arcsec): mean=", np.mean(res_f)*206265,
              "max=", np.max(res_f)*206265)

    # ===== Орбітальні елементи =====
    try:
        h_vec, e_vec, inc, raan, argp, nu = rv2coe(mu, r2, v2)
    except Exception as e:
        raise RuntimeError("rv2coe failed in improved Laplace") from e

    e = np.linalg.norm(e_vec)
    rnorm = np.linalg.norm(r2)
    vnorm = np.linalg.norm(v2)
    energy = 0.5*vnorm*vnorm - mu/rnorm
    a = -mu / (2*energy) if abs(energy) > 1e-12 else np.inf

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
            tle = create_tle_manual(elements, norad=norad, cospar=cospar, epoch=times[mid].jd)
        except Exception:
            tle = None
    else:
        tle = None

    return {
        "r": r2,
        "v": v2,
        "elements": elements,
        "ls": opt,
        "tle":tle
    }
