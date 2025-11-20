# debug.py
from astropy.time import Time
import numpy as np
from methods_laplace_new2 import laplace_od

# --- Дані з Vallado Example 7-2 (LEO) ---
times = Time([
    "2001-05-23T20:16:24.024",
    "2001-05-23T20:36:48.938",
    "2001-05-23T20:57:13.852"
], scale="utc")

ra_deg = np.array([283.92195, 294.67483, 303.82137])
dec_deg = np.array([-34.38700, -41.46583, -45.36542])

obs = (times, ra_deg, dec_deg, None, None, None)

# --- Тимчасово перевизначаємо residuals з дебаг-принтами ---
original_residuals = None

def debug_residuals(x):
    global original_residuals
    from scipy.integrate import solve_ivp
    from methods_laplace_new2 import _two_body_ode, MU, dt_s, R_sites, los_all, n

    r2 = x[:3]
    v2 = x[3:]

    sol = solve_ivp(
        _two_body_ode,
        (dt_s.min() - 200, dt_s.max() + 200),
        np.hstack((r2, v2)),
        args=(MU,),
        method="DOP853",
        rtol=1e-12, atol=1e-12, dense_output=True
    )

    print("\n=== DEBUG residuals call ===")
    print("x0 (r2 v2):", x)
    print("r2 norm:", np.linalg.norm(r2))
    print("v2 norm:", np.linalg.norm(v2))

    res = np.empty(n)
    for j in range(n):
        r_sat = sol.sol(dt_s[j])[:3]
        rho_vec = r_sat - R_sites[j]
        norm = np.linalg.norm(rho_vec)
        dot = np.dot(rho_vec, los_all[j])
        cos_raw = dot / norm if norm > 0 else 0

        print(f"  obs {j}: rho_norm = {norm:.6f} km, dot = {dot:.6f}, cos_raw = {cos_raw:.16f}")

        if cos_raw > 1.0 + 1e-10:
            print("  !!! COS > 1.0 DETECTED !!!")
        if np.isnan(cos_raw):
            print("  !!! NaN DETECTED !!!")

        cos_clip = np.clip(cos_raw, -1.0, 1.0)
        res[j] = np.arccos(cos_clip)

    print("  residuals (arcsec):", res * 206265)
    print("============================\n")
    return res

# Підміняємо residuals
import methods_laplace_new2
methods_laplace_new2.residuals = debug_residuals

# Запускаємо
print("Запускаємо laplace_od з дебагом...")
result = laplace_od(obs, lat=19.0, lon=-155.0, h=0, dbg=True)
print("Фінальний результат:", result["elements"])