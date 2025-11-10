import numpy as np
from scipy.optimize import least_squares
from astropy.coordinates import GCRS, ICRS, CartesianRepresentation
import astropy.units as u
from poliastro.bodies import Earth
from poliastro.twobody.orbit import Orbit

def propagate_state_to_radec(r0, v0, epoch, times):
    orb = Orbit.from_vectors(Earth, r0*u.km, v0*u.km/u.s, epoch=epoch)
    ras = []
    decs = []
    for t in times:
        tof = (t - epoch).to(u.s)
        rr = orb.propagate(tof).r.to(u.km).value
        # g = GCRS(x=rr[0]*u.km, y=rr[1]*u.km, z=rr[2]*u.km, obstime=t)
        rep = CartesianRepresentation(rr[0] * u.km, rr[1] * u.km, rr[2] * u.km)
        g= GCRS(rep, obstime=t)
        # ic = g.transform_to('icrs')
        ic = g.transform_to(ICRS())
        ras.append(ic.ra.deg)
        decs.append(ic.dec.deg)
    return np.array(ras), np.array(decs)

def residuals_vec(x, times, ras_obs, decs_obs):
    r0 = x[0:3]
    v0 = x[3:6]
    epoch = times[len(times)//2]
    ras_pred, decs_pred = propagate_state_to_radec(r0, v0, epoch, times)
    dra = (ras_pred - ras_obs) * 3600.0 * np.cos(np.deg2rad(decs_obs))
    ddec = (decs_pred - decs_obs) * 3600.0
    return np.hstack((dra, ddec))

def refine_solution(obs, state, lat, lon, h):
    times, ras, decs, errs, mags, site_n = obs
    r0, v0 = state["r"], state["v"]
    x0 = np.hstack((r0, v0))
    res = least_squares(lambda x: residuals_vec(x, times, ras, decs), x0, verbose=1, max_nfev=200)
    r_opt = res.x[0:3]
    v_opt = res.x[3:6]
    return r_opt, v_opt

def compute_rms(obs, state, lat, lon, h):
    times, ras, decs, errs, mags, site_n = obs
    r0, v0 = state
    ras_pred, decs_pred = propagate_state_to_radec(r0, v0, times[len(times)//2], times)
    dra = (ras_pred - ras) * 3600.0 * np.cos(np.deg2rad(decs))
    ddec = (decs_pred - decs) * 3600.0
    return np.sqrt(np.mean(np.hstack((dra, ddec))**2))
