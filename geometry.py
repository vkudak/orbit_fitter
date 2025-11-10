import numpy as np
from astropy.coordinates import EarthLocation
import astropy.units as u

def ra_dec_to_unitvec_icrs(ras_deg, decs_deg):
    ras = np.deg2rad(ras_deg)
    decs = np.deg2rad(decs_deg)
    x = np.cos(decs) * np.cos(ras)
    y = np.cos(decs) * np.sin(ras)
    z = np.sin(decs)
    v = np.vstack((x,y,z)).T
    v = v / np.linalg.norm(v, axis=1)[:, None]
    return v

def station_positions_gcrs(lat_deg, lon_deg, height_m, times):
    loc = EarthLocation(lat=lat_deg*u.deg, lon=lon_deg*u.deg, height=height_m*u.m)
    rs = []
    for t in times:
        g = loc.get_gcrs(t)
        rs.append(g.cartesian.xyz.to(u.km).value)
    return np.vstack(rs)
