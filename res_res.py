# diagnostics_test2.py
import numpy as np
import re
from astropy.time import Time
import matplotlib.pyplot as plt
import math

FILE = "test2.res"

def parse_ra_field(f):
    # expect string like 17420028 -> HHMMSScc  (cc fractional seconds / 100)
    s = f.strip()
    neg = False
    if s.startswith('-'):
        neg = True
        s = s[1:]
    # pad if short
    s = s.zfill(8)
    hh = int(s[0:2])
    mm = int(s[2:4])
    ss = int(s[4:6])
    frac = int(s[6:]) / 100.0
    sec = ss + frac
    hours = hh + mm/60.0 + sec/3600.0
    deg = hours * 15.0
    if neg:
        deg = -deg
    return deg

def parse_dec_field(f):
    # expect string like -17405744 -> sign DDMMSScc
    s = f.strip()
    sign = 1
    if s.startswith('+'):
        s = s[1:]
    if s.startswith('-'):
        sign = -1
        s = s[1:]
    s = s.zfill(8)
    dd = int(s[0:2])
    mm = int(s[2:4])
    ss = int(s[4:6])
    frac = int(s[6:]) / 100.0
    sec = ss + frac
    deg = dd + mm/60.0 + sec/3600.0
    return sign * deg

def parse_time_fields(datestr, timestr):
    # datestr: DDMMYY, timestr: HHMMSSmmm
    dd = int(datestr[0:2])
    mm = int(datestr[2:4])
    yy = int(datestr[4:6])
    year = 2000 + yy
    # times
    ts = timestr.zfill(9)
    hh = int(ts[0:2])
    minu = int(ts[2:4])
    ss = int(ts[4:6])
    msec = int(ts[6:9])
    iso = f"{year:04d}-{mm:02d}-{dd:02d}T{hh:02d}:{minu:02d}:{ss:02d}.{msec:03d}"
    return Time(iso, scale='utc')

def read_test2(path):
    obs_blocks = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('СИСТ'):
            header = ln.split()
            # header like: СИСТ 10092 15543
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith('КОНЕЦ'):
                parts = re.split(r'\s+', lines[i])
                # expect: date time ra dec quality
                if len(parts) >= 4:
                    date_s = parts[0]
                    time_s = parts[1]
                    ra_s = parts[2]
                    dec_s = parts[3]
                    block.append((date_s, time_s, ra_s, dec_s))
                i += 1
            obs_blocks.append({
                'header': header,
                'data': block
            })
        else:
            i += 1
    return obs_blocks

def ra_dec_to_unitvec_deg(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    v = np.vstack((x,y,z)).T
    # normalize just in case
    v = v / np.linalg.norm(v, axis=1)[:,None]
    return v

def solve_range_quadratic(Rs, los, R_target):
    # For each row of Rs and los compute positive root of |R + rho·los| = R_target
    out = []
    for s,l in zip(Rs, los):
        A = 1.0
        B = 2.0 * np.dot(s, l)
        C = np.dot(s,s) - R_target**2
        disc = B*B - 4*A*C
        if disc < 0:
            out.append(np.nan)
            continue
        r1 = (-B + math.sqrt(disc)) / 2.0
        r2 = (-B - math.sqrt(disc)) / 2.0
        cand = [r for r in (r1,r2) if r>0]
        if cand:
            out.append(min(cand))
        else:
            out.append(max(r1,r2))
    return np.array(out)

# ---------------- main diagnostics ----------------
blocks = read_test2(FILE)
print(f"Found {len(blocks)} observation blocks in {FILE}")
# We'll analyze first block for now
for bi,blk in enumerate(blocks):
    print("\n=== BLOCK", bi, "header:", blk['header'], "n=", len(blk['data']), "rows ===")
    dates=[]
    times=[]
    ras=[]
    decs=[]
    for (d,t,ra_s,dec_s) in blk['data']:
        try:
            times.append(parse_time_fields(d,t))
            ras.append(parse_ra_field(ra_s))
            decs.append(parse_dec_field(dec_s))
        except Exception as e:
            print("Parse error on line:", d,t,ra_s,dec_s, "err:",e)
    times = Time(times)
    ras = np.array(ras)
    decs = np.array(decs)
    print("Sample parsed (first 6):")
    for j in range(min(6,len(times))):
        print(times[j].iso, f"RA={ras[j]:.6f}deg DEC={decs[j]:.6f}deg")
    # compute LOS vectors
    los = ra_dec_to_unitvec_deg(ras,decs)
    # angular step between consecutive LOS in arcsec
    angs = []
    for j in range(1,len(los)):
        dot = np.clip(np.dot(los[j], los[j-1]), -1.0, 1.0)
        ang = math.degrees(math.acos(dot)) * 3600.0
        angs.append(ang)
    angs = np.array(angs)
    print("Angular step stats (arcsec): mean %.3f std %.3f max %.3f" % (np.mean(angs), np.std(angs), np.max(angs) if len(angs)>0 else np.nan))
    # quick time axis in seconds from mid
    mid = len(times)//2
    t0 = times[mid]
    dt = np.array([(t - t0).to('s').value for t in times])
    # simple plot RA/Dec vs dt
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(dt, ras, '.-')
    plt.xlabel('dt (s) from mid'); plt.ylabel('RA (deg)')
    plt.subplot(1,2,2)
    plt.plot(dt, decs, '.-')
    plt.xlabel('dt (s)'); plt.ylabel('Dec (deg)')
    plt.suptitle(f"Block {bi} RA/Dec vs time")
    plt.tight_layout(); plt.show()
    # Now check range-for-assumed-spheres: we need station positions
    # For diagnostics I will set R_site = Earth radius * unit vector at some approximate lat/lon
    # But best is to use your real station_positions_gcrs(...) from your geometry module.
    try:
        from geometry import station_positions_gcrs
        # guess lat/lon/h unknown -> the geometry module expects lat,lon,h,times.
        # In your dataset header we had station id in header[1] but not coords.
        # Here we call station_positions_gcrs with placeholder lat/lon/h; if you have real coords, use them.
        # Try to get a plausible lat/lon from your main code or known station id. For now use 50N,30E,100m.
        Rs = station_positions_gcrs(48.5635505, 22.453751, 231.1325, times)
    except Exception as e:
        print("Could not call station_positions_gcrs automatically (need real station coords).", e)
        # fallback: place observer near Earth surface at 50N/30E approx in km (rough)
        Re = 6378.136
        lat = math.radians(48.5635505); lon = math.radians(22.453751)
        x = Re * math.cos(lat) * math.cos(lon)
        y = Re * math.cos(lat) * math.sin(lon)
        z = Re * math.sin(lat)
        Rs = np.tile(np.array([x,y,z]), (len(times),1))
        print("Using fallback R_site ~", Rs[0])
    # compute candidate rho for several target radii
    for Rg in [7000.0, 20000.0, 35786.0+6378.136, 42000.0]:
        rhos = solve_range_quadratic(Rs, los, Rg)
        print(f"R_target={Rg:.1f} km -> rho stats (km): mean {np.nanmean(rhos):.1f} std {np.nanstd(rhos):.1f} min {np.nanmin(rhos):.1f} max {np.nanmax(rhos):.1f}")
    # table first 10
    print("\nFirst 10 obs summary:")
    for j in range(min(30,len(times))):
        rho_geo = solve_range_quadratic([Rs[j]], [los[j]], 42164.0)[0]
        print(times[j].iso, f"RA={ras[j]:.6f}", f"DEC={decs[j]:.6f}", f"ang_step_arcsec={(angs[j-1] if j>0 else 0):.3f}", f"rho_geo={rho_geo:.1f}")
    # show angular step time series
    if len(angs)>0:
        plt.figure()
        t_mid = (times[:-1] - times[mid]).to('s').value
        plt.plot(t_mid, angs, '.-')
        plt.xlabel('dt(s)'); plt.ylabel('angular step (arcsec)')
        plt.title(f'Angular step between consecutive LOS block {bi}')
        plt.show()
