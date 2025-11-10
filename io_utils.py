import numpy as np
from astropy.time import Time

def parse_hhmmss_ra(ra_str):
    hh = int(ra_str[0:2])
    mm = int(ra_str[2:4])
    ss = float(ra_str[4:6])
    frac = 0.0
    if len(ra_str) > 6:
        tail = ra_str[6:]
        frac = float('0.' + tail)
    ss = ss + frac
    return (hh + mm/60.0 + ss/3600.0) * 15.0

def parse_ddmmss_dec(dec_str):
    sign = 1
    core = dec_str
    if dec_str[0] in ['+','-']:
        if dec_str[0] == '-':
            sign = -1
        core = dec_str[1:]
    dd = int(core[0:2])
    mm = int(core[2:4])
    ss = float(core[4:6])
    frac = 0.0
    if len(core) > 6:
        tail = core[6:]
        frac = float('0.' + tail)
    ss = ss + frac
    return sign * (dd + mm/60.0 + ss/3600.0)

def read_observations(filename):
    """Read observations in the user's format (UTC, RA=HHMMSSss, DEC=+DDMMSSss).
    Returns astropy.time.Time array, numpy arrays ras_deg, decs_deg, errs, mags.
    """
    times = []
    ras = []
    decs = []
    errs = []
    mags = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('СИСТ') or s.startswith('КОНЕЦ'):
                continue
            parts = s.split()
            if len(parts) < 5:
                continue
            date_str, time_str, ra_str, dec_str, val_str = parts[:5]
            yy = int(date_str[0:2])
            year = 2000 + yy
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            hh = int(time_str[0:2])
            mm = int(time_str[2:4])
            ss = int(time_str[4:6])
            frac = 0.0
            if len(time_str) > 6:
                tail = time_str[6:]
                frac = float('0.' + tail)
            timestr = f"{year:04d}-{month:02d}-{day:02d}T{hh:02d}:{mm:02d}:{ss:02d}"
            if frac:
                timestr += f"{frac:.6f}"[1:]
            t = Time(timestr, format='isot', scale='utc')

            ra_deg = parse_hhmmss_ra(ra_str)
            dec_deg = parse_ddmmss_dec(dec_str)

            err = int(val_str[0:3])
            mag = int(val_str[3:]) / 10.0 if len(val_str) > 3 else 0.0

            times.append(t)
            ras.append(ra_deg)
            decs.append(dec_deg)
            errs.append(err)
            mags.append(mag)
    return Time(times), np.array(ras), np.array(decs), np.array(errs), np.array(mags)

def read_tle_lines(line1, line2):
    return line1.strip(), line2.strip()
