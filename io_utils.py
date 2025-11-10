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

def read_observations_old(filename):
    """Read observations in the user's format (UTC, RA=HHMMSSss, DEC=+DDMMSSss).
    Returns astropy.time.Time array, numpy arrays ras_deg, decs_deg, errs, mags.
    """
    times = []
    ras = []
    decs = []
    errs = []
    mags = []
    objects_data = {}

    with (open(filename, 'r', encoding='utf-8') as f):  # windows-1251
        current_object = None
        current_data = []
        current_point_number = None  # Номер пункту
        for line in f:
            s = line.strip()
            if len(s.split()) == 3: # CИСТ 10092 023855
                current_object = int(s.split()[2])
                current_point_number = int(s.split()[1])
            if len(s.split()) == 5: # data
                parts = s.split()

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

            if len(s)==5: # END
                objects_data[current_object] = [
                    Time(times),
                    np.array(ras), np.array(decs),
                    np.array(errs), np.array(mags),
                    current_point_number
                ]

    return objects_data

def _safe_decode(line_bytes):
    """Try UTF-8, then CP1251, else decode with replacement."""
    try:
        return line_bytes.decode('utf-8').strip()
    except UnicodeDecodeError:
        try:
            return line_bytes.decode('cp1251').strip()
        except UnicodeDecodeError:
            return line_bytes.decode('utf-8', errors='replace').strip()


def _append_times(existing_time, new_times):
    """Concatenate two astropy Time objects preserving scale/format."""
    if existing_time is None:
        return new_times
    combined_unix = np.concatenate([existing_time.unix, new_times.unix])
    return Time(combined_unix, format='unix', scale='utc')


def add_to_objects(objects_data, obj_id, times_list, ras, decs, errs, mags, point_number):
    """Create or append measurements for obj_id.
       point_number is applied to all new entries.
    """
    n_new = len(times_list)
    if n_new == 0:
        return

    new_times = Time(times_list)
    ras = np.array(ras)
    decs = np.array(decs)
    errs = np.array(errs)
    mags = np.array(mags)
    points = np.full(n_new, point_number, dtype=int)

    if obj_id not in objects_data:
        objects_data[obj_id] = {
            "time": new_times,
            "ra": ras,
            "dec": decs,
            "err": errs,
            "mag": mags,
            "point": points
        }
    else:
        existing = objects_data[obj_id]
        existing["time"] = _append_times(existing["time"], new_times)
        existing["ra"] = np.concatenate([existing["ra"], ras])
        existing["dec"] = np.concatenate([existing["dec"], decs])
        existing["err"] = np.concatenate([existing["err"], errs])
        existing["mag"] = np.concatenate([existing["mag"], mags])
        existing["point"] = np.concatenate([existing["point"], points])


def read_observations(filename):
    """Read observations in the user's format (UTC, RA=HHMMSSss, DEC=+DDMMSSss).
    Returns dict:
      {
        object_id: {
          "time": astropy.Time,
          "ra": np.array,
          "dec": np.array,
          "err": np.array,
          "mag": np.array,
          "point": np.array
        }
      }
    """
    objects_data = {}
    current_object = None
    current_point_number = None
    times, ras, decs, errs, mags = [], [], [], [], []

    with open(filename, 'rb') as f:
        for line_bytes in f:
            s = _safe_decode(line_bytes)
            if not s:
                continue
            parts = s.split()

            # Початок блоку
            if len(parts) == 3:
                if current_object is not None and len(times) > 0:
                    add_to_objects(objects_data, current_object, times, ras, decs, errs, mags, current_point_number)
                    times, ras, decs, errs, mags = [], [], [], [], []

                current_point_number = int(parts[1])
                try:
                    current_object = int(parts[2])
                except ValueError:
                    current_object = None
                    current_point_number = None

            # Рядок з виміром
            elif len(parts) >= 5:
                date_str, time_str, ra_str, dec_str, val_str = parts[:5]
                try:
                    yy = int(date_str[0:2])
                    year = 2000 + yy
                    month = int(date_str[2:4])
                    day = int(date_str[4:6])
                    hh = int(time_str[0:2])
                    mm = int(time_str[2:4])
                    ss = int(time_str[4:6])
                    frac = 0.0
                    if len(time_str) > 6 and time_str[6:].isdigit():
                        frac = float('0.' + time_str[6:])
                    timestr = f"{year:04d}-{month:02d}-{day:02d}T{hh:02d}:{mm:02d}:{ss:02d}"
                    if frac:
                        timestr += f"{frac:.6f}"[1:]
                    t = Time(timestr, format='isot', scale='utc')

                    ra_deg = parse_hhmmss_ra(ra_str)
                    dec_deg = parse_ddmmss_dec(dec_str)
                    err = int(val_str[0:3]) if val_str[:3].isdigit() else 0
                    mag = int(val_str[3:]) / 10.0 if len(val_str) > 3 and val_str[3:].lstrip('+-').isdigit() else 0.0

                    times.append(t)
                    ras.append(ra_deg)
                    decs.append(dec_deg)
                    errs.append(err)
                    mags.append(mag)
                except Exception:
                    continue

            # Кінець блоку
            elif len(s) == 5:
                if current_object is not None and len(times) > 0:
                    add_to_objects(objects_data, current_object, times, ras, decs, errs, mags, current_point_number)
                    times, ras, decs, errs, mags = [], [], [], [], []
                current_object = None
                current_point_number = None

    # Дозапис після кінця файлу
    if current_object is not None and len(times) > 0:
        add_to_objects(objects_data, current_object, times, ras, decs, errs, mags, current_point_number)

    return objects_data

def read_tle_lines(line1, line2):
    return line1.strip(), line2.strip()
