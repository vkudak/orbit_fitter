import csv
import numpy as np
from sgp4.api import Satrec, jday
from astropy.time import Time
from astropy.coordinates import TEME, GCRS, ICRS, CartesianRepresentation
import astropy.units as u
import warnings

warnings.filterwarnings("ignore", category=UserWarning, append=True)

def propagate_tle_to_radec(tle1, tle2, times):
    """Прогноз координат RA/DEC з TLE на задані моменти часу."""
    sat = Satrec.twoline2rv(tle1, tle2)

    ras, decs = [], []
    for t in times:
        # перетворення часу у юліанську добу
        jd, fr = jday(t.datetime.year, t.datetime.month, t.datetime.day,
                      t.datetime.hour, t.datetime.minute, t.datetime.second + t.datetime.microsecond * 1e-6)
        e, r, v = sat.sgp4(jd, fr)
        if e != 0:
            ras.append(np.nan)
            decs.append(np.nan)
            continue

        try:
            # TEME → GCRS → ICRS
            rep = CartesianRepresentation(r[0] * u.km, r[1] * u.km, r[2] * u.km)
            teme = TEME(rep, obstime=t)
            gcrs = teme.transform_to(GCRS(obstime=t))
            icrs = gcrs.transform_to(ICRS())
            ras.append(icrs.ra.deg)
            decs.append(icrs.dec.deg)
        except Exception:
            ras.append(np.nan)
            decs.append(np.nan)

    return np.array(ras), np.array(decs)


def compare_with_tle(obs, state, tle_line1, tle_line2, out_csv=None):
    """Порівнює спостереження з прогнозом TLE."""
    times, ras_obs, decs_obs, errs, mags, _ = obs

    # Перевірка формату часу
    if not isinstance(times[0], Time):
        times = Time(times)

    # Прогноз координат TLE
    ras_tle, decs_tle = propagate_tle_to_radec(tle_line1, tle_line2, times)

    # Обчислення різниць у дугових секундах
    dra = (ras_tle - ras_obs) * 3600.0 * np.cos(np.deg2rad(decs_obs))
    ddec = (decs_tle - decs_obs) * 3600.0

    valid = np.isfinite(dra) & np.isfinite(ddec)
    if np.count_nonzero(valid) == 0:
        print("⚠️  Немає спільних валідних точок для порівняння з TLE.")
        return np.nan

    rms = np.sqrt(np.nanmean(np.hstack((dra[valid], ddec[valid])) ** 2))
    print(f"TLE comparison RMS (arcsec): {rms:.3f}")

    # Запис у CSV (опційно)
    if out_csv:
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['time_iso', 'obs_ra_deg', 'obs_dec_deg',
                        'tle_ra_deg', 'tle_dec_deg', 'dra_arcsec', 'ddec_arcsec'])
            for i in range(len(times)):
                w.writerow([
                    times[i].iso, ras_obs[i], decs_obs[i],
                    ras_tle[i], decs_tle[i], dra[i], ddec[i]
                ])

    return rms


def elements_to_tle(elements, epoch_time, sat_name='SAT'):
    """
    NOT WORKING!!!!!
    Генерує TLE для Satrec на основі орбітальних елементів.
    elements: словник з a, e, i, raan, argp, nu
    epoch_time: astropy Time об'єкт
    """
    # Переводимо у фізичні величини для TLE
    # a у km -> n (mean motion, rev/day)
    mu = 398600.4418  # km^3/s^2
    a = elements['a']
    n = np.sqrt(mu / a**3) * 86400.0 / (2.0 * np.pi)  # rev/day

    # Створюємо "порожню" Satrec і заповнюємо через twoline2rv
    sat = Satrec()
    # Використовуємо позиційні параметри для TLE
    sat.sgp4init(
        Satrec.wgs72,       # gravity model
        'i',                # type
        99999,              # satnum
        epoch_time.jd,      # epoch
        0.0,                # bstar
        0.0,                # ndot
        0.0,                # nddot
        elements['e'],      # eccentricity
        np.radians(elements['argp']),   # argpo
        np.radians(elements['i']),      # inclo
        np.radians(elements['nu']),     # mo
        n * 2*np.pi / 86400.0,          # no_kozai
        np.radians(elements['raan'])    # nodeo
    )

    # Генеруємо рядки TLE
    line1, line2 = sat.sgp4_tle()
    return line1, line2

def elements_to_tle_manual(elements, satnum=99999, epoch=0.0, name='SAT'):
    """
    Проста генерація TLE рядків вручну на основі орбітальних елементів.
    elements: словник з a(km), e, i(deg), raan(deg), argp(deg), nu(deg)
    epoch: astropy Time JD (можна epoch.jd)
    """
    a = elements['a']           # km
    e = elements['e']
    i = elements['i']           # deg
    raan = elements['raan']     # deg
    argp = elements['argp']     # deg
    nu = elements['nu']         # deg

    mu = 398600.4418            # km^3/s^2
    # Mean motion [rev/day]
    n = np.sqrt(mu / a**3) * 86400.0 / (2*np.pi)

    # Eccentricity для TLE у форматі 7 знаків без децимальної крапки
    e_tle = int(e * 1e7)

    # Line 1 (заглушка, мінімальний обов'язковий формат)
    line1 = f"1 {satnum:05d}U 00000A   {epoch:10.8f}  .00000000  00000-0  00000-0 0  9990"

    # Line 2 (елементи орбіти)
    line2 = (
        f"2 {satnum:05d} {i:8.4f} {raan:8.4f} {e_tle:07d} {argp:8.4f} "
        f"{nu:8.4f} {n:11.8f}00000"
    )
    return line1, line2